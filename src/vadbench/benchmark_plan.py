"""把版本化 benchmark YAML 编排为逐 case 加载的真实性能运行。"""

from __future__ import annotations

import gc
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.benchmark import (
    PERFORMANCE_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkSettings,
    BenchmarkWorkload,
    assess_sampling_comparability,
    run_benchmark_suite,
    write_performance_result,
)
from vadbench.config import load_experiment, load_yaml
from vadbench.contracts import ClipBatch
from vadbench.data.video import VideoInfo, decode_rgb_frames, probe_video
from vadbench.orchestration import compression_from_experiment, create_encoder_from_experiment


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sample_indices(info: VideoInfo, sample_fps: float, sampled_frames: int) -> np.ndarray:
    if sample_fps <= 0 or sampled_frames <= 0:
        raise ValueError("sample_fps 与 sampled_frames 必须大于 0")
    stride = max(1, int(round(info.fps / sample_fps)))
    indices = np.arange(sampled_frames, dtype=np.int64) * stride
    if int(indices[-1]) >= info.num_frames:
        raise ValueError(
            f"视频只有 {info.num_frames} 帧，无法以 stride={stride} 采 {sampled_frames} 帧"
        )
    return indices


def _batch_groups(
    decoded: np.ndarray,
    *,
    indices: np.ndarray,
    fps: float,
    video_id: str,
    mode: str,
    units: int,
    frames_per_unit: int,
) -> ClipBatch | tuple[ClipBatch, ...]:
    expected = units * frames_per_unit
    if decoded.shape[0] != expected or indices.shape[0] != expected:
        raise ValueError(f"预处理期望 {expected} 帧，实际 {decoded.shape[0]}/{indices.shape[0]}")
    frames = decoded.reshape(units, frames_per_unit, *decoded.shape[1:])
    frame_indices = indices.reshape(units, frames_per_unit)
    timestamps = frame_indices.astype(np.float64) / float(fps)

    def one(start: int, stop: int) -> ClipBatch:
        clip_indices = list(range(start, stop))
        return ClipBatch(
            frames=frames[start:stop],
            timestamps_s=timestamps[start:stop],
            video_ids=(video_id,) * (stop - start),
            frame_indices=frame_indices[start:stop],
            metadata={
                "clip_ids": [f"{video_id}:benchmark-{index:04d}" for index in clip_indices],
                "clip_indices": clip_indices,
                "sampling_kind": "shared_ordered_frames",
            },
        )

    if mode == "fixed":
        return one(0, units)
    if mode == "streaming":
        return tuple(one(index, index + 1) for index in range(units))
    raise ValueError(f"未知 benchmark mode：{mode}")


def _case_experiment(
    case_spec: Mapping[str, Any], root: Path, *, device: str | None
) -> dict[str, Any]:
    experiment = load_experiment(_resolve(root, str(case_spec["experiment"])))
    encoder = dict(case_spec.get("encoder", {}))
    if device is not None:
        encoder["device"] = device
    mode = str(case_spec["mode"])
    streaming = {
        **dict(experiment.get("streaming", {})),
        "enabled": mode == "streaming",
    }
    if case_spec.get("compression") is not None:
        streaming["compression"] = dict(case_spec["compression"])
    return _deep_merge(experiment, {"encoder": encoder, "streaming": streaming})


def _require_case_result(result: Mapping[str, Any], requirements: Mapping[str, Any]) -> None:
    if not requirements:
        return
    repeats = result.get("repeats", [])
    if not repeats:
        raise RuntimeError(f"{result.get('name')}: benchmark 没有 repeat 结果")
    checks = {
        "native_compression_calls_min": "native_compression_calls",
        "native_compression_applied_steps_min": "native_compression_applied_steps",
    }
    for requirement, field in checks.items():
        if requirement not in requirements:
            continue
        observed = min(int(repeat["cache"][field]) for repeat in repeats)
        expected = int(requirements[requirement])
        if observed < expected:
            raise RuntimeError(
                f"{result.get('name')}: {field} 每 repeat 最小值 {observed} < {expected}"
            )


def _release_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover - optional dependency
        pass


def run_benchmark_plan(
    plan_path: str | Path,
    *,
    project_root: str | Path = ".",
    video: str | Path | None = None,
    device: str | None = None,
    warmup: int | None = None,
    repeat_count: int | None = None,
    output: str | Path | None = None,
    encoder_factory: Callable[..., tuple[Any, Mapping[str, Any]]] = create_encoder_from_experiment,
    probe_fn: Callable[..., VideoInfo] = probe_video,
    decode_fn: Callable[..., np.ndarray] = decode_rgb_frames,
) -> dict[str, Any]:
    """串行加载每个 encoder case，运行后立即释放，避免多模型同时占用 A100。"""

    root = Path(project_root).resolve()
    plan = load_yaml(_resolve(root, plan_path))
    benchmark = dict(plan.get("benchmark", {}))
    input_spec = dict(plan.get("input", {}))
    case_specs = list(plan.get("cases", ()))
    if not case_specs:
        raise ValueError("benchmark plan 至少需要一个 case")
    selected_video = _resolve(root, video or str(input_spec["video"]))
    info = probe_fn(selected_video)
    sample_fps = float(input_spec["sample_fps"])
    sampled_frames = int(input_spec["sampled_frames"])
    indices = _sample_indices(info, sample_fps, sampled_frames)
    actual_stride = int(indices[1] - indices[0]) if len(indices) > 1 else 1
    video_seconds = float((int(indices[-1]) - int(indices[0]) + actual_stride) / info.fps)

    settings = BenchmarkSettings(
        warmup=int(benchmark.get("warmup", 1) if warmup is None else warmup),
        repeat=int(benchmark.get("repeat", 5) if repeat_count is None else repeat_count),
        synchronize_cuda=bool(benchmark.get("synchronize_cuda", True)),
        device=device or benchmark.get("device"),
    )

    def decoded_factory() -> np.ndarray:
        return decode_fn(selected_video, indices)

    case_results: list[dict[str, Any]] = []
    machine: Mapping[str, Any] | None = None

    for raw_case in case_specs:
        case_spec = dict(raw_case)
        experiment = _case_experiment(case_spec, root, device=settings.device)
        adapter, definition = encoder_factory(experiment, project_root=root)
        grouping = dict(case_spec["grouping"])
        units = int(grouping["units"])
        frames_per_unit = int(grouping["frames_per_unit"])
        if units * frames_per_unit != sampled_frames:
            raise ValueError(f"{case_spec['name']}: grouping 与 sampled_frames 不一致")

        def preprocess(
            decoded: np.ndarray,
            *,
            mode: str = str(case_spec["mode"]),
            units_value: int = units,
            frames_value: int = frames_per_unit,
        ) -> ClipBatch | tuple[ClipBatch, ...]:
            return _batch_groups(
                np.asarray(decoded),
                indices=indices,
                fps=info.fps,
                video_id=selected_video.stem,
                mode=mode,
                units=units_value,
                frames_per_unit=frames_value,
            )

        workload = BenchmarkWorkload(
            name=str(input_spec.get("sampling_protocol", "benchmark")),
            mode=str(case_spec["mode"]),
            decode=decoded_factory,
            preprocess=preprocess,
            sampling={
                **input_spec,
                "source_fps": info.fps,
                "actual_frame_stride": actual_stride,
                "source_video_path": str(selected_video),
            },
            video_seconds=video_seconds,
            task=str(input_spec.get("task", "encoder_performance_only")),
        )
        compression = (
            compression_from_experiment(experiment) if case_spec["mode"] == "streaming" else None
        )
        case = BenchmarkCase(
            name=str(case_spec["name"]),
            adapter=adapter,
            workload=workload,
            compression=compression,
            config={
                "case": case_spec,
                "experiment": experiment,
                "encoder_definition": definition,
            },
        )
        try:
            single = run_benchmark_suite((case,), settings)
            case_result = single["cases"][0]
            _require_case_result(case_result, dict(case_spec.get("result_requirements", {})))
            case_results.append(case_result)
            machine = single["provenance"]["machine"]
        finally:
            del case
            del adapter
            _release_runtime()

    result = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "comparison": assess_sampling_comparability(case_results),
        "provenance": {
            "machine": dict(machine or {"hostname": platform.node()}),
            "plan": str(_resolve(root, plan_path)),
        },
        "cases": case_results,
    }
    destination = _resolve(root, output or str(benchmark["output"]))
    write_performance_result(result, destination)
    return result


__all__ = ["run_benchmark_plan"]
