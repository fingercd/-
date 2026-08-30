"""真实权重 encoder 的单视频冒烟执行器。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from vadbench.data.manifest import VideoManifestRecord
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import (
    build_clip_batch,
    iter_streaming_chunk_batches,
    probe_video,
)
from vadbench.orchestration import compression_from_experiment, create_encoder_from_experiment


def _shape(value: Any | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value.shape]


def _dtype(value: Any | None) -> str | None:
    return None if value is None else str(getattr(value, "dtype", "unknown"))


def _gpu_peak_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:  # pragma: no cover - optional dependency
        pass
    return None


def run_encoder_smoke(
    config: Mapping[str, Any],
    video_path: str | Path,
    *,
    project_root: str | Path = ".",
    max_chunks: int = 2,
) -> dict[str, Any]:
    """加载真实 adapter/权重并执行固定 clip 或连续 chunk 前向。"""

    if max_chunks <= 0:
        raise ValueError("max_chunks 必须大于 0")
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    info = probe_video(path)
    adapter, definition = create_encoder_from_experiment(config, project_root=project_root)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:  # pragma: no cover
        pass

    started = time.perf_counter()
    streaming = config.get("streaming", {})
    if streaming.get("enabled", False):
        record = VideoManifestRecord(
            video_id=path.stem,
            path=path.name,
            split="test",
            category="Smoke",
            is_anomaly=False,
            num_frames=info.num_frames,
            fps=info.fps,
            duration_seconds=info.duration_seconds,
            metadata={"smoke_input": True},
        )
        chunks = iter_streaming_chunk_batches(
            (record,),
            path.parent,
            chunk_frames=int(streaming.get("chunk_frames", 16)),
            sample_fps=(
                float(streaming["sample_fps"]) if streaming.get("sample_fps") is not None else None
            ),
        )
        state = adapter.init_state(record.video_id)
        policy = compression_from_experiment(config)
        steps: list[dict[str, Any]] = []
        for chunk in islice(chunks, max_chunks):
            step = adapter.encode_step(chunk, state, train=False, compression=policy)
            state = step.state
            steps.append(
                {
                    "step_index": state.step_index,
                    "features_shape": _shape(None if step.output is None else step.output.features),
                    "features_dtype": _dtype(None if step.output is None else step.output.features),
                    "pooled_shape": _shape(None if step.output is None else step.output.pooled),
                    "cache_layers": len(state.caches),
                    "cache_tokens_max": max(
                        (view.sequence_length for view in state.caches.values()), default=0
                    ),
                    "telemetry": dict(step.telemetry),
                }
            )
        adapter.finalize(state)
        result: dict[str, Any] = {
            "mode": "streaming",
            "steps": steps,
            "chunks_requested": max_chunks,
            "chunks_completed": len(steps),
        }
    else:
        frames = adapter.capabilities.fixed_num_frames
        if frames is None:
            raise ValueError("fixed adapter 未声明 fixed_num_frames")
        sampler = config.get("sampler", {})
        sample = sample_fixed_clip(
            info.num_frames,
            clip_frames=frames,
            frame_stride=int(sampler.get("frame_stride", 1)),
            position="center",
        )
        batch = build_clip_batch(
            path,
            path.stem,
            (sample,),
            metadata={"clip_ids": [f"{path.stem}:smoke"], "clip_indices": [0]},
        )
        output = adapter.encode(batch, train=False)
        result = {
            "mode": "fixed",
            "features_shape": _shape(output.features),
            "features_dtype": _dtype(output.features),
            "pooled_shape": _shape(output.pooled),
            "timeline_tokens": output.timeline.num_tokens,
            "aux": dict(output.aux),
        }

    result.update(
        {
            "adapter": config["encoder"]["adapter"],
            "encoder_definition": definition.get("name"),
            "video": {
                "path": str(path),
                "num_frames": info.num_frames,
                "fps": info.fps,
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": _gpu_peak_bytes(),
        }
    )
    return result


def write_smoke_result(result: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = ["run_encoder_smoke", "write_smoke_result"]
