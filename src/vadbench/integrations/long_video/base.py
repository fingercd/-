"""Dependency-isolated adapters for long-video and video-VLM integrations.

The upstream projects covered by this package do not share a Python API.  This
module provides the deliberately small bridge that they do share: a canonical
``ClipBatch`` (``BTHWC``/``uint8``) at the input boundary and
``EncoderOutput``/``StreamStep`` at the output boundary.  Heavy dependencies are
never imported at module import time.  A caller may inject a worker/model for a
local smoke test, or provide an explicit loader/command for a pinned upstream
checkout.  The adapter never downloads assets implicitly.

The classes in the sibling modules are thin declarations of integration id,
feature stage, and capabilities.  Keeping the implementation here means that
new VLMs can be added without copying subtly different timeline and state
handling code.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from vadbench.contracts import (
    CacheKind,
    CacheUpdate,
    CacheView,
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    StreamingVideoEncoderAdapter,
    StreamState,
    StreamStep,
    TokenTimeline,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
    validate_stream_step,
)
from vadbench.integrations.common import (
    normalize_encoder_output,
    select_feature_tensor,
    validate_output_health,
)

DEFAULT_NEUTRAL_PROMPT = (
    "Describe the visible video content, actions, objects, and temporal changes "
    "without assuming any event label."
)
"""A category-independent prompt shared by prompt-based VLM adapters."""

_CACHE_MODES = frozenset({"off", "identity"})
_MISSING = object()


class LongVideoIntegrationError(RuntimeError):
    """Base class for structured long-video integration failures."""


class LongVideoAssetError(FileNotFoundError, LongVideoIntegrationError):
    """Structured, fail-closed error for absent or unconfigured upstream assets.

    ``FileNotFoundError`` is retained as the base class so ordinary callers can
    handle missing checkouts in the usual way.  ``to_dict`` is used by matrix and
    smoke runners to persist a machine-readable failure rather than replacing a
    missing model with a random/mock implementation.
    """

    def __init__(
        self,
        *,
        integration_id: str,
        code: str = "missing_asset",
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.integration_id = integration_id
        self.code = code
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "integration_id": self.integration_id,
            "message": self.message,
            "details": _jsonable(self.details),
        }


# Explicit aliases make the error discoverable for callers that use either
# terminology in their matrix code.
MissingLongVideoAssetError = LongVideoAssetError
MissingAssetError = LongVideoAssetError
ExternalAssetError = LongVideoAssetError


class LongVideoWorkerError(LongVideoIntegrationError):
    """A worker process/model returned an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        integration_id: str,
        code: str = "worker_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.integration_id = integration_id
        self.code = code
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "integration_id": self.integration_id,
            "message": self.message,
            "details": _jsonable(self.details),
        }


ExternalWorkerError = LongVideoWorkerError
StructuredLongVideoError = LongVideoIntegrationError


def _jsonable(value: Any) -> Any:
    """Convert small worker metadata (and optional arrays) to JSON values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, StreamState):
        return {
            "video_id": value.video_id,
            "step_index": value.step_index,
            "opaque": _jsonable(value.opaque),
            "next_timestamp_s": value.next_timestamp_s,
            "metadata": _jsonable(value.metadata),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - unusual torch-like object
            raise TypeError(f"无法序列化 worker tensor: {type(value).__name__}") from exc
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    # Opaque stream state is intentionally not sent to a subprocess unless the
    # upstream worker makes it JSON-safe.  Raising here prevents silent state
    # loss between chunks.
    raise TypeError(f"worker metadata 含不可序列化类型 {type(value).__name__}")


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[4]
    except IndexError:  # pragma: no cover - only for exotic import layouts
        return Path.cwd().resolve()


def _resolve_path(value: str | os.PathLike[str] | None, root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _path_text(value: str | os.PathLike[str] | None, root: Path) -> str | None:
    path = _resolve_path(value, root)
    return None if path is None else str(path)


def _normalise_cache_mode(value: str | None) -> str:
    if value is None:
        return "identity"
    text = str(value).strip().lower().replace("-", "_")
    aliases = {"none": "off", "disabled": "off", "false": "off", "true": "identity"}
    text = aliases.get(text, text)
    if text not in _CACHE_MODES:
        raise ValueError(
            f"{value!r} 不是允许的缓存模式；本接入只支持 off 或 identity，不执行 KV 压缩"
        )
    return text


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def _next_timestamp_for_chunk(chunk: ClipBatch, previous: float | None = None) -> float:
    valid_length = int(chunk.valid_lengths[0])
    times = _to_numpy(chunk.timestamps_s)[0, :valid_length]
    next_time = _finite_float(times[-1], 0.0)
    if valid_length > 1:
        positive = np.diff(times)
        positive = positive[positive > 0]
        if positive.size:
            next_time += float(np.median(positive))
    if previous is not None and next_time <= previous:
        next_time = previous + 1e-6
    return next_time


def _to_numpy(value: Any) -> np.ndarray:
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if str(getattr(tensor, "dtype", "")).lower() in {"torch.bfloat16", "bfloat16"}:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raw = np.asarray(value).shape
    return tuple(int(item) for item in raw)


def _timeline_for_batch(batch: ClipBatch, token_count: int) -> TokenTimeline:
    """Build a monotonic, explicit approximate token-to-frame timeline."""

    if type(token_count) is not int or token_count <= 0:
        raise ValueError(f"token_count 必须是正整数，实际为 {token_count!r}")
    timestamps = _to_numpy(batch.timestamps_s).astype(np.float64, copy=False)
    if timestamps.ndim != 2:
        raise ValueError("batch.timestamps_s 必须是二维数组")
    lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
    starts = np.zeros((batch.batch_size, token_count), dtype=np.float64)
    ends = np.zeros_like(starts)
    source_start = np.zeros((batch.batch_size, token_count), dtype=np.int64)
    source_end = np.ones_like(source_start)
    for row in range(batch.batch_size):
        frame_count = int(lengths[row])
        if frame_count <= 0:
            raise ValueError("每个 ClipBatch 样本至少需要一帧有效帧")
        times = timestamps[row, :frame_count]
        slots = np.minimum(
            (np.arange(token_count, dtype=np.int64) * frame_count) // token_count,
            frame_count - 1,
        )
        starts[row] = times[slots]
        if frame_count > 1:
            diffs = np.diff(times)
            positive = diffs[diffs > 0]
            duration = float(np.median(positive)) if positive.size else 0.0
        else:
            duration = 0.0
        frame_ends = np.empty(frame_count, dtype=np.float64)
        if frame_count > 1:
            frame_ends[:-1] = times[1:]
        frame_ends[-1] = times[-1] + duration
        ends[row] = frame_ends[slots]
        if batch.frame_indices is None:
            indices = np.arange(frame_count, dtype=np.int64)
        else:
            indices = _to_numpy(batch.frame_indices)[row, :frame_count].astype(np.int64, copy=False)
        source_start[row] = indices[slots]
        source_end[row] = source_start[row] + 1
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=source_start,
        source_frame_end=source_end,
    )


# Public spelling for custom upstream adapters that want to reuse the exact
# timeline policy without depending on a private helper name.
build_token_timeline = _timeline_for_batch


def _timeline_from_value(value: Any, batch: ClipBatch, token_count: int) -> TokenTimeline | None:
    if isinstance(value, TokenTimeline):
        return value
    if not isinstance(value, Mapping):
        return None
    starts = value.get("start_s", value.get("start_seconds"))
    ends = value.get("end_s", value.get("end_seconds"))
    if starts is None or ends is None:
        return None
    return TokenTimeline(
        start_s=np.asarray(starts),
        end_s=np.asarray(ends),
        valid_mask=None if value.get("valid_mask") is None else np.asarray(value["valid_mask"]),
        source_frame_start=(
            None
            if value.get("source_frame_start") is None
            else np.asarray(value["source_frame_start"])
        ),
        source_frame_end=(
            None if value.get("source_frame_end") is None else np.asarray(value["source_frame_end"])
        ),
    )


def _with_canonical_aux(
    output: EncoderOutput,
    *,
    integration_id: str,
    feature_stage: str,
    prompt: str,
    cache_mode: str,
    extra: Mapping[str, Any] | None = None,
) -> EncoderOutput:
    aux = dict(output.aux)
    aux.update(
        {
            "integration_id": integration_id,
            "feature_stage": feature_stage,
            "prompt": prompt,
            "cache_mode": cache_mode,
            "implementation_source": "external_worker_facade",
        }
    )
    if extra:
        aux.update(extra)
    return EncoderOutput(
        features=output.features,
        pooled=output.pooled,
        timeline=output.timeline,
        aux=aux,
    )


def _validate_json_aux(output: EncoderOutput, integration_id: str) -> None:
    try:
        json.dumps(_jsonable(dict(output.aux)), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LongVideoWorkerError(
            f"{integration_id} worker aux 含不可序列化值: {exc}",
            integration_id=integration_id,
            code="invalid_metadata",
        ) from exc


def _normalise_output(
    raw: Any,
    batch: ClipBatch,
    *,
    integration_id: str,
    feature_stage: str,
    prompt: str,
    cache_mode: str,
    preprocess_profile: str = "bthwc_uint8",
) -> EncoderOutput:
    """Normalize tensors, model-output objects, and worker mappings."""

    if isinstance(raw, EncoderOutput):
        validate_encoder_output(raw, batch)
        canonical = _with_canonical_aux(
            raw,
            integration_id=integration_id,
            feature_stage=feature_stage,
            prompt=prompt,
            cache_mode=cache_mode,
        )
        try:
            validate_output_health(canonical)
            _validate_json_aux(canonical, integration_id)
        except Exception as exc:
            raise LongVideoWorkerError(
                f"{integration_id} worker 输出包含非有限值或非法时间轴: {exc}",
                integration_id=integration_id,
                code="invalid_output_health",
            ) from exc
        return canonical

    payload = raw
    supplied_pooled: Any | None = None
    supplied_timeline: Any | None = None
    supplied_aux: Mapping[str, Any] | None = None
    if isinstance(raw, Mapping):
        supplied_timeline = raw.get("timeline")
        supplied_pooled = raw.get("pooled", raw.get("pooler_output"))
        candidate = raw.get("output", _MISSING)
        if candidate is not _MISSING:
            payload = candidate
        elif "features" in raw:
            payload = raw
        supplied_aux = raw.get("aux") if isinstance(raw.get("aux"), Mapping) else None

    try:
        selected, _ = select_feature_tensor(payload, batch_size=batch.batch_size)
    except Exception as exc:
        raise LongVideoWorkerError(
            f"{integration_id} worker 未返回可识别的特征张量: {exc}",
            integration_id=integration_id,
            code="invalid_output",
        ) from exc
    # JSON subprocess workers naturally return nested lists.  The common
    # normalizer intentionally preserves backend tensors by identity, so make
    # list candidates explicit arrays here before handing them over.
    if not hasattr(selected, "shape"):
        selected = np.asarray(selected)
        if isinstance(payload, Mapping):
            payload = dict(payload)
            replaced = False
            for field_name in (
                "features",
                "output",
                "last_hidden_state",
                "video_features",
                "visual_features",
                "projected_visual",
                "visual_memory",
                "decoder_contextual",
            ):
                if field_name in payload and not hasattr(payload[field_name], "shape"):
                    payload[field_name] = selected
                    replaced = True
                    break
            if not replaced:
                # Nested ``hidden_states``/tuple outputs are made explicit so
                # the dependency-free normalizer cannot accidentally preserve a
                # Python list as the public feature tensor.
                payload = {"features": selected}
        else:
            payload = selected
    if supplied_pooled is not None and not hasattr(supplied_pooled, "shape"):
        supplied_pooled = np.asarray(supplied_pooled)
    selected_shape = _shape(selected)
    token_count = 1 if len(selected_shape) == 2 else selected_shape[1]
    timeline = _timeline_from_value(supplied_timeline, batch, token_count)
    if timeline is None:
        timeline = _timeline_for_batch(batch, token_count)
    try:
        output = normalize_encoder_output(
            payload,
            timeline=timeline,
            feature_stage=feature_stage,
            pooled=supplied_pooled,
            sequence_source=("worker_output" if isinstance(raw, Mapping) else "upstream_output"),
            preprocess_profile=preprocess_profile,
            aux=supplied_aux,
        )
    except Exception as exc:
        raise LongVideoWorkerError(
            f"{integration_id} worker 输出不满足 EncoderOutput 契约: {exc}",
            integration_id=integration_id,
            code="invalid_output",
        ) from exc
    canonical = _with_canonical_aux(
        output,
        integration_id=integration_id,
        feature_stage=feature_stage,
        prompt=prompt,
        cache_mode=cache_mode,
    )
    try:
        validate_output_health(canonical)
        _validate_json_aux(canonical, integration_id)
    except Exception as exc:
        raise LongVideoWorkerError(
            f"{integration_id} worker 输出包含非有限值或非法时间轴: {exc}",
            integration_id=integration_id,
            code="invalid_output_health",
        ) from exc
    return canonical


normalize_long_video_output = _normalise_output


def _filtered_kwargs(function: Callable[..., Any], values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(values)
    return {key: value for key, value in values.items() if key in parameters}


def _invoke_candidates(
    function: Callable[..., Any], candidates: Sequence[tuple[tuple[Any, ...], Mapping[str, Any]]]
) -> Any:
    """Bind before calling, so a TypeError inside upstream code is not hidden."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        args, kwargs = candidates[0]
        return function(*args, **dict(kwargs))
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **dict(kwargs))
        except TypeError:
            continue
        return function(*args, **dict(kwargs))
    args, kwargs = candidates[0]
    raise TypeError(
        f"无法匹配 upstream callable {getattr(function, '__name__', function)!r} 的参数；"
        f"签名为 {signature}"
    )


def _invoke_fixed(
    function: Callable[..., Any],
    batch: ClipBatch,
    *,
    prompt: str,
    stage: str,
    train: bool,
    processor: Any | None = None,
    device: str | None = None,
) -> Any:
    values = {
        "batch": batch,
        "clip": batch,
        "clip_batch": batch,
        "frames": batch.frames,
        "video_frames": batch.frames,
        "prompt": prompt,
        "question": prompt,
        "feature_stage": stage,
        "train": train,
        "processor": processor,
        "device": device,
    }
    kwargs = _filtered_kwargs(function, values)
    first_name = ""
    try:
        first = next(
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        )
        first_name = first.name.lower()
    except (IndexError, StopIteration, TypeError, ValueError):
        pass
    frame_first = first_name in {"frame", "frames", "images", "video", "video_frames"}
    candidates = (
        (((batch.frames,), kwargs), ((batch,), kwargs), ((), kwargs))
        if frame_first
        else (((batch,), kwargs), ((batch.frames,), kwargs), ((), kwargs))
    )
    return _invoke_candidates(function, candidates)


def _invoke_stream(
    function: Callable[..., Any],
    chunk: ClipBatch,
    state: StreamState,
    *,
    prompt: str,
    stage: str,
    compression: Any,
    processor: Any | None = None,
    device: str | None = None,
) -> Any:
    values = {
        "chunk": chunk,
        "batch": chunk,
        "clip": chunk,
        "frames": chunk.frames,
        "video_frames": chunk.frames,
        "state": state,
        "stream_state": state,
        "worker_state": state.opaque,
        "opaque": state.opaque,
        "prompt": prompt,
        "question": prompt,
        "feature_stage": stage,
        "compression": compression,
        "cache_policy": compression,
        "processor": processor,
        "device": device,
    }
    kwargs = _filtered_kwargs(function, values)
    first_name = ""
    second_name = ""
    try:
        positional = [
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        first = positional[0]
        first_name = first.name.lower()
        if len(positional) > 1:
            second_name = positional[1].name.lower()
    except (IndexError, StopIteration, TypeError, ValueError):
        pass
    frame_first = first_name in {"frame", "frames", "images", "video", "video_frames"}
    opaque_second = second_name in {"worker_state", "opaque", "cache_state", "backend_state"}
    if isinstance(getattr(function, "__self__", None), ExternalPythonWorker):
        opaque_second = True
    state_pair = (chunk, state.opaque) if opaque_second else (chunk, state)
    if frame_first:
        candidates = (
            ((chunk.frames, state_pair[1]), kwargs),
            ((chunk, state_pair[1]), kwargs),
            ((chunk.frames, state), kwargs),
            ((chunk, state), kwargs),
            ((chunk.frames,), kwargs),
            ((chunk,), kwargs),
            ((), kwargs),
        )
    else:
        candidates = (
            ((chunk, state_pair[1]), kwargs),
            ((chunk.frames, state), kwargs),
            ((chunk.frames, state.opaque), kwargs),
            ((chunk,), kwargs),
            ((), kwargs),
        )
    return _invoke_candidates(function, candidates)


def _invoke_loader(
    function: Callable[..., Any],
    *,
    checkout_path: Path | None,
    model_path: Path | None,
    device: str,
    processor: Any | None,
    prompt: str,
) -> Any:
    values = {
        "checkout_path": None if checkout_path is None else str(checkout_path),
        "model_path": None if model_path is None else str(model_path),
        "checkpoint_path": None if model_path is None else str(model_path),
        "device": device,
        "processor": processor,
        "prompt": prompt,
    }
    kwargs = _filtered_kwargs(function, values)
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = [((), kwargs)]
    if model_path is not None:
        candidates.append(((str(model_path),), {}))
    if checkout_path is not None and model_path is not None:
        candidates.append(((str(checkout_path), str(model_path)), {}))
    return _invoke_candidates(function, tuple(candidates))


def _load_entrypoint(checkout_path: Path, entrypoint: str, loader_kwargs: Mapping[str, Any]) -> Any:
    """Load an explicit ``relative_file.py:function`` without mutating sys.path."""

    file_name, separator, function_name = str(entrypoint).partition(":")
    if not separator or not file_name or not function_name:
        raise ValueError("entrypoint 必须采用 relative_file.py:function 格式")
    source = (checkout_path / file_name).resolve()
    try:
        source.relative_to(checkout_path.resolve())
    except ValueError as exc:
        raise ValueError("entrypoint 不能越出 checkout_path") from exc
    if not source.is_file():
        raise FileNotFoundError(f"upstream entrypoint 不存在：{source}")
    module_name = f"vadbench_external_{source.stem}_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法建立 upstream module spec：{source}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    target: Any = module
    for part in function_name.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"upstream entrypoint 不是 callable：{entrypoint}")
    return _invoke_loader(target, **loader_kwargs)


def _unpack_loaded_worker(loaded: Any, processor: Any | None) -> tuple[Any, Any | None]:
    """Pick a model from common upstream ``(tokenizer, model, processor, ...)`` returns."""

    if isinstance(loaded, Mapping) and "model" in loaded:
        selected_processor = loaded.get("processor", processor)
        return loaded["model"], selected_processor
    if isinstance(loaded, (tuple, list)):
        # The frequent two-item ``(model, processor)`` form is unambiguous.
        if len(loaded) == 2 and (
            callable(getattr(loaded[0], "encode", None)) or callable(loaded[0])
        ):
            return loaded[0], loaded[1] if processor is None else processor
        model_candidate: Any | None = None
        processor_candidate = processor
        for item in loaded:
            if callable(getattr(item, "encode", None)):
                model_candidate = item
                break
        if model_candidate is None:
            for item in loaded:
                if callable(item) and not isinstance(item, str):
                    model_candidate = item
                    break
        if processor_candidate is None:
            for item in loaded:
                if hasattr(item, "preprocess"):
                    processor_candidate = item
                    break
        if model_candidate is not None:
            return model_candidate, processor_candidate
    return loaded, processor


class ExternalPythonWorker:
    """Small no-network subprocess facade for an isolated upstream runtime.

    The command receives one JSON request on stdin and returns one JSON object
    on stdout.  Frames are represented as lists only for the small smoke chunks;
    production callers can provide a custom ``runner`` that exchanges NPY/NPZ
    sidecars through the project's worker protocol.  No shell is involved and
    no package/model download is attempted.
    """

    protocol_version = "vadbench.external-worker.v1"

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        integration_id: str,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 300.0,
        runner: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        if isinstance(command, str):
            command_parts = tuple(shlex.split(command))
        else:
            command_parts = tuple(str(item) for item in command)
        if not command_parts and runner is None:
            raise LongVideoAssetError(
                integration_id=integration_id,
                code="worker_unconfigured",
                message=f"{integration_id} 未配置 external worker command",
            )
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须大于 0")
        self.command = command_parts
        self.integration_id = integration_id
        self.cwd = None if cwd is None else str(Path(cwd).expanduser())
        self.env = dict(env or {})
        self.timeout_s = float(timeout_s)
        self.runner = runner

    def _request(self, operation: str, **payload: Any) -> Any:
        request = {
            "schema_version": 1,
            "protocol": self.protocol_version,
            "integration_id": self.integration_id,
            "operation": operation,
            **payload,
        }
        if self.runner is not None:
            try:
                response = self.runner(request)
            except Exception as exc:
                raise LongVideoWorkerError(
                    f"external runner 执行失败: {exc}",
                    integration_id=self.integration_id,
                    code="worker_execution_failed",
                ) from exc
            if isinstance(response, Mapping):
                if response.get("status", "ok") in {"failed", "error"}:
                    raise LongVideoWorkerError(
                        str(response.get("message", "external runner 报告失败")),
                        integration_id=self.integration_id,
                        code=str(response.get("code", "worker_execution_failed")),
                        details=response,
                    )
                return response.get("result", response)
            return response
        try:
            encoded = json.dumps(_jsonable(request), ensure_ascii=False)
            environment = os.environ.copy()
            environment.update(self.env)
            completed = subprocess.run(
                self.command,
                input=encoded,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                env=environment,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LongVideoWorkerError(
                f"external worker 启动/执行失败: {exc}",
                integration_id=self.integration_id,
                code="worker_execution_failed",
            ) from exc
        if completed.returncode != 0:
            raise LongVideoWorkerError(
                f"external worker 退出码 {completed.returncode}: "
                f"{completed.stderr.strip()[-1024:]}",
                integration_id=self.integration_id,
                code="worker_execution_failed",
                details={"returncode": completed.returncode},
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise LongVideoWorkerError(
                "external worker stdout 不是合法 JSON",
                integration_id=self.integration_id,
                code="worker_protocol_error",
            ) from exc
        if not isinstance(response, Mapping):
            raise LongVideoWorkerError(
                "external worker response 必须是 JSON object",
                integration_id=self.integration_id,
                code="worker_protocol_error",
            )
        if response.get("status", "ok") in {"failed", "error"}:
            raise LongVideoWorkerError(
                str(response.get("message", "external worker 报告失败")),
                integration_id=self.integration_id,
                code=str(response.get("code", "worker_execution_failed")),
                details=response,
            )
        return response.get("result", response)

    def run(self, operation: str, **payload: Any) -> Any:
        """Public generic operation hook for custom matrix/worker runners."""

        return self._request(operation, **payload)

    def encode(self, batch: ClipBatch, *, prompt: str, feature_stage: str) -> Any:
        return self._request(
            "encode",
            prompt=prompt,
            feature_stage=feature_stage,
            frames=_jsonable(batch.frames),
            timestamps_s=_jsonable(batch.timestamps_s),
            frame_indices=None if batch.frame_indices is None else _jsonable(batch.frame_indices),
            video_ids=list(batch.video_ids),
            valid_mask=None if batch.valid_mask is None else _jsonable(batch.valid_mask),
            metadata=_jsonable(batch.metadata),
        )

    def init_state(self, video_id: str, *, prompt: str, feature_stage: str) -> Any:
        return self._request(
            "init_state",
            video_id=video_id,
            prompt=prompt,
            feature_stage=feature_stage,
        )

    def encode_step(
        self,
        chunk: ClipBatch,
        state: Any,
        *,
        prompt: str,
        feature_stage: str,
        compression: Any = None,
    ) -> Any:
        return self._request(
            "encode_step",
            prompt=prompt,
            feature_stage=feature_stage,
            compression=("identity" if compression is not None else "off"),
            state=_jsonable(state),
            frames=_jsonable(chunk.frames),
            timestamps_s=_jsonable(chunk.timestamps_s),
            frame_indices=None if chunk.frame_indices is None else _jsonable(chunk.frame_indices),
            video_ids=list(chunk.video_ids),
            valid_mask=None if chunk.valid_mask is None else _jsonable(chunk.valid_mask),
            metadata=_jsonable(chunk.metadata),
        )

    def finalize(self, state: Any, *, prompt: str, feature_stage: str) -> Any:
        return self._request(
            "finalize",
            state=_jsonable(state),
            prompt=prompt,
            feature_stage=feature_stage,
        )


class _ExternalAdapterBase:
    """Shared constructor, asset checks, and output metadata."""

    integration_id: str = "external"
    default_feature_stage: str = "projected_visual"
    default_checkout_path: str | None = None
    default_model_path: str | None = None
    default_entrypoint: str | None = None
    backend: str = "external"
    capabilities: EncoderCapabilities
    run_mode: str = "fixed"

    def __init__(
        self,
        *,
        checkout_path: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        project_root: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        model_name: str | None = None,
        runtime: str = "in_process",
        preprocess_profile: str = "bthwc_uint8",
        prompt: str = DEFAULT_NEUTRAL_PROMPT,
        feature_stage: str | None = None,
        cache_mode: str = "identity",
        cache_policy: Any | None = None,
        compression_mode: str | None = None,
        kv_compression: str | None = None,
        cache_compression: str | None = None,
        worker: Any | None = None,
        upstream_worker: Any | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        load_model_fn: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        worker_factory: Callable[..., Any] | None = None,
        entrypoint: str | None = None,
        upstream_entrypoint: str | None = None,
        worker_command: str | Sequence[str] | None = None,
        external_command: str | Sequence[str] | None = None,
        worker_cwd: str | os.PathLike[str] | None = None,
        worker_env: Mapping[str, str] | None = None,
        worker_timeout_s: float = 300.0,
        worker_runner: Callable[[Mapping[str, Any]], Any] | None = None,
        strict_assets: bool = True,
        **_: Any,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空中性文本")
        self.integration_id = type(self).integration_id
        self.feature_stage = str(feature_stage or type(self).default_feature_stage)
        self.prompt = prompt.strip()
        alias_mode = compression_mode or kv_compression or cache_compression
        if alias_mode is not None and cache_mode == "identity":
            cache_mode = alias_mode
        if cache_policy is not None and cache_mode == "identity":
            cache_mode = getattr(cache_policy, "name", cache_policy)
        self.cache_mode = _normalise_cache_mode(cache_mode)
        self.device = str(device)
        self.model_name = None if model_name is None else str(model_name)
        self.runtime = str(runtime)
        self.preprocess_profile = str(preprocess_profile)
        self.project_root = (
            _project_root()
            if project_root is None and root is None
            else Path(project_root if project_root is not None else root).expanduser().resolve()
        )
        checkout_value = (
            type(self).default_checkout_path if checkout_path is None else checkout_path
        )
        if model_path is None:
            model_path = checkpoint_path
        model_value = type(self).default_model_path if model_path is None else model_path
        self.checkout_path = _resolve_path(checkout_value, self.project_root)
        self.model_path = _resolve_path(model_value, self.project_root)
        self.processor = processor
        self.worker: Any | None = (
            worker
            if worker is not None
            else upstream_worker
            if upstream_worker is not None
            else model
        )
        if load_model_fn is None:
            load_model_fn = model_loader or worker_factory
        if entrypoint is None:
            entrypoint = upstream_entrypoint
        if worker_command is None:
            worker_command = external_command
        self.implementation_source = "external_worker_facade"

        if self.worker is None and (worker_command is not None or worker_runner is not None):
            self.worker = ExternalPythonWorker(
                worker_command or (),
                integration_id=self.integration_id,
                cwd=worker_cwd,
                env=worker_env,
                timeout_s=worker_timeout_s,
                runner=worker_runner,
            )
            self.implementation_source = "external_python_worker"
        elif self.worker is None and load_model_fn is not None:
            if strict_assets:
                self._require_assets(require_checkout=False)
            try:
                loaded = _invoke_loader(
                    load_model_fn,
                    checkout_path=self.checkout_path,
                    model_path=self.model_path,
                    device=self.device,
                    processor=processor,
                    prompt=self.prompt,
                )
            except LongVideoAssetError:
                raise
            except Exception as exc:
                raise LongVideoWorkerError(
                    f"无法加载 {self.integration_id} upstream model: {exc}",
                    integration_id=self.integration_id,
                    code="model_load_failed",
                ) from exc
            self.worker, loaded_processor = _unpack_loaded_worker(loaded, self.processor)
            if self.processor is None:
                self.processor = loaded_processor
            self.implementation_source = "explicit_loader"
        elif self.worker is None and (entrypoint or type(self).default_entrypoint):
            selected_entrypoint = entrypoint or type(self).default_entrypoint
            if self.checkout_path is None:
                raise LongVideoAssetError(
                    integration_id=self.integration_id,
                    code="missing_asset",
                    message=f"{self.integration_id} 未配置 checkout_path，无法加载 {selected_entrypoint}",
                    details={"checkout_path": None, "entrypoint": selected_entrypoint},
                )
            if strict_assets:
                self._require_assets(require_checkout=True)
            try:
                loaded = _load_entrypoint(
                    self.checkout_path,
                    selected_entrypoint,
                    {
                        "checkout_path": self.checkout_path,
                        "model_path": self.model_path,
                        "device": self.device,
                        "processor": self.processor,
                        "prompt": self.prompt,
                    },
                )
                self.worker, loaded_processor = _unpack_loaded_worker(loaded, self.processor)
                if self.processor is None:
                    self.processor = loaded_processor
            except LongVideoAssetError:
                raise
            except Exception as exc:
                raise LongVideoWorkerError(
                    f"无法导入 {self.integration_id} upstream entrypoint: {exc}",
                    integration_id=self.integration_id,
                    code="model_load_failed",
                ) from exc
            self.implementation_source = "pinned_upstream_entrypoint"

        if self.worker is None:
            # A planned catalog item without a local worker must fail closed.  In
            # particular, do not instantiate a random torch module as a smoke
            # substitute merely because the target is listed in the catalog.
            self._require_assets(require_checkout=True)
            raise LongVideoAssetError(
                integration_id=self.integration_id,
                code="worker_unconfigured",
                message=(
                    f"{self.integration_id} 没有可运行的 worker/model；请提供显式 worker、"
                    "worker_command 或 load_model_fn（不会隐式联网）"
                ),
                details={
                    "checkout_path": _path_text(checkout_value, self.project_root),
                    "model_path": _path_text(model_value, self.project_root),
                },
            )

    def _require_assets(self, *, require_checkout: bool) -> None:
        missing: list[dict[str, str]] = []
        if require_checkout and self.checkout_path is None:
            missing.append({"kind": "checkout", "path": "<unset>"})
        elif (
            require_checkout and self.checkout_path is not None and not self.checkout_path.exists()
        ):
            missing.append({"kind": "checkout", "path": str(self.checkout_path)})
        if self.model_path is None:
            missing.append({"kind": "checkpoint", "path": "<unset>"})
        elif not self.model_path.exists():
            missing.append({"kind": "checkpoint", "path": str(self.model_path)})
        if missing:
            raise LongVideoAssetError(
                integration_id=self.integration_id,
                code="missing_asset",
                message=(
                    f"{self.integration_id} 缺少本地 upstream 资产；已 fail-closed，"
                    "请先按 lock 显式准备 checkout/checkpoint"
                ),
                details={"missing": missing},
            )

    def validate_assets(self) -> dict[str, Any]:
        """Return local asset status without loading heavy dependencies."""

        return {
            "integration_id": self.integration_id,
            "checkout_path": None if self.checkout_path is None else str(self.checkout_path),
            "checkout_exists": bool(self.checkout_path and self.checkout_path.exists()),
            "model_path": None if self.model_path is None else str(self.model_path),
            "model_exists": bool(self.model_path and self.model_path.exists()),
        }

    def _fixed_call(self, batch: ClipBatch, *, train: bool = False) -> Any:
        worker = self.worker
        if worker is None:  # pragma: no cover - constructor prevents this
            raise LongVideoWorkerError(
                "worker 未初始化", integration_id=self.integration_id, code="worker_unconfigured"
            )
        function = getattr(worker, "encode", None)
        if not callable(function):
            function = worker if callable(worker) else None
        if function is None:
            raise LongVideoWorkerError(
                f"{self.integration_id} worker 缺少 encode/callable 接口",
                integration_id=self.integration_id,
                code="worker_protocol_error",
            )
        return _invoke_fixed(
            function,
            batch,
            prompt=self.prompt,
            stage=self.feature_stage,
            train=train,
            processor=self.processor,
            device=self.device,
        )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        started = time.perf_counter()
        try:
            raw = self._fixed_call(batch, train=train)
            output = _normalise_output(
                raw,
                batch,
                integration_id=self.integration_id,
                feature_stage=self.feature_stage,
                prompt=self.prompt,
                cache_mode=self.cache_mode,
                preprocess_profile=self.preprocess_profile,
            )
        except (LongVideoAssetError, LongVideoWorkerError):
            raise
        except Exception as exc:
            raise LongVideoWorkerError(
                f"{self.integration_id} fixed forward 失败: {exc}",
                integration_id=self.integration_id,
                code="forward_failed",
            ) from exc
        return _with_canonical_aux(
            output,
            integration_id=self.integration_id,
            feature_stage=self.feature_stage,
            prompt=self.prompt,
            cache_mode=self.cache_mode,
            extra={
                "forward_seconds": time.perf_counter() - started,
                "implementation_source": self.implementation_source,
                "backend": self.backend,
                "cache_owner": getattr(self, "cache_owner", None),
            },
        )

    # Upstream wrappers often call this operation ``encode_video``.  Keep it
    # as a thin alias while preserving the framework's canonical ``encode``.
    def encode_video(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        return self.encode(batch, train=train)


class ExternalFixedVideoAdapter(_ExternalAdapterBase, VideoEncoderAdapter):
    """Base class for fixed/long (non-persistent) VLM integrations."""

    run_mode = "long"

    def __init__(self, **kwargs: Any) -> None:
        # Fixed/long adapters do not carry persistent state; ``off`` is the
        # unambiguous default.  Streaming subclasses retain ``identity`` as
        # their no-compression baseline.
        kwargs.setdefault("cache_mode", "off")
        self.capabilities = type(self).capabilities
        super().__init__(**kwargs)


def _split_stream_result(raw: Any) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], bool]:
    """Extract output/state/telemetry from common upstream worker shapes."""

    if isinstance(raw, StreamStep):
        return raw.output, raw.state, raw.telemetry, raw.cache_updates, raw.final
    if isinstance(raw, Mapping):
        output = raw.get("output")
        if output is None:
            output = raw.get("features", raw.get("embedding"))
        state = raw.get("state", raw.get("worker_state", _MISSING))
        telemetry = raw.get("telemetry", {})
        updates = raw.get("cache_updates", {})
        final = bool(raw.get("final", False))
        if not isinstance(telemetry, Mapping):
            telemetry = {"worker_telemetry": telemetry}
        if not isinstance(updates, Mapping):
            updates = {}
        return output, state, telemetry, updates, final
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0], raw[1], {}, {}, False
    return raw, _MISSING, {}, {}, False


def _declared_cache_kind(capabilities: EncoderCapabilities) -> CacheKind | None:
    if capabilities.supports_kv_cache:
        return CacheKind.KV
    if capabilities.supports_token_cache:
        return CacheKind.TOKEN
    if capabilities.supports_visual_memory_cache:
        return CacheKind.VISUAL_MEMORY
    return None


def _coerce_cache_view(
    value: Any,
    *,
    chunk: ClipBatch,
    capabilities: EncoderCapabilities,
    name: str,
) -> CacheView | None:
    """Accept CacheView or a small worker cache mapping without guessing kinds."""

    kind = _declared_cache_kind(capabilities)
    if kind is None:
        return None
    if isinstance(value, CacheUpdate):
        return value.view
    if isinstance(value, CacheView):
        if value.kind != kind:
            raise ValueError(f"{name} cache kind={value.kind.value} 与声明 {kind.value} 不一致")
        return value
    if isinstance(value, Mapping) and "view" in value:
        return _coerce_cache_view(value["view"], chunk=chunk, capabilities=capabilities, name=name)

    tensors: Mapping[str, Any] | None = None
    sequence_axis: int | None = None
    timeline_value: Any | None = None
    if isinstance(value, Mapping) and isinstance(value.get("tensors"), Mapping):
        tensors = value["tensors"]
        sequence_axis = value.get("sequence_axis", -2)
        timeline_value = value.get("timeline")
    elif isinstance(value, Mapping):
        # A worker may return a flat layer->tensor mapping.  Only accept entries
        # with array-like shapes; arbitrary bookkeeping is not a cache.
        candidates = {
            str(key): item
            for key, item in value.items()
            if hasattr(item, "shape") or isinstance(item, (list, tuple, np.ndarray))
        }
        if candidates:
            tensors = candidates
            sequence_axis = -2
    elif hasattr(value, "shape") or isinstance(value, (list, tuple, np.ndarray)):
        tensors = {name: value}
        sequence_axis = 1
    if not tensors:
        return None
    try:
        normalized_tensors = {
            key: item if hasattr(item, "shape") else np.asarray(item)
            for key, item in tensors.items()
        }
        first = next(iter(normalized_tensors.values()))
        shape = _shape(first)
        axis = int(sequence_axis if sequence_axis is not None else -2)
        axis = axis if axis >= 0 else len(shape) + axis
        if axis <= 0 or axis >= len(shape):
            raise ValueError(f"sequence_axis={sequence_axis} 对 shape={shape} 无效")
        sequence_length = shape[axis]
        timeline = _timeline_from_value(timeline_value, chunk, sequence_length)
        if timeline is None:
            timeline = _timeline_for_batch(chunk, sequence_length)
        return CacheView(
            kind=kind,
            tensors=normalized_tensors,
            sequence_axis=int(sequence_axis if sequence_axis is not None else -2),
            timeline=timeline,
            metadata={"integration_id": name, "compression": "disabled"},
        )
    except Exception as exc:
        raise ValueError(f"{name} cache 不能规范化: {exc}") from exc


class ExternalStreamingVideoAdapter(_ExternalAdapterBase, StreamingVideoEncoderAdapter):
    """Base class for explicit-state visual-memory/decoder-KV integrations."""

    run_mode = "streaming"

    def __init__(self, **kwargs: Any) -> None:
        self.capabilities = type(self).capabilities
        self._last_chunks: dict[str, ClipBatch] = {}
        super().__init__(**kwargs)

    def init_state(self, video_id: str) -> StreamState:
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id 必须是非空字符串")
        worker_state: Any = None
        function = getattr(self.worker, "init_state", None)
        if callable(function):
            values = {
                "video_id": video_id,
                "prompt": self.prompt,
                "feature_stage": self.feature_stage,
            }
            kwargs = _filtered_kwargs(function, values)
            try:
                worker_state = _invoke_candidates(
                    function,
                    (((video_id,), kwargs), ((), kwargs)),
                )
            except Exception as exc:
                raise LongVideoWorkerError(
                    f"{self.integration_id} init_state 失败: {exc}",
                    integration_id=self.integration_id,
                    code="state_init_failed",
                ) from exc
        return StreamState(
            video_id=video_id,
            opaque=worker_state,
            metadata={
                "integration_id": self.integration_id,
                "feature_stage": self.feature_stage,
                "prompt": self.prompt,
                "cache_mode": self.cache_mode,
                "cache_compression": "disabled",
            },
        )

    @staticmethod
    def _compression_name(compression: Any) -> str:
        if compression is None:
            return "off"
        if compression is False:
            return "off"
        if compression is True:
            return "identity"
        if isinstance(compression, str):
            return _normalise_cache_mode(compression)
        if isinstance(compression, Mapping):
            name = compression.get("name", compression.get("policy"))
            if name is None:
                raise ValueError("compression mapping 需要 name=off 或 identity")
            return _normalise_cache_mode(str(name))
        name = getattr(compression, "name", None)
        if name is None:
            raise ValueError("compression 只能是 None、off 或 identity policy")
        return _normalise_cache_mode(str(name))

    def _stream_call(self, chunk: ClipBatch, state: StreamState, compression: Any) -> Any:
        worker = self.worker
        function = getattr(worker, "encode_step", None) if worker is not None else None
        if not callable(function):
            function = worker if callable(worker) else None
        if function is None:
            raise LongVideoWorkerError(
                f"{self.integration_id} worker 缺少 encode_step/callable 接口",
                integration_id=self.integration_id,
                code="worker_protocol_error",
            )
        return _invoke_stream(
            function,
            chunk,
            state,
            prompt=self.prompt,
            stage=self.feature_stage,
            compression=compression,
            processor=self.processor,
            device=self.device,
        )

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: Any = None,
    ) -> StreamStep:
        validate_clip_for_capabilities(chunk, self.capabilities, streaming=True, train=train)
        if not isinstance(state, StreamState):
            raise TypeError("state 必须是 StreamState")
        if state.video_id != chunk.video_ids[0]:
            raise ValueError("chunk.video_ids 必须与 state.video_id 一致")
        try:
            compression_name = self._compression_name(compression)
        except ValueError as exc:
            raise LongVideoWorkerError(
                f"{self.integration_id} 不支持 KV 压缩：{exc}",
                integration_id=self.integration_id,
                code="compression_disabled",
            ) from exc
        started = time.perf_counter()
        try:
            raw = self._stream_call(
                chunk, state, compression if compression_name != "off" else None
            )
            output_raw, worker_state, telemetry, updates, final = _split_stream_result(raw)
            raw_cache_payload: Any | None = None
            if isinstance(raw, Mapping):
                raw_cache_payload = raw.get("caches", raw.get("cache"))
            cache_views: dict[str, CacheView] = {}
            if raw_cache_payload is not None:
                if (
                    isinstance(raw_cache_payload, Mapping) and "tensors" in raw_cache_payload
                ) or not isinstance(raw_cache_payload, Mapping):
                    raw_cache_payload = {"default": raw_cache_payload}
                for cache_name, cache_value in raw_cache_payload.items():
                    try:
                        view = _coerce_cache_view(
                            cache_value,
                            chunk=chunk,
                            capabilities=self.capabilities,
                            name=f"{self.integration_id}.{cache_name}",
                        )
                    except Exception as exc:
                        raise LongVideoWorkerError(
                            f"{self.integration_id} worker cache 输出非法: {exc}",
                            integration_id=self.integration_id,
                            code="invalid_cache",
                        ) from exc
                    if view is not None:
                        cache_views[str(cache_name)] = view
            output = (
                None
                if output_raw is None
                else _normalise_output(
                    output_raw,
                    chunk,
                    integration_id=self.integration_id,
                    feature_stage=self.feature_stage,
                    prompt=self.prompt,
                    cache_mode=compression_name,
                    preprocess_profile=self.preprocess_profile,
                )
            )
            if output is not None:
                output = _with_canonical_aux(
                    output,
                    integration_id=self.integration_id,
                    feature_stage=self.feature_stage,
                    prompt=self.prompt,
                    cache_mode=compression_name,
                    extra={
                        "implementation_source": self.implementation_source,
                        "backend": self.backend,
                        "cache_owner": getattr(self, "cache_owner", None),
                    },
                )
            if worker_state is _MISSING:
                worker_state = state.opaque
            if isinstance(worker_state, StreamState):
                next_state = worker_state
                if next_state.video_id != state.video_id:
                    raise ValueError("worker 不能切换 video_id")
                if next_state.step_index <= state.step_index:
                    next_state = next_state.replace(step_index=state.step_index + 1)
                next_time = _next_timestamp_for_chunk(chunk, state.next_timestamp_s)
                previous_limit = -1.0 if state.next_timestamp_s is None else state.next_timestamp_s
                next_state = next_state.replace(
                    caches={**dict(state.caches), **cache_views, **dict(next_state.caches)},
                    next_timestamp_s=(
                        next_state.next_timestamp_s
                        if next_state.next_timestamp_s is not None
                        and next_state.next_timestamp_s > previous_limit
                        else next_time
                    ),
                    metadata={
                        **dict(state.metadata),
                        **dict(next_state.metadata),
                        "cache_mode": compression_name,
                    },
                )
            else:
                valid_length = int(chunk.valid_lengths[0])
                next_time = _next_timestamp_for_chunk(chunk, state.next_timestamp_s)
                next_state = state.replace(
                    step_index=state.step_index + 1,
                    opaque=worker_state,
                    caches={**dict(state.caches), **cache_views},
                    next_timestamp_s=next_time,
                    metadata={
                        **dict(state.metadata),
                        "cache_mode": compression_name,
                        "last_chunk_frames": valid_length,
                    },
                )
            cache_updates: dict[str, CacheUpdate] = {}
            for name, update in updates.items():
                if isinstance(update, CacheUpdate):
                    cache_updates[str(name)] = update
                elif isinstance(update, CacheView):
                    cache_updates[str(name)] = CacheUpdate.append(update)
                else:
                    try:
                        view = _coerce_cache_view(
                            update,
                            chunk=chunk,
                            capabilities=self.capabilities,
                            name=f"{self.integration_id}.{name}",
                        )
                    except Exception as exc:
                        raise LongVideoWorkerError(
                            f"{self.integration_id} worker cache update 非法: {exc}",
                            integration_id=self.integration_id,
                            code="invalid_cache",
                        ) from exc
                    if view is not None:
                        cache_updates[str(name)] = CacheUpdate.append(view)
            merged_telemetry = {
                **dict(telemetry),
                "integration_id": self.integration_id,
                "prompt": self.prompt,
                "cache_mode": compression_name,
                "cache_compression": "disabled",
                "forward_seconds": time.perf_counter() - started,
                "implementation_source": self.implementation_source,
            }
            step = StreamStep(
                output=output,
                state=next_state,
                cache_updates=cache_updates,
                telemetry=merged_telemetry,
                final=final,
            )
            validate_stream_step(
                step,
                previous_state=state,
                chunk=chunk,
                capabilities=self.capabilities,
            )
            self._last_chunks[state.video_id] = chunk
            return step
        except (LongVideoAssetError, LongVideoWorkerError):
            raise
        except Exception as exc:
            raise LongVideoWorkerError(
                f"{self.integration_id} streaming forward 失败: {exc}",
                integration_id=self.integration_id,
                code="forward_failed",
            ) from exc

    def finalize(self, state: StreamState) -> EncoderOutput | None:
        function = getattr(self.worker, "finalize", None) if self.worker is not None else None
        if not callable(function):
            return None
        values = {
            "state": state,
            "stream_state": state,
            "worker_state": state.opaque,
            "opaque": state.opaque,
            "prompt": self.prompt,
            "feature_stage": self.feature_stage,
        }
        kwargs = _filtered_kwargs(function, values)
        try:
            raw = _invoke_candidates(
                function, (((state,), kwargs), ((state.opaque,), kwargs), ((), kwargs))
            )
        except Exception as exc:
            raise LongVideoWorkerError(
                f"{self.integration_id} finalize 失败: {exc}",
                integration_id=self.integration_id,
                code="finalize_failed",
            ) from exc
        if raw is None:
            return None
        batch = self._last_chunks.get(state.video_id)
        if isinstance(raw, EncoderOutput):
            return _with_canonical_aux(
                raw,
                integration_id=self.integration_id,
                feature_stage=self.feature_stage,
                prompt=self.prompt,
                cache_mode=self.cache_mode,
            )
        if batch is None:
            raise LongVideoWorkerError(
                f"{self.integration_id} finalize 返回原始张量但没有可用 chunk timeline",
                integration_id=self.integration_id,
                code="invalid_output",
            )
        output = _normalise_output(
            raw,
            batch,
            integration_id=self.integration_id,
            feature_stage=self.feature_stage,
            prompt=self.prompt,
            cache_mode=self.cache_mode,
            preprocess_profile=self.preprocess_profile,
        )
        return _with_canonical_aux(
            output,
            integration_id=self.integration_id,
            feature_stage=self.feature_stage,
            prompt=self.prompt,
            cache_mode=self.cache_mode,
            extra={
                "implementation_source": self.implementation_source,
                "backend": self.backend,
                "cache_owner": getattr(self, "cache_owner", None),
            },
        )

    def encode_chunk(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: Any = None,
    ) -> StreamStep:
        """Compatibility alias used by a few upstream streaming examples."""

        return self.encode_step(chunk, state, train=train, compression=compression)


__all__ = [
    "DEFAULT_NEUTRAL_PROMPT",
    "build_token_timeline",
    "ExternalFixedVideoAdapter",
    "ExternalAssetError",
    "ExternalPythonWorker",
    "ExternalStreamingVideoAdapter",
    "ExternalWorkerError",
    "LongVideoAssetError",
    "LongVideoIntegrationError",
    "LongVideoWorkerError",
    "StructuredLongVideoError",
    "MissingAssetError",
    "MissingLongVideoAssetError",
    "normalize_long_video_output",
]
