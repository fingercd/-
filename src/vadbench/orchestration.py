"""把实验 YAML、adapter registry、采样 batch 和缓存策略连接起来。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vadbench.compression import build_cache_policy
from vadbench.config import load_yaml, validate_capabilities
from vadbench.contracts import ClipBatch
from vadbench.registry import ENCODER_REGISTRY

BUILTIN_ENCODER_CONFIGS = {
    "videomaev2": "configs/encoders/videomaev2-base.yaml",
    "hermes_llava_ov": "configs/encoders/hermes-llava-ov-0.5b.yaml",
}


def load_encoder_definition(
    adapter_id: str,
    *,
    project_root: str | Path = ".",
    path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    selected = Path(path) if path is not None else Path(BUILTIN_ENCODER_CONFIGS[adapter_id])
    if not selected.is_absolute():
        selected = root / selected
    definition = load_yaml(selected)
    if definition.get("adapter") != adapter_id:
        raise ValueError(
            f"encoder definition adapter={definition.get('adapter')!r}，预期 {adapter_id!r}"
        )
    return definition


def _absolute_constructor_paths(constructor: dict[str, Any], project_root: Path) -> None:
    for key in ("model_name", "model_path", "checkout_path"):
        value = constructor.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (project_root / path).resolve()
        if candidate.exists() or key in {"model_path", "checkout_path"}:
            constructor[key] = str(candidate)


def create_encoder_from_experiment(
    config: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
) -> tuple[Any, dict[str, Any]]:
    encoder_config = config.get("encoder")
    if not isinstance(encoder_config, Mapping):
        raise ValueError("experiment 缺少 encoder 对象")
    adapter_id = str(encoder_config.get("adapter", ""))
    if not adapter_id:
        raise ValueError("encoder.adapter 不能为空")
    spec = ENCODER_REGISTRY.get_spec(adapter_id)
    validate_capabilities(config, spec.capabilities)
    definition_path = encoder_config.get("definition")
    definition = load_encoder_definition(
        adapter_id,
        project_root=project_root,
        path=definition_path,
    )
    constructor = dict(definition.get("constructor", {}))
    params = encoder_config.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("encoder.params 必须是对象")
    constructor.update(params)
    if encoder_config.get("device") is not None:
        constructor["device"] = str(encoder_config["device"])
    _absolute_constructor_paths(constructor, Path(project_root).resolve())
    return ENCODER_REGISTRY.create(adapter_id, **constructor), definition


def _slice_metadata(value: Any, start: int, stop: int, batch_size: int) -> Any:
    if isinstance(value, (str, bytes, Mapping)):
        return value
    if isinstance(value, Sequence) and len(value) == batch_size:
        sliced = value[start:stop]
        return tuple(sliced) if isinstance(value, tuple) else list(sliced)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == batch_size:
        return value[start:stop]
    return value


def slice_clip_batch(batch: ClipBatch, start: int, stop: int) -> ClipBatch:
    if not 0 <= start < stop <= batch.batch_size:
        raise ValueError(f"非法 batch slice [{start}:{stop}] / {batch.batch_size}")
    metadata = {
        key: _slice_metadata(value, start, stop, batch.batch_size)
        for key, value in batch.metadata.items()
    }
    return ClipBatch(
        frames=batch.frames[start:stop],
        timestamps_s=batch.timestamps_s[start:stop],
        video_ids=batch.video_ids[start:stop],
        valid_mask=None if batch.valid_mask is None else batch.valid_mask[start:stop],
        frame_indices=None if batch.frame_indices is None else batch.frame_indices[start:stop],
        metadata=metadata,
    )


def iter_microbatches(batches: Iterator[ClipBatch], micro_batch_size: int) -> Iterator[ClipBatch]:
    if isinstance(micro_batch_size, bool) or micro_batch_size <= 0:
        raise ValueError("micro_batch_size 必须是正整数")
    for batch in batches:
        for start in range(0, batch.batch_size, micro_batch_size):
            yield slice_clip_batch(batch, start, min(start + micro_batch_size, batch.batch_size))


def compression_from_experiment(config: Mapping[str, Any]) -> Any | None:
    streaming = config.get("streaming", {})
    compression = streaming.get("compression", {}) if isinstance(streaming, Mapping) else {}
    if not isinstance(compression, Mapping):
        raise ValueError("streaming.compression 必须是对象")
    name = str(compression.get("policy", "identity"))
    if name in {"hermes_native", "native"}:
        return None
    max_tokens = compression.get("max_tokens")
    if max_tokens is None and name == "keep_recent":
        max_tokens = compression.get("kv_budget_tokens")
    return build_cache_policy(name, max_tokens=None if max_tokens is None else int(max_tokens))


__all__ = [
    "BUILTIN_ENCODER_CONFIGS",
    "compression_from_experiment",
    "create_encoder_from_experiment",
    "iter_microbatches",
    "load_encoder_definition",
    "slice_clip_batch",
]
