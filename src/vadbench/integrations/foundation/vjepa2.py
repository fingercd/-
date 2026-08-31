"""Native Meta V-JEPA 2 fixed-clip integration.

The Hugging Face V-JEPA 2 model exposes ``get_vision_features`` rather than a
standard ``forward(pixel_values=...)`` method.  The loader below follows the
official model-card API and keeps all preprocessing inside this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import ClipBatch
from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


def _move(value: Any, device: str | None) -> Any:
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if device is not None and hasattr(value, "to"):
        try:
            return value.to(device)
        except (TypeError, RuntimeError):
            return value
    return value


class _VJEPA2Encoder:
    def __init__(self, model: Any, processor: Any, *, device: str | None) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def encode(self, batch: ClipBatch) -> Any:
        videos = [
            [
                np.asarray(frame, dtype=np.uint8)
                for frame in np.asarray(batch.frames)[row, : int(length)]
            ]
            for row, length in enumerate(batch.valid_lengths)
        ]
        processed = self.processor(videos, return_tensors="pt")
        if isinstance(processed, Mapping):
            inputs = _move(dict(processed), self.device)
        else:
            inputs = _move(processed, self.device)
        get_features = getattr(self.model, "get_vision_features", None)
        if not callable(get_features):
            raise RuntimeError("V-JEPA 2 model 缺少官方 get_vision_features 接口")
        import torch

        with torch.no_grad():
            if isinstance(inputs, Mapping):
                return get_features(**inputs)
            return get_features(inputs)


def load_vjepa2(model_path: str | Path, device: str | None = None, **_: Any) -> Any:
    """Load the pinned local V-JEPA 2 checkpoint through Transformers."""

    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("V-JEPA 2 需要 transformers>=4.56 与 AutoVideoProcessor") from exc
    path = str(model_path)
    model = AutoModel.from_pretrained(path, local_files_only=True)
    processor = AutoVideoProcessor.from_pretrained(path, local_files_only=True)
    if device is not None and callable(getattr(model, "to", None)):
        model.to(device)
    if callable(getattr(model, "eval", None)):
        model.eval()
    return _VJEPA2Encoder(model, processor, device=device)


class VJEPA2Adapter(FoundationVideoAdapter):
    """Adapt the official V-JEPA 2 video encoder to VADBench."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "vjepa2"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "vjepa2-bthwc-v1"
    DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
    DEFAULT_RUNTIME = "in_process"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("runtime", "in_process")
        kwargs.setdefault("loader", load_vjepa2)
        kwargs.pop("upstream_entrypoint", None)
        kwargs.pop("entrypoint", None)
        super().__init__(**kwargs)


VJepa2Adapter = VJEPA2Adapter
VJepaAdapter = VJEPA2Adapter

__all__ = ["VJEPA2Adapter", "VJepa2Adapter", "VJepaAdapter", "load_vjepa2"]
