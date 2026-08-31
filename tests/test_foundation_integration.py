from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    ExternalPythonFoundationBridge,
    FoundationAssetError,
    FoundationUpstreamError,
    InProcessFoundationBridge,
)
from vadbench.integrations.foundation.internvideo2 import InternVideo2Adapter
from vadbench.integrations.foundation.umt import UMTAdapter
from vadbench.integrations.foundation.uniformerv2 import UniFormerV2Adapter
from vadbench.integrations.foundation.videomamba import VideoMambaAdapter
from vadbench.integrations.foundation.vjepa2 import VJEPA2Adapter
from vadbench.registry import ENCODER_REGISTRY

ADAPTERS = {
    "umt": UMTAdapter,
    "uniformerv2": UniFormerV2Adapter,
    "internvideo2": InternVideo2Adapter,
    "videomamba": VideoMambaAdapter,
    "vjepa2": VJEPA2Adapter,
}


def _batch(*, batch_size: int = 2, frames: int = 4) -> ClipBatch:
    pixels = np.arange(batch_size * frames * 3 * 5 * 3, dtype=np.uint8).reshape(
        batch_size, frames, 3, 5, 3
    )
    timestamps = np.stack(
        [np.arange(frames, dtype=np.float64) / 10.0 + float(row) for row in range(batch_size)]
    )
    indices = np.stack([np.arange(frames, dtype=np.int64) + row * 10 for row in range(batch_size)])
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        video_ids=tuple(f"video-{row}" for row in range(batch_size)),
        frame_indices=indices,
    )


class _FakeUpstream:
    def __init__(self, *, output_kind: str = "sequence") -> None:
        self.output_kind = output_kind
        self.calls: list[Any] = []

    def __call__(self, frames: Any) -> Any:
        self.calls.append(frames)
        assert frames.dtype == np.uint8
        assert frames.ndim == 5  # canonical BTHWC boundary
        batch_size = int(frames.shape[0])
        if self.output_kind == "pooled":
            return np.arange(batch_size * 6, dtype=np.float32).reshape(batch_size, 6)
        features = np.arange(batch_size * 3 * 6, dtype=np.float32).reshape(batch_size, 3, 6)
        pooled = features.mean(axis=1)
        return {"features": features, "pooled": pooled}


@pytest.mark.parametrize("adapter_id", tuple(ADAPTERS))
def test_foundation_ids_construct_through_lazy_registry(adapter_id: str) -> None:
    fake = _FakeUpstream()
    adapter = ENCODER_REGISTRY.create(adapter_id, encoder=fake)

    assert adapter.capabilities == FOUNDATION_CAPABILITIES
    assert adapter.backend == adapter_id
    assert isinstance(adapter.bridge, ExternalPythonFoundationBridge)
    output = adapter.encode(_batch())

    assert output.features.shape == (2, 3, 6)
    assert output.pooled is not None and output.pooled.shape == (2, 6)
    assert output.timeline.start_s.shape == (2, 3)
    assert output.timeline.end_s.shape == (2, 3)
    assert output.timeline.source_frame_start.shape == (2, 3)
    assert output.timeline.source_frame_end.shape == (2, 3)
    assert np.all(np.diff(output.timeline.start_s, axis=1) >= 0)
    assert np.all(output.timeline.source_frame_end > output.timeline.source_frame_start)
    assert output.aux["feature_stage"] == adapter.feature_stage
    assert output.aux["sequence_source"] == "features"
    assert output.aux["backend"] == adapter_id
    assert output.aux["implementation_source"] == "lazy_upstream_bridge"
    assert output.aux["input_layout"] == "BTHWC"
    assert len(fake.calls) == 1


@pytest.mark.parametrize("adapter_id", tuple(ADAPTERS))
def test_foundation_bridge_normalizes_pooled_only_upstream(adapter_id: str) -> None:
    adapter = ADAPTERS[adapter_id](encoder=_FakeUpstream(output_kind="pooled"))
    output = adapter.encode(_batch(batch_size=1, frames=2))

    assert output.features.shape == (1, 1, 6)
    assert output.pooled is not None and output.pooled.shape == (1, 6)
    assert output.aux["sequence_source"] == "pooled_singleton"
    assert output.timeline.num_tokens == 1
    assert output.timeline.source_frame_start.shape == (1, 1)


@pytest.mark.parametrize("adapter_id", tuple(ADAPTERS))
def test_missing_local_checkpoint_fails_closed(adapter_id: str, tmp_path: Path) -> None:
    missing = tmp_path / adapter_id / "checkpoint"
    with pytest.raises(FoundationAssetError, match="不会自动联网"):
        ADAPTERS[adapter_id](model_path=missing)


@pytest.mark.parametrize("adapter_id", tuple(ADAPTERS))
def test_loader_is_explicit_and_lazy_after_local_asset_check(
    adapter_id: str, tmp_path: Path
) -> None:
    checkpoint = tmp_path / adapter_id / "checkpoint.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")
    calls: list[dict[str, Any]] = []
    fake = _FakeUpstream()

    def loader(model_path: str, *, device: str | None = None) -> Any:
        calls.append({"model_path": model_path, "device": device})
        return fake

    adapter = ADAPTERS[adapter_id](
        model_path=checkpoint,
        model_name=f"{adapter_id}-local",
        device="cpu",
        loader=loader,
    )
    assert calls == []
    assert adapter.encoder is None
    adapter.encode(_batch(batch_size=1))
    assert calls == [{"model_path": str(checkpoint.resolve()), "device": "cpu"}]
    assert adapter.encoder is fake


def test_in_process_bridge_is_available_without_heavy_imports() -> None:
    fake = _FakeUpstream()
    adapter = UMTAdapter(encoder=fake, runtime="in_process")
    assert isinstance(adapter.bridge, InProcessFoundationBridge)
    adapter.encode(_batch(batch_size=1))


def test_foundation_modules_do_not_import_model_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.partition(".")[0]
        if root in {"torch", "torchvision", "transformers", "timm", "mamba_ssm"}:
            raise AssertionError(f"foundation module imported heavy dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    # Classes are already imported above; constructing with a fake must still
    # stay entirely within the light-weight bridge.
    for adapter_class in ADAPTERS.values():
        adapter_class(encoder=_FakeUpstream()).encode(_batch(batch_size=1))


def test_existing_local_asset_without_loader_reports_upstream_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"placeholder")
    adapter = UMTAdapter(model_path=checkpoint)
    with pytest.raises(FoundationUpstreamError, match="未配置显式 upstream loader"):
        adapter.encode(_batch(batch_size=1))
