from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import (
    ClipBatch,
    EncoderOutput,
    StreamState,
    TokenTimeline,
    validate_stream_step,
)
from vadbench.integrations import compatibility


def _batch(*, start: int = 0, frames: int = 4) -> ClipBatch:
    indices = np.arange(start, start + frames, dtype=np.int64)[None, :]
    timestamps = indices.astype(np.float64) / 30.0
    return ClipBatch(
        frames=np.zeros((1, frames, 8, 8, 3), dtype=np.uint8),
        timestamps_s=timestamps,
        video_ids=("video",),
        frame_indices=indices,
    )


class _FakePublicAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        del train
        features = np.ones((batch.batch_size, 1, 4), dtype=np.float32)
        timeline = TokenTimeline(
            start_s=np.asarray(batch.timestamps_s)[:, :1],
            end_s=np.asarray(batch.timestamps_s)[:, -1:] + 1.0 / 30.0,
            source_frame_start=np.asarray(batch.frame_indices)[:, :1],
            source_frame_end=np.asarray(batch.frame_indices)[:, -1:] + 1,
        )
        return EncoderOutput(
            features=features,
            pooled=features.mean(axis=1),
            timeline=timeline,
            aux={"feature_stage": "pooled", "sequence_source": "fake_public"},
        )


def test_fixed_compatibility_bridge_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(compatibility, "TorchvisionVideoAdapter", _FakePublicAdapter)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"verified-public-checkpoint-fixture")

    adapter = compatibility.create_compatibility_adapter(
        "timesformer",
        model_path=checkpoint,
        project_root=tmp_path,
        device="cpu",
        requested_feature_stage="last_hidden_state",
    )
    output = adapter.encode(_batch())

    assert output.features.shape == (1, 1, 4)
    assert output.timeline.num_tokens == 1
    assert output.aux["integration_id"] == "timesformer"
    assert output.aux["native_route_available"] is False
    assert output.aux["compatibility_bridge"] == "torchvision-r2plus1d_18"

    with pytest.raises(FileNotFoundError, match="公开 checkpoint"):
        compatibility.create_compatibility_adapter(
            "timesformer",
            model_path=tmp_path / "missing.pth",
            project_root=tmp_path,
        )


def test_streaming_compatibility_bridge_exposes_identity_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(compatibility, "TorchvisionVideoAdapter", _FakePublicAdapter)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"verified-public-checkpoint-fixture")
    adapter = compatibility.create_compatibility_adapter(
        "mukv",
        model_path=checkpoint,
        project_root=tmp_path,
        device="cpu",
    )
    state = adapter.init_state("video")
    assert isinstance(state, StreamState)

    first_batch = _batch(start=0)
    first = adapter.encode_step(first_batch, state, compression="identity")
    validate_stream_step(
        first,
        previous_state=state,
        chunk=first_batch,
        capabilities=adapter.capabilities,
    )
    second_batch = _batch(start=4)
    second = adapter.encode_step(second_batch, first.state, compression="off")
    validate_stream_step(
        second,
        previous_state=first.state,
        chunk=second_batch,
        capabilities=adapter.capabilities,
    )

    assert second.state.step_index == 2
    assert second.state.caches["default"].kind.value == "decoder_kv"
    assert second.state.caches["default"].sequence_length == 2
    assert second.telemetry["cache_mode"] == "identity"
    assert second.output is not None
    assert second.output.aux["native_route_available"] is False
