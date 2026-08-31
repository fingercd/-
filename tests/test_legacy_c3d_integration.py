from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG
from vadbench.integrations.legacy import (
    DEFAULT_CAPABILITIES,
    LegacyAssetError,
    LegacyDependencyError,
    LegacyVideoAdapter,
)


def _batch(*, frames: int = 16, batch_size: int = 1) -> ClipBatch:
    pixels = np.arange(batch_size * frames * 4 * 6 * 3, dtype=np.uint8).reshape(
        batch_size, frames, 4, 6, 3
    )
    timestamps = (np.arange(frames, dtype=np.float64) / 30.0)[None, :].repeat(batch_size, 0)
    indices = np.arange(frames, dtype=np.int64)[None, :].repeat(batch_size, 0)
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        frame_indices=indices,
        video_ids=tuple(f"video-{i}" for i in range(batch_size)),
    )


class _FakeEncoder:
    def __init__(self) -> None:
        self.last_input: Any = None

    def encode(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        self.last_input = frames
        batch = int(frames.shape[0])
        return {"fc6": np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)}


def test_fake_encoder_receives_contiguous_c3d_bcthw_and_emits_contract() -> None:
    encoder = _FakeEncoder()
    adapter = LegacyVideoAdapter(encoder=encoder)

    output = adapter.encode(_batch())

    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert encoder.last_input.dtype == np.float32
    assert output.features.shape == (1, 1, 4)
    assert output.pooled.shape == (1, 4)
    assert output.timeline.start_s.tolist() == [[0.0]]
    assert output.timeline.source_frame_start.tolist() == [[0]]
    assert output.timeline.source_frame_end.tolist() == [[16]]
    assert output.aux["feature_stage"] == "fc_features"
    assert output.aux["sequence_source"] == "fc6"
    assert output.aux["preprocess_profile"] == "c3d-16x112-v1"
    assert output.aux["input_layout"] == "BCTHW"
    assert output.aux["input_shape"] == [1, 3, 16, 112, 112]


def test_short_or_long_input_is_defensively_normalized_to_profile() -> None:
    encoder = _FakeEncoder()
    adapter = LegacyVideoAdapter(encoder=encoder, clip_frames=16, image_size=112)

    short_output = adapter.encode(_batch(frames=4))
    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert short_output.timeline.source_frame_end.tolist() == [[4]]
    assert short_output.aux["temporal_adjustment"] == ["repeat_last_frame"]

    long_output = adapter.encode(_batch(frames=20))
    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert long_output.timeline.source_frame_end.tolist() == [[20]]
    assert long_output.aux["temporal_adjustment"] == ["truncate_first_contiguous"]


def test_capabilities_match_catalog_generic_fixed_profile() -> None:
    adapter = LegacyVideoAdapter(encoder=_FakeEncoder())
    assert adapter.capabilities == DEFAULT_CAPABILITIES
    assert adapter.capabilities == DEFAULT_INTEGRATION_CATALOG.get("c3d").capabilities
    assert adapter.capabilities.supports_fixed_clip is True
    assert adapter.capabilities.supports_streaming is False
    assert adapter.capabilities.supports_kv_cache is False
    assert adapter.capabilities.supports_token_cache is False
    assert adapter.capabilities.fixed_num_frames is None


def test_explicit_loader_receives_only_local_asset_paths(tmp_path: Path) -> None:
    checkout = tmp_path / "external" / "C3D"
    checkpoint = tmp_path / "weights" / "c3d.caffemodel"
    prototxt = checkout / "deploy.prototxt"
    checkout.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"local-checkpoint")
    prototxt.write_text("name: 'c3d'\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def loader(
        *,
        checkout_path: str,
        checkpoint_path: str,
        prototxt_path: str,
        device: str,
        **_: Any,
    ) -> _FakeEncoder:
        seen.update(
            checkout=checkout_path,
            checkpoint=checkpoint_path,
            prototxt=prototxt_path,
            device=device,
        )
        return _FakeEncoder()

    adapter = LegacyVideoAdapter(
        loader=loader,
        project_root=tmp_path,
        checkout_path="external/C3D",
        checkpoint_path="weights/c3d.caffemodel",
        prototxt_path="external/C3D/deploy.prototxt",
        device="cpu",
    )
    adapter.encode(_batch())
    assert seen == {
        "checkout": str(checkout.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "prototxt": str(prototxt.resolve()),
        "device": "cpu",
    }


def test_missing_assets_fail_closed_with_structured_error(tmp_path: Path) -> None:
    with pytest.raises(LegacyAssetError) as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path="external/C3D",
            checkpoint_path="weights/missing.caffemodel",
        )
    error = captured.value.to_dict()
    assert error["code"] == "missing_asset"
    assert error["integration_id"] == "c3d"
    assert {item["kind"] for item in error["details"]["missing"]} == {"checkout", "checkpoint"}


def test_missing_caffe_is_reported_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "C3D"
    checkpoint = tmp_path / "model.caffemodel"
    prototxt = checkout / "deploy.prototxt"
    checkout.mkdir()
    checkpoint.write_bytes(b"weights")
    prototxt.write_text("name: 'c3d'\n", encoding="utf-8")
    original_import = builtins.__import__

    def deny_caffe(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "caffe":
            raise ImportError("caffe unavailable in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_caffe)
    with pytest.raises(LegacyDependencyError) as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path=checkout,
            checkpoint_path=checkpoint,
            prototxt_path=prototxt,
        )
    assert captured.value.code == "missing_dependency"
    assert captured.value.to_dict()["details"]["dependency"] == "caffe"


@pytest.mark.parametrize(
    "entrypoint",
    ["../outside.py:load", "/tmp/loader.py:load", "loader.txt:load", "loader.py"],
)
def test_dynamic_entrypoint_validation_fails_closed(tmp_path: Path, entrypoint: str) -> None:
    checkout = tmp_path / "C3D"
    checkout.mkdir()
    with pytest.raises(LegacyAssetError, match="entrypoint") as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path=checkout,
            entrypoint=entrypoint,
            strict_assets=False,
        )
    assert captured.value.code == "invalid_entrypoint"
