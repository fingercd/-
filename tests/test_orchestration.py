from __future__ import annotations

import numpy as np

from vadbench.contracts import ClipBatch
from vadbench.orchestration import (
    compression_from_experiment,
    load_encoder_definition,
    slice_clip_batch,
)


def test_builtin_encoder_definition_matches_registry() -> None:
    definition = load_encoder_definition("videomaev2")
    assert definition["adapter"] == "videomaev2"
    assert definition["constructor"]["model_name"] == "weights/videomaev2-base-hf"


def test_slice_clip_batch_keeps_row_metadata_aligned() -> None:
    batch = ClipBatch(
        frames=np.zeros((3, 2, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32),
        video_ids=("v", "v", "v"),
        frame_indices=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64),
        metadata={"clip_ids": ["a", "b", "c"], "sampling": "fixed"},
    )
    sliced = slice_clip_batch(batch, 1, 3)
    assert sliced.video_ids == ("v", "v")
    assert sliced.metadata["clip_ids"] == ["b", "c"]
    assert sliced.metadata["sampling"] == "fixed"


def test_native_compression_is_owned_by_adapter() -> None:
    config = {
        "streaming": {"compression": {"policy": "hermes_native"}},
    }
    assert compression_from_experiment(config) is None
