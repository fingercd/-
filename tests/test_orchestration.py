from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.orchestration import (
    compression_from_experiment,
    load_encoder_definition,
    slice_clip_batch,
)


def _write_definition(path: Path) -> None:
    path.write_text(
        "schema_version: 1\nadapter: videomaev2\nconstructor: {}\n",
        encoding="utf-8",
    )


def test_builtin_encoder_definition_matches_registry() -> None:
    definition = load_encoder_definition("videomaev2")
    assert definition["adapter"] == "videomaev2"
    assert definition["constructor"]["model_name"] == "weights/videomaev2-base-hf"


def test_explicit_definition_path_inside_project_root_loads(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    definition_path = project_root / "inside.yaml"
    _write_definition(definition_path)

    definition = load_encoder_definition(
        "videomaev2",
        project_root=project_root,
        path="inside.yaml",
    )
    assert definition["adapter"] == "videomaev2"


def test_explicit_definition_path_cannot_traverse_outside_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.yaml"
    _write_definition(outside)

    with pytest.raises(ValueError, match="越出 project_root"):
        load_encoder_definition(
            "videomaev2",
            project_root=project_root,
            path="../outside.yaml",
        )


def test_explicit_definition_symlink_cannot_escape_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.yaml"
    _write_definition(outside)
    linked = project_root / "linked.yaml"
    try:
        linked.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows policy can disable symlinks
        pytest.skip(f"当前平台不能创建测试软链：{exc}")

    with pytest.raises(ValueError, match="越出 project_root"):
        load_encoder_definition(
            "videomaev2",
            project_root=project_root,
            path=linked,
        )


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
