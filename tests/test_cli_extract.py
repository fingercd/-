from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import vadbench.cli as cli
from vadbench.contracts import ClipBatch, EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl


class FakeAdapter:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_training=False,
        fixed_num_frames=2,
        min_frames=2,
        max_frames=2,
    )

    def encode(self, batch, train=False):
        assert not train
        timeline = TokenTimeline(
            start_s=np.asarray([[0.0]], dtype=np.float32),
            end_s=np.asarray([[1.0]], dtype=np.float32),
            valid_mask=np.asarray([[True]]),
            source_frame_start=np.asarray([[0]], dtype=np.int64),
            source_frame_end=np.asarray([[2]], dtype=np.int64),
        )
        return EncoderOutput(
            features=np.ones((batch.batch_size, 1, 2), dtype=np.float32),
            pooled=np.ones((batch.batch_size, 2), dtype=np.float32),
            timeline=timeline,
        )


def test_cli_extract_builds_feature_index(tmp_path: Path, monkeypatch, capsys) -> None:
    record = VideoManifestRecord(
        video_id="normal",
        path="Normal/normal.mp4",
        split="train",
        category="Normal",
        is_anomaly=False,
    )
    manifest = write_manifest_jsonl((record,), tmp_path / "train.jsonl")
    config = {
        "schema_version": 1,
        "dataset": {
            "root": str(tmp_path),
            "train_manifest": str(manifest),
            "test_manifest": str(manifest),
        },
        "sampler": {"segments_per_video": 1, "clip_frames": 2, "frame_stride": 1},
        "encoder": {"adapter": "fake", "checkpoint": "fake", "micro_batch_size": 1},
        "streaming": {"enabled": False},
        "task": {"supervision": "video"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "extract"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    batch = ClipBatch(
        frames=np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0.0, 1.0]], dtype=np.float32),
        video_ids=("normal",),
        frame_indices=np.asarray([[0, 1]], dtype=np.int64),
        metadata={"clip_ids": ["normal:0"], "clip_indices": [0]},
    )
    monkeypatch.setattr(
        cli,
        "create_encoder_from_experiment",
        lambda config, project_root: (
            FakeAdapter(),
            {"adapter": "fake", "constructor": {}, "checkpoint": {}},
        ),
    )
    monkeypatch.setattr(cli, "iter_fixed_segment_batches", lambda *args, **kwargs: iter((batch,)))

    assert cli.main(["extract", "-c", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["feature_records"] == 1
    assert Path(payload["feature_index"]).is_file()
