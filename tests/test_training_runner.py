from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vadbench.data.manifest import VideoManifestRecord
from vadbench.engine.runner import (
    TORCH_AVAILABLE,
    HeadOnlyTrainingConfig,
    train_feature_head,
)
from vadbench.features import FeatureStore, compute_encoder_fingerprint


def _manifest(video_id: str, *, anomaly: bool, split: str) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=20,
        fps=10.0,
        duration_seconds=2.0,
    )


def _store_dataset(tmp_path: Path) -> tuple[FeatureStore, list, list, str]:
    store = FeatureStore(tmp_path / "features")
    fingerprint = compute_encoder_fingerprint({"adapter": "runner-test"})
    train = [_manifest(f"train-{index}", anomaly=index >= 2, split="train") for index in range(4)]
    validation = [_manifest(f"val-{index}", anomaly=bool(index), split="val") for index in range(2)]
    for record in [*train, *validation]:
        for clip_index in range(2):
            base = float(record.is_anomaly) * 3.0 + clip_index
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:clip-{clip_index}",
                clip_index=clip_index,
                encoder_fingerprint=fingerprint,
                features=np.full((2, 4), base, dtype=np.float32),
                pooled=np.full(4, base, dtype=np.float32),
                start_s=float(clip_index),
                end_s=float(clip_index + 1),
                metadata={
                    "source": {
                        "split": record.split.value,
                        "is_anomaly": record.is_anomaly,
                        "sampling_kind": "uniform_segments",
                        "segment_start_frames": clip_index * 10,
                        "segment_end_frames": (clip_index + 1) * 10,
                    }
                },
            )
    return store, train, validation, fingerprint


def test_config_rejects_fake_encoder_finetuning() -> None:
    with pytest.raises(ValueError, match="head-only"):
        HeadOnlyTrainingConfig.from_mapping(
            {"task": {"kind": "weak_mil"}, "encoder": {"trainable": True}}
        )


def test_nested_config_maps_task_head_and_training_options() -> None:
    config = HeadOnlyTrainingConfig.from_mapping(
        {
            "task": {"kind": "weak_mil", "pooling": "topk", "top_k": 2},
            "training": {"epochs": 3, "batch_size": 4, "lr": 0.002, "max_steps": 5},
            "sampler": {"segments_per_video": 32},
            "encoder": {"trainable": False},
        }
    )
    assert config.task == "weak_mil"
    assert config.head == "topk"
    assert config.head_kwargs == {"k": 2}
    assert config.learning_rate == 0.002
    assert config.max_steps == 5
    assert config.expected_clips == 32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_runner_trains_validates_and_saves_checkpoint_history(tmp_path: Path) -> None:
    store, train, validation, fingerprint = _store_dataset(tmp_path)
    result = train_feature_head(
        {
            "task": {"kind": "weak_mil", "pooling": "attention"},
            "training": {
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": 0.01,
                "max_steps": 3,
                "seed": 4,
            },
            "sampler": {"segments_per_video": 2},
            "encoder": {"trainable": False},
        },
        feature_store=store,
        train_manifest=train,
        validation_manifest=validation,
        output_dir=tmp_path / "run",
        encoder_fingerprint=fingerprint,
        device="cpu",
    )
    assert result.global_step == 3
    assert result.epochs_completed == 2
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.checkpoint_manifest_path).is_file()
    assert Path(result.history_path).is_file()
    history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
    assert history["status"] == "max_steps_reached"
    assert history["encoder_fingerprint"] == fingerprint
    assert len(history["epochs"]) == 2
    assert all("validation" in epoch for epoch in history["epochs"])
    assert history["checkpoint"]["step"] == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_runner_rejects_strong_training_without_temporal_truth(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    fingerprint = compute_encoder_fingerprint({"adapter": "strong-test"})
    manifest = [_manifest("video", anomaly=True, split="train")]
    store.write(
        video_id="video",
        clip_id="video:clip-0",
        clip_index=0,
        encoder_fingerprint=fingerprint,
        features=np.ones((1, 3), dtype=np.float32),
        pooled=np.ones(3, dtype=np.float32),
        start_s=0,
        end_s=1,
        metadata={"source": {"split": "train", "is_anomaly": True}},
    )
    with pytest.raises(ValueError, match="no manifest videos"):
        train_feature_head(
            HeadOnlyTrainingConfig(task="temporal_supervised"),
            feature_store=store,
            train_manifest=manifest,
            output_dir=tmp_path / "run",
            device="cpu",
        )
