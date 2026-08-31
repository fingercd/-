from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from vadbench.benchmark import PERFORMANCE_SCHEMA_VERSION
from vadbench.benchmark_plan import run_benchmark_plan
from vadbench.contracts import EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.video import VideoInfo


class FakeFixed:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        fixed_num_frames=2,
        min_frames=2,
        max_frames=2,
    )

    def encode(self, batch, train=False):
        timeline = TokenTimeline(
            start_s=np.min(batch.timestamps_s, axis=1, keepdims=True),
            end_s=np.max(batch.timestamps_s, axis=1, keepdims=True) + 1.0,
        )
        return EncoderOutput(
            features=np.ones((batch.batch_size, 1, 3), dtype=np.float32),
            pooled=np.ones((batch.batch_size, 3), dtype=np.float32),
            timeline=timeline,
            aux={"feature_stage": "fixed"},
        )


def test_plan_runs_serial_case_and_writes_result(tmp_path: Path) -> None:
    experiment = {
        "schema_version": 1,
        "dataset": {"root": "data", "train_manifest": "train", "test_manifest": "test"},
        "encoder": {"adapter": "fake"},
        "streaming": {"enabled": False},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": "outputs", "run_name": "fake"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    plan = {
        "schema_version": 1,
        "benchmark": {"warmup": 0, "repeat": 2, "device": "cpu", "output": "result.json"},
        "input": {
            "task": "encoder_performance_only",
            "sampling_protocol": "same",
            "video": "video.mp4",
            "sample_fps": 2.0,
            "sampled_frames": 4,
        },
        "cases": [
            {
                "name": "fixed",
                "mode": "fixed",
                "experiment": "experiment.yaml",
                "encoder": {"adapter": "fake"},
                "grouping": {"units": 2, "frames_per_unit": 2},
                "compression": None,
            }
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    created = []

    def factory(config, project_root):
        created.append(config)
        return FakeFixed(), {"name": "fake"}

    info = VideoInfo(path=tmp_path / "video.mp4", num_frames=20, fps=4.0, width=8, height=8)
    result = run_benchmark_plan(
        plan_path,
        project_root=tmp_path,
        encoder_factory=factory,
        probe_fn=lambda path: info,
        decode_fn=lambda path, indices: np.zeros((4, 8, 8, 3), dtype=np.uint8),
    )
    assert result["schema_version"] == PERFORMANCE_SCHEMA_VERSION
    assert len(created) == 1
    assert len(result["cases"][0]["repeats"]) == 2
    assert (tmp_path / "result.json").is_file()
    schema = json.loads(
        Path("schemas/performance-result-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)


def test_plan_rejects_grouping_mismatch(tmp_path: Path) -> None:
    experiment = {
        "schema_version": 1,
        "dataset": {"root": "data", "train_manifest": "train", "test_manifest": "test"},
        "encoder": {"adapter": "fake"},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": "outputs", "run_name": "fake"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    plan = {
        "benchmark": {"warmup": 0, "repeat": 1, "device": "cpu", "output": "out.json"},
        "input": {"video": "video.mp4", "sample_fps": 1.0, "sampled_frames": 4},
        "cases": [
            {
                "name": "bad",
                "mode": "fixed",
                "experiment": "experiment.yaml",
                "encoder": {"adapter": "fake"},
                "grouping": {"units": 1, "frames_per_unit": 2},
            }
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    info = VideoInfo(path=tmp_path / "video.mp4", num_frames=20, fps=4.0, width=8, height=8)
    try:
        run_benchmark_plan(
            path,
            project_root=tmp_path,
            encoder_factory=lambda config, project_root: (FakeFixed(), {}),
            probe_fn=lambda path: info,
            decode_fn=lambda path, indices: np.zeros((4, 8, 8, 3), dtype=np.uint8),
        )
    except ValueError as exc:
        assert "grouping" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected grouping mismatch")
