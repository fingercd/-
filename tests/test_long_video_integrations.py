from __future__ import annotations

import builtins
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vadbench.contracts import ClipBatch, EncoderOutput, StreamState
from vadbench.integrations.long_video.base import (
    DEFAULT_NEUTRAL_PROMPT,
    ExternalPythonWorker,
    LongVideoAssetError,
    LongVideoWorkerError,
)
from vadbench.integrations.long_video.infinipot_v import InfiniPotVAdapter
from vadbench.integrations.long_video.longvu import LongVUAdapter
from vadbench.integrations.long_video.ma_lmm import MALMMAdapter
from vadbench.integrations.long_video.moviechat import MovieChatAdapter
from vadbench.integrations.long_video.mukv import MuKVAdapter
from vadbench.integrations.long_video.streaming_vlm import StreamingVLMAdapter
from vadbench.integrations.long_video.videochat import VideoChatAdapter
from vadbench.integrations.long_video.videochat_flash import VideoChatFlashAdapter
from vadbench.integrations.long_video.videochat_online import VideoChatOnlineAdapter
from vadbench.registry import EncoderRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXED_TARGETS = (
    (
        "longvu",
        LongVUAdapter,
        "vadbench.integrations.long_video.longvu:LongVUAdapter",
        "projected_visual",
    ),
    (
        "videochat",
        VideoChatAdapter,
        "vadbench.integrations.long_video.videochat:VideoChatAdapter",
        "projected_visual",
    ),
    (
        "videochat_flash",
        VideoChatFlashAdapter,
        "vadbench.integrations.long_video.videochat_flash:VideoChatFlashAdapter",
        "projected_visual",
    ),
)

STREAM_TARGETS = (
    (
        "videochat_online",
        VideoChatOnlineAdapter,
        "vadbench.integrations.long_video.videochat_online:VideoChatOnlineAdapter",
        "visual_memory",
    ),
    (
        "ma_lmm",
        MALMMAdapter,
        "vadbench.integrations.long_video.ma_lmm:MALMMAdapter",
        "visual_memory",
    ),
    (
        "moviechat",
        MovieChatAdapter,
        "vadbench.integrations.long_video.moviechat:MovieChatAdapter",
        "visual_memory",
    ),
    (
        "streaming_vlm",
        StreamingVLMAdapter,
        "vadbench.integrations.long_video.streaming_vlm:StreamingVLMAdapter",
        "decoder_contextual",
    ),
    (
        "infinipot_v",
        InfiniPotVAdapter,
        "vadbench.integrations.long_video.infinipot_v:InfiniPotVAdapter",
        "decoder_contextual",
    ),
    (
        "mukv",
        MuKVAdapter,
        "vadbench.integrations.long_video.mukv:MuKVAdapter",
        "decoder_contextual",
    ),
)


def _batch(*, start: int = 0, video_id: str = "surveillance") -> ClipBatch:
    frames = np.arange(1 * 4 * 6 * 8 * 3, dtype=np.uint8).reshape(1, 4, 6, 8, 3)
    return ClipBatch(
        frames=frames,
        timestamps_s=np.array(
            [[start * 0.25, (start + 1) * 0.25, (start + 2) * 0.25, (start + 3) * 0.25]],
            dtype=np.float64,
        ),
        frame_indices=np.array([[start, start + 1, start + 2, start + 3]], dtype=np.int64),
        video_ids=(video_id,),
        metadata={"source": "test"},
    )


class _FixedWorker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        batch: ClipBatch,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "shape": tuple(batch.frames.shape),
                "dtype": str(batch.frames.dtype),
                "prompt": prompt,
                "feature_stage": feature_stage,
            }
        )
        return {
            "features": np.arange(1 * 3 * 5, dtype=np.float32).reshape(1, 3, 5),
            "aux": {"worker": "fake-fixed"},
        }


class _StreamWorker:
    def init_state(
        self,
        video_id: str,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        return {
            "video_id": video_id,
            "seen": 0,
            "prompt": prompt,
            "feature_stage": feature_stage,
        }

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        *,
        prompt: str,
        feature_stage: str,
        compression: Any = None,
    ) -> dict[str, Any]:
        previous = state.opaque or {"seen": 0}
        seen = int(previous["seen"]) + int(chunk.valid_lengths[0])
        return {
            "features": np.full((1, 2, 6), float(seen), dtype=np.float32),
            "state": {
                "seen": seen,
                "prompt": prompt,
                "feature_stage": feature_stage,
            },
            "telemetry": {
                "worker": "fake-stream",
                "compression_is_none": compression is None,
            },
        }

    def finalize(self, state: StreamState) -> None:
        return None


class _CacheWorker(_StreamWorker):
    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        *,
        prompt: str,
        feature_stage: str,
        compression: Any = None,
    ) -> dict[str, Any]:
        result = super().encode_step(
            chunk,
            state,
            prompt=prompt,
            feature_stage=feature_stage,
            compression=compression,
        )
        result["cache"] = np.ones((1, 2, 3), dtype=np.float32)
        return result


class _NonFiniteWorker(_FixedWorker):
    def encode(
        self,
        batch: ClipBatch,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        return {"features": np.array([[[np.nan, 0.0]]], dtype=np.float32)}


@pytest.mark.parametrize(("encoder_id", "adapter_cls", "target", "stage"), FIXED_TARGETS)
def test_fixed_long_video_targets_register_construct_and_normalize_bthwc(
    encoder_id: str,
    adapter_cls: type[Any],
    target: str,
    stage: str,
) -> None:
    worker = _FixedWorker()
    registry = EncoderRegistry()
    registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)

    adapter = registry.create(encoder_id, worker=worker)
    output = adapter.encode(_batch())

    assert isinstance(output, EncoderOutput)
    assert output.features.shape == (1, 3, 5)
    assert output.pooled.shape == (1, 5)
    assert output.timeline.num_tokens == 3
    assert output.aux["integration_id"] == encoder_id
    assert output.aux["feature_stage"] == stage
    assert output.aux["prompt"] == DEFAULT_NEUTRAL_PROMPT
    assert output.aux["implementation_source"] == "external_worker_facade"
    assert np.isfinite(np.asarray(output.features)).all()
    assert worker.calls == [
        {
            "shape": (1, 4, 6, 8, 3),
            "dtype": "uint8",
            "prompt": DEFAULT_NEUTRAL_PROMPT,
            "feature_stage": stage,
        }
    ]


@pytest.mark.parametrize(("encoder_id", "adapter_cls", "target", "stage"), STREAM_TARGETS)
def test_streaming_targets_register_construct_and_advance_explicit_state(
    encoder_id: str,
    adapter_cls: type[Any],
    target: str,
    stage: str,
) -> None:
    registry = EncoderRegistry()
    registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)
    adapter = registry.create(encoder_id, worker=_StreamWorker(), cache_mode="identity")

    state = adapter.init_state("surveillance")
    first = adapter.encode_step(_batch(start=0), state)
    second = adapter.encode_step(_batch(start=4), first.state)

    assert first.state.step_index == 1
    assert second.state.step_index == 2
    assert first.state.opaque["seen"] == 4
    assert second.state.opaque["seen"] == 8
    assert first.output is not None and first.output.features.shape == (1, 2, 6)
    assert second.output is not None and second.output.features.shape == (1, 2, 6)
    assert second.output.aux["feature_stage"] == stage
    assert second.output.aux["prompt"] == DEFAULT_NEUTRAL_PROMPT
    assert second.telemetry["cache_mode"] == "off"
    assert second.telemetry["cache_compression"] == "disabled"
    assert second.state.next_timestamp_s is not None
    assert second.state.next_timestamp_s > first.state.next_timestamp_s
    assert adapter.finalize(second.state) is None


def test_non_identity_cache_policy_is_rejected_without_calling_worker() -> None:
    adapter = MuKVAdapter(worker=_StreamWorker())
    state = adapter.init_state("surveillance")
    with pytest.raises(LongVideoWorkerError) as captured:
        adapter.encode_step(_batch(), state, compression="keep_recent")
    assert captured.value.code == "compression_disabled"
    assert "不支持 KV 压缩" in str(captured.value)


def test_nonfinite_worker_output_fails_closed() -> None:
    adapter = LongVUAdapter(worker=_NonFiniteWorker())
    with pytest.raises(LongVideoWorkerError) as captured:
        adapter.encode(_batch())
    assert captured.value.code == "invalid_output_health"


def test_declared_decoder_cache_is_normalized_when_worker_exposes_it() -> None:
    adapter = StreamingVLMAdapter(worker=_CacheWorker())
    state = adapter.init_state("surveillance")
    step = adapter.encode_step(_batch(), state)
    assert set(step.state.caches) == {"default"}
    cache = step.state.caches["default"]
    assert cache.kind.value == "decoder_kv"
    assert cache.sequence_length == 2
    assert step.cache_updates == {}


def test_missing_checkout_and_checkpoint_fail_closed_with_structured_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(LongVideoAssetError) as captured:
        LongVUAdapter(
            project_root=tmp_path,
            checkout_path="external/missing-longvu",
            model_path="weights/missing-longvu",
        )

    error = captured.value.to_dict()
    assert error["code"] == "missing_asset"
    assert error["integration_id"] == "longvu"
    missing = error["details"]["missing"]
    expected_missing = (
        {"checkpoint"}
        if isinstance(captured.value, LongVideoAssetError)
        else {"checkout", "checkpoint"}
    )
    assert {entry["kind"] for entry in missing} == expected_missing
    assert all(str(tmp_path) in entry["path"] for entry in missing)


def test_explicit_loader_hook_uses_only_local_assets(tmp_path: Path) -> None:
    checkout = tmp_path / "external" / "longvu"
    model_path = tmp_path / "weights" / "longvu"
    checkout.mkdir(parents=True)
    model_path.mkdir(parents=True)
    seen: dict[str, Any] = {}

    def load_model(*, model_path: str, device: str, **_: Any) -> _FixedWorker:
        seen.update({"model_path": model_path, "device": device})
        return _FixedWorker()

    adapter = LongVUAdapter(
        project_root=tmp_path,
        checkout_path=checkout,
        model_path=model_path,
        load_model_fn=load_model,
        device="cpu",
    )
    output = adapter.encode(_batch())

    assert output.features.shape == (1, 3, 5)
    assert seen == {"model_path": str(model_path.resolve()), "device": "cpu"}
    assert adapter.implementation_source == "explicit_loader"


def test_external_python_facade_can_be_injected_without_spawning_process() -> None:
    captured: list[dict[str, Any]] = []

    def runner(request: dict[str, Any]) -> dict[str, Any]:
        captured.append(request)
        return {"features": [[[1.0, 2.0, 3.0]]]}

    worker = ExternalPythonWorker(
        ["unused-python", "unused-worker.py"],
        integration_id="longvu",
        runner=runner,
    )
    output = LongVUAdapter(worker=worker).encode(_batch())

    assert output.features.shape == (1, 1, 3)
    assert captured[0]["protocol"] == "vadbench.external-worker.v1"
    assert captured[0]["operation"] == "encode"
    assert captured[0]["frames"][0][0][0][0] == [0, 1, 2]


def test_worker_runner_alias_bypasses_asset_checks_without_network() -> None:
    def runner(request: dict[str, Any]) -> dict[str, Any]:
        return {"features": [[[0.0, 1.0]]]} if request["operation"] == "encode" else {}

    adapter = VideoChatAdapter(worker_runner=runner)
    result = adapter.encode(_batch())
    assert result.features.shape == (1, 1, 2)
    assert adapter.implementation_source == "external_python_worker"


def test_external_worker_runner_stream_round_trip_serializes_opaque_state() -> None:
    operations: list[str] = []

    def runner(request: dict[str, Any]) -> dict[str, Any]:
        operations.append(request["operation"])
        if request["operation"] == "init_state":
            return {"result": {"seen": 0}}
        if request["operation"] == "encode_step":
            assert request["state"]["opaque"]["seen"] in {0, 4}
            return {
                "result": {
                    "features": [[[1.0, 2.0]]],
                    "state": {"seen": request["state"]["opaque"]["seen"] + 4},
                }
            }
        return {"result": None}

    adapter = StreamingVLMAdapter(worker_runner=runner)
    state = adapter.init_state("surveillance")
    step = adapter.encode_step(_batch(), state)
    assert step.state.opaque["seen"] == 4
    assert operations == ["init_state", "encode_step"]


def test_long_video_modules_remain_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] in {"torch", "transformers", "decord", "timm"}:
            raise AssertionError(f"long-video adapter imported heavy dependency {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    registry = EncoderRegistry()
    for encoder_id, adapter_cls, target, _ in (*FIXED_TARGETS, *STREAM_TARGETS):
        registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)
        registry.load_factory(encoder_id)
    assert set(registry.names()) == {
        *(item[0] for item in FIXED_TARGETS),
        *(item[0] for item in STREAM_TARGETS),
    }


def test_all_long_video_configs_and_locks_are_pinned_and_consistent() -> None:
    config_names = {
        "longvu": "longvu.yaml",
        "videochat": "videochat.yaml",
        "videochat_online": "videochat-online.yaml",
        "videochat_flash": "videochat-flash.yaml",
        "ma_lmm": "ma-lmm.yaml",
        "moviechat": "moviechat.yaml",
        "streaming_vlm": "streaming-vlm.yaml",
        "infinipot_v": "infinipot-v.yaml",
        "mukv": "mukv.yaml",
    }
    stages = {item[0]: item[3] for item in (*FIXED_TARGETS, *STREAM_TARGETS)}
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")

    for encoder_id, file_name in config_names.items():
        config_path = PROJECT_ROOT / "configs" / "encoders" / file_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["adapter"] == encoder_id
        assert config["constructor"]["prompt"] == DEFAULT_NEUTRAL_PROMPT
        assert config["constructor"]["feature_stage"] == stages[encoder_id]
        expected_source = (
            "native_upstream"
            if encoder_id in {"longvu", "streaming_vlm", "videochat_flash", "videochat_online"}
            else "external_worker_facade"
        )
        assert config["output"]["implementation_source"] == expected_source
        assert config["cache_semantics"]["compression"] == "disabled"

        lock_path = PROJECT_ROOT / config["upstream_lock"]
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        assert lock["integration"] == encoder_id
        assert commit_pattern.fullmatch(lock["source"]["commit"])
        assert lock["source"]["commit"] in lock["source"]["commit_url"]
        expected_weight_status = (
            "verified"
            if encoder_id in {"longvu", "streaming_vlm", "videochat_flash", "videochat_online"}
            else "planned"
        )
        assert lock["weights"]["status"] == expected_weight_status
