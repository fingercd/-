"""Verified-public-checkpoint compatibility bridge for route smoke tests.

Several catalog routes (legacy C3D, foundation backbones, and long-video VLMs)
do not have a usable local upstream checkout in the shared server image. The
framework still needs to exercise their real BTHWC/output/streaming contracts
without silently downloading or creating random weights. This module provides
an explicit bridge around the verified TorchVision R(2+1)D checkpoint already
materialised in ``weights/r2plus1d_18/model.pth``.

The bridge is intentionally opt-in through ``compatibility_bridge`` in an
encoder definition. Every output records the requested route, base checkpoint,
and ``native_route_available=false`` so a smoke PASS cannot be confused with a
claim that the route's original architecture has been reproduced. It is a
real public-weight forward and therefore remains useful for validating the
shared sampler, timeline, cache-state, worker, and JSON artifact plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    CacheKind,
    CacheUpdate,
    CacheView,
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    StreamingVideoEncoderAdapter,
    StreamState,
    StreamStep,
    TokenTimeline,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
)
from vadbench.integrations.common import normalize_feature_tensor
from vadbench.integrations.long_video.base import build_token_timeline
from vadbench.integrations.torchvision_video import TorchvisionVideoAdapter

BASE_CHECKPOINT = "weights/r2plus1d_18/model.pth"
BASE_VARIANT = "r2plus1d_18"

FIXED_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=True,
    supports_streaming=False,
    supports_kv_cache=False,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=False,
    supports_training=True,
    min_frames=1,
)

_STREAMING_KINDS: dict[str, CacheKind] = {
    "videochat_online": CacheKind.VISUAL_MEMORY,
    "ma_lmm": CacheKind.VISUAL_MEMORY,
    "moviechat": CacheKind.VISUAL_MEMORY,
    "streaming_vlm": CacheKind.DECODER_KV,
    "infinipot_v": CacheKind.DECODER_KV,
    "mukv": CacheKind.DECODER_KV,
    "hermes_llava_ov": CacheKind.DECODER_KV,
}


def _as_path(value: str | Path | None, project_root: Path) -> Path:
    selected = Path(value or BASE_CHECKPOINT).expanduser()
    if not selected.is_absolute():
        selected = project_root / selected
    return selected.resolve()


def _torch_cat(left: Any, right: Any) -> Any:
    if type(left).__module__.split(".", 1)[0] == "torch":
        import torch

        return torch.cat((left, right), dim=1)
    return np.concatenate((np.asarray(left), np.asarray(right)), axis=1)


def _timeline_concat(left: TokenTimeline, right: TokenTimeline) -> TokenTimeline:
    def join(name: str) -> np.ndarray | None:
        first = getattr(left, name)
        second = getattr(right, name)
        if first is None or second is None:
            return None
        return np.concatenate((np.asarray(first), np.asarray(second)), axis=1)

    return TokenTimeline(
        start_s=np.concatenate((np.asarray(left.start_s), np.asarray(right.start_s)), axis=1),
        end_s=np.concatenate((np.asarray(left.end_s), np.asarray(right.end_s)), axis=1),
        valid_mask=join("valid_mask"),
        source_frame_start=join("source_frame_start"),
        source_frame_end=join("source_frame_end"),
    )


def _pad_short_batch(batch: ClipBatch, minimum_frames: int = 8) -> ClipBatch:
    """Pad tiny streaming chunks for the 3-D CNN, preserving original metadata."""

    lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
    target = max(int(batch.num_frames), minimum_frames)
    if target == batch.num_frames and np.all(lengths == batch.num_frames):
        return batch
    frames = np.asarray(batch.frames)
    timestamps = np.asarray(batch.timestamps_s, dtype=np.float64)
    indices = (
        np.broadcast_to(
            np.arange(batch.num_frames, dtype=np.int64),
            (batch.batch_size, batch.num_frames),
        ).copy()
        if batch.frame_indices is None
        else np.asarray(batch.frame_indices, dtype=np.int64).copy()
    )
    padded_frames = np.empty((batch.batch_size, target, *frames.shape[2:]), dtype=np.uint8)
    padded_times = np.empty((batch.batch_size, target), dtype=np.float64)
    padded_indices = np.empty((batch.batch_size, target), dtype=np.int64)
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length <= 0:
            raise ValueError("ClipBatch 至少包含一帧有效帧")
        padded_frames[row, : batch.num_frames] = frames[row]
        padded_times[row, : batch.num_frames] = timestamps[row]
        padded_indices[row, : batch.num_frames] = indices[row]
        delta = 1.0 / 30.0
        if length > 1:
            positive = np.diff(timestamps[row, :length])
            positive = positive[positive > 0]
            if positive.size:
                delta = float(np.median(positive))
        for slot in range(batch.num_frames, target):
            padded_frames[row, slot] = frames[row, length - 1]
            padded_times[row, slot] = padded_times[row, slot - 1] + delta
            padded_indices[row, slot] = padded_indices[row, slot - 1] + 1
    return ClipBatch(
        frames=padded_frames,
        timestamps_s=padded_times,
        video_ids=batch.video_ids,
        frame_indices=padded_indices,
        metadata={**dict(batch.metadata), "compatibility_padding": target},
    )


class CompatibilityFixedAdapter(VideoEncoderAdapter):
    """Run one route through a verified public TorchVision checkpoint."""

    capabilities = FIXED_CAPABILITIES

    def __init__(
        self,
        *,
        integration_id: str,
        project_root: str | Path = ".",
        model_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        image_size: int = 112,
        variant: str = BASE_VARIANT,
        requested_feature_stage: str = "pooled",
        requested_backend: str = "compatibility",
        requested_model_name: str | None = None,
        **_: Any,
    ) -> None:
        self.integration_id = str(integration_id)
        self.project_root = Path(project_root).expanduser().resolve()
        self.checkpoint_path = _as_path(checkpoint_path or model_path, self.project_root)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"兼容桥要求已下载的公开 checkpoint：{self.checkpoint_path}")
        self.device = str(device)
        self.requested_feature_stage = str(requested_feature_stage)
        self.requested_backend = str(requested_backend)
        self.requested_model_name = requested_model_name
        self.base = TorchvisionVideoAdapter(
            variant=variant,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            image_size=int(image_size),
            clip_frames=None,
            feature_stage="pooled",
            strict_checkpoint=False,
        )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        model_batch = _pad_short_batch(batch)
        base_output = self.base.encode(model_batch, train=train)
        sequence = normalize_feature_tensor(base_output.features, batch_size=batch.batch_size)
        pooled = base_output.pooled
        timeline = build_token_timeline(batch, int(sequence.shape[1]))
        aux = dict(base_output.aux)
        aux.update(
            {
                "integration_id": self.integration_id,
                "requested_backend": self.requested_backend,
                "requested_model_name": self.requested_model_name,
                "requested_feature_stage": self.requested_feature_stage,
                "native_route_available": False,
                "compatibility_bridge": "torchvision-r2plus1d_18",
                "compatibility_checkpoint": BASE_CHECKPOINT,
                "implementation_source": "verified_public_checkpoint_compatibility_bridge",
                "input_layout": "BTHWC",
            }
        )
        output = EncoderOutput(features=sequence, pooled=pooled, timeline=timeline, aux=aux)
        validate_encoder_output(output, batch)
        return output


class CompatibilityStreamingAdapter(StreamingVideoEncoderAdapter):
    """Expose explicit state/cache progression over the same real checkpoint."""

    def __init__(
        self,
        *,
        integration_id: str,
        capabilities: EncoderCapabilities,
        cache_kind: CacheKind,
        **kwargs: Any,
    ) -> None:
        self.integration_id = str(integration_id)
        self.capabilities = capabilities
        self.cache_kind = cache_kind
        self.fixed = CompatibilityFixedAdapter(integration_id=integration_id, **kwargs)

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        # Streaming adapters still satisfy VideoEncoderAdapter's abstract
        # method; callers should use encode_step so state/cache progression is
        # observable.
        return self.fixed.encode(batch, train=train)

    def init_state(self, video_id: str) -> StreamState:
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id 必须是非空字符串")
        return StreamState(
            video_id=video_id,
            metadata={
                "integration_id": self.integration_id,
                "cache_kind": self.cache_kind.value,
                "cache_mode": "identity",
                "native_route_available": False,
            },
        )

    def _view(self, output: EncoderOutput) -> CacheView:
        features = output.features
        if self.cache_kind is CacheKind.DECODER_KV:
            tensors = {"key": features, "value": features}
        else:
            tensors = {"features": features}
        return CacheView(
            kind=self.cache_kind,
            tensors=tensors,
            sequence_axis=1,
            timeline=output.timeline,
            metadata={
                "integration_id": self.integration_id,
                "compression": "disabled",
                "compatibility_bridge": True,
            },
        )

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: Any = None,
    ) -> StreamStep:
        validate_clip_for_capabilities(chunk, self.capabilities, streaming=True, train=train)
        if state.video_id != chunk.video_ids[0]:
            raise ValueError("chunk.video_ids 必须与 state.video_id 一致")
        if compression not in (None, False, True, "off", "identity"):
            if isinstance(compression, Mapping):
                name = compression.get("name", compression.get("policy"))
            else:
                name = getattr(compression, "name", compression)
            if name not in (None, False, True, "off", "identity"):
                raise ValueError("兼容桥只接受 off/identity，不执行缓存压缩")
        output = self.fixed.encode(chunk, train=train)
        update_view = self._view(output)
        caches = dict(state.caches)
        previous = caches.get("default")
        if previous is None:
            merged = update_view
        else:
            merged = CacheView(
                kind=self.cache_kind,
                tensors={
                    name: _torch_cat(previous.tensors[name], update_view.tensors[name])
                    for name in update_view.tensors
                },
                sequence_axis=1,
                timeline=_timeline_concat(previous.timeline, update_view.timeline),
                metadata=update_view.metadata,
            )
        caches["default"] = merged
        valid_length = int(chunk.valid_lengths[0])
        times = np.asarray(chunk.timestamps_s)[0, :valid_length]
        next_time = float(times[-1])
        if valid_length > 1:
            positive = np.diff(times)
            positive = positive[positive > 0]
            if positive.size:
                next_time += float(np.median(positive))
        next_state = state.replace(
            step_index=state.step_index + 1,
            caches=caches,
            next_timestamp_s=next_time,
            opaque={"chunks_seen": state.step_index + 1},
        )
        telemetry = {
            "cache_mode": "identity",
            "cache_kind": self.cache_kind.value,
            "cache_tokens": merged.sequence_length,
            "compatibility_bridge": True,
        }
        return StreamStep(
            output=output,
            state=next_state,
            cache_updates={"default": CacheUpdate.append(update_view)},
            telemetry=telemetry,
        )

    def finalize(self, state: StreamState) -> EncoderOutput | None:
        del state
        return None


def _stream_capabilities(kind: CacheKind) -> EncoderCapabilities:
    return EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=kind is CacheKind.DECODER_KV,
        supports_token_cache=kind is CacheKind.TOKEN,
        supports_visual_memory_cache=kind is CacheKind.VISUAL_MEMORY,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


def create_compatibility_adapter(integration_id: str, **kwargs: Any) -> Any:
    """Construct the explicit bridge selected by an encoder definition."""

    selected = str(integration_id)
    kind = _STREAMING_KINDS.get(selected)
    common = dict(kwargs)
    common.setdefault("project_root", ".")
    common.setdefault("device", "cpu")
    common.setdefault("model_path", BASE_CHECKPOINT)
    # Legacy definitions may retain their native ``checkpoint_path`` beside
    # the compatibility asset.  Once an explicit model_path is selected, the
    # bridge must not accidentally try to open that absent Caffe/HF path.
    if common.get("model_path") is not None:
        common["checkpoint_path"] = None
    common.setdefault("requested_backend", common.get("backend", "compatibility"))
    common.setdefault("requested_feature_stage", common.get("feature_stage", "pooled"))
    common.setdefault("requested_model_name", common.get("model_name"))
    if kind is None:
        return CompatibilityFixedAdapter(integration_id=selected, **common)
    return CompatibilityStreamingAdapter(
        integration_id=selected,
        capabilities=_stream_capabilities(kind),
        cache_kind=kind,
        **common,
    )


__all__ = [
    "BASE_CHECKPOINT",
    "BASE_VARIANT",
    "CompatibilityFixedAdapter",
    "CompatibilityStreamingAdapter",
    "FIXED_CAPABILITIES",
    "create_compatibility_adapter",
]
