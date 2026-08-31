"""VideoChat-Flash integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalFixedVideoAdapter


class VideoChatFlashAdapter(ExternalFixedVideoAdapter):
    """Bridge VideoChat-Flash projected visual context."""

    integration_id = "videochat_flash"
    default_feature_stage = "projected_visual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/videochat-flash"
    default_model_path = "weights/videochat-flash"
    backend = "videochat-flash"
    cache_owner = None
    run_mode = "long"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_streaming=False,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=False,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


DEFAULT_CAPABILITIES = VideoChatFlashAdapter.capabilities
FEATURE_STAGE = VideoChatFlashAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "VideoChatFlashAdapter"]
