"""VideoChat-Online visual-memory integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class VideoChatOnlineAdapter(ExternalStreamingVideoAdapter):
    """Expose VideoChat-Online's pyramid memory as explicit stream state."""

    integration_id = "videochat_online"
    default_feature_stage = "visual_memory"
    feature_stage = default_feature_stage
    default_checkout_path = "external/videochat-online"
    default_model_path = "weights/videochat-online"
    backend = "videochat-online"
    cache_owner = "visual_memory"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=True,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


DEFAULT_CAPABILITIES = VideoChatOnlineAdapter.capabilities
FEATURE_STAGE = VideoChatOnlineAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "VideoChatOnlineAdapter"]
