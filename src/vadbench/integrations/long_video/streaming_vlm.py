"""StreamingVLM decoder-context integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class StreamingVLMAdapter(ExternalStreamingVideoAdapter):
    """Expose the upstream decoder-contextual stream representation."""

    integration_id = "streaming_vlm"
    default_feature_stage = "decoder_contextual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/streaming-vlm"
    default_model_path = "weights/streaming-vlm"
    backend = "streaming-vlm"
    cache_owner = "decoder_kv"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=True,
        supports_token_cache=False,
        supports_visual_memory_cache=False,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


DEFAULT_CAPABILITIES = StreamingVLMAdapter.capabilities
FEATURE_STAGE = StreamingVLMAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "StreamingVLMAdapter"]
