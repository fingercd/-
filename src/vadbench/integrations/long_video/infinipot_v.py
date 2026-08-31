"""InfiniPot-V decoder-context integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class InfiniPotVAdapter(ExternalStreamingVideoAdapter):
    """Expose InfiniPot-V stream output while leaving compression disabled."""

    integration_id = "infinipot_v"
    default_feature_stage = "decoder_contextual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/infinipot-v"
    default_model_path = "weights/infinipot-v"
    backend = "infinipot-v"
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


DEFAULT_CAPABILITIES = InfiniPotVAdapter.capabilities
FEATURE_STAGE = InfiniPotVAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "InfiniPotVAdapter"]
