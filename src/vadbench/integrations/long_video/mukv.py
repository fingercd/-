"""MuKV decoder-context integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class MuKVAdapter(ExternalStreamingVideoAdapter):
    """Expose MuKV's decoder-contextual stream representation without compression."""

    integration_id = "mukv"
    default_feature_stage = "decoder_contextual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/mukv"
    default_model_path = "weights/mukv"
    backend = "mukv"
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


DEFAULT_CAPABILITIES = MuKVAdapter.capabilities
FEATURE_STAGE = MuKVAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "MuKVAdapter"]
