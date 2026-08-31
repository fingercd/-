"""LongVU integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalFixedVideoAdapter


class LongVUAdapter(ExternalFixedVideoAdapter):
    """Bridge LongVU's projected visual representation into VADBench."""

    integration_id = "longvu"
    default_feature_stage = "projected_visual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/longvu"
    default_model_path = "weights/longvu"
    backend = "longvu"
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


DEFAULT_CAPABILITIES = LongVUAdapter.capabilities
FEATURE_STAGE = LongVUAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "LongVUAdapter"]
