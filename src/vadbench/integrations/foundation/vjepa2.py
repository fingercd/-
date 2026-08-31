"""Meta V-JEPA 2 fixed-clip integration."""

from __future__ import annotations

from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


class VJEPA2Adapter(FoundationVideoAdapter):
    """Adapt a V-JEPA 2 video encoder to the canonical feature contract."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "vjepa2"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "vjepa2-bthwc-v1"
    DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
    DEFAULT_RUNTIME = "external_python"


# Common capitalization variants used by downstream configuration code.
VJepa2Adapter = VJEPA2Adapter
VJepaAdapter = VJEPA2Adapter

__all__ = ["VJEPA2Adapter", "VJepa2Adapter", "VJepaAdapter"]
