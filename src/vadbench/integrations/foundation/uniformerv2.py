"""UniFormerV2 fixed-clip integration."""

from __future__ import annotations

from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


class UniFormerV2Adapter(FoundationVideoAdapter):
    """Adapt the official UniFormerV2 video backbone through the shared bridge."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "uniformerv2"
    FEATURE_STAGE = "pooled"
    PREPROCESS_PROFILE = "uniformerv2-bthwc-v1"
    DEFAULT_MODEL_NAME = "OpenGVLab/UniFormerV2-B/16"
    DEFAULT_RUNTIME = "external_python"


UniformerV2Adapter = UniFormerV2Adapter

__all__ = ["UniFormerV2Adapter", "UniformerV2Adapter"]
