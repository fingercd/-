"""VideoMamba fixed-clip integration."""

from __future__ import annotations

from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


class VideoMambaAdapter(FoundationVideoAdapter):
    """Adapt a VideoMamba backbone; SSM internals are not advertised as cache."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "videomamba"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "videomamba-bthwc-v1"
    DEFAULT_MODEL_NAME = "OpenGVLab/VideoMamba-S"
    DEFAULT_RUNTIME = "external_python"


__all__ = ["VideoMambaAdapter"]
