"""UMT fixed-clip integration.

Only the light-weight bridge is imported here.  The official UMT checkout and
its model dependencies are resolved through an explicit local loader at runtime.
"""

from __future__ import annotations

from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


class UMTAdapter(FoundationVideoAdapter):
    """Adapt an UMT visual backbone to VADBench's BTHWC contract."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "umt"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "umt-bthwc-v1"
    DEFAULT_MODEL_NAME = "OpenGVLab/UMT-B"
    DEFAULT_RUNTIME = "external_python"


UmtAdapter = UMTAdapter

__all__ = ["UMTAdapter", "UmtAdapter"]
