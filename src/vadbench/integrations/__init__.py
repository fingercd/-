"""Lazy registrations for optional video-encoder integrations.

Importing this package is safe in the minimal ``vadbench`` environment: model
libraries and external checkouts are only imported when a registry entry is
instantiated.
"""

from __future__ import annotations

from typing import Any

from vadbench.contracts import EncoderCapabilities
from vadbench.registry import ENCODER_REGISTRY

VIDEOMAEV2_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=True,
    supports_streaming=False,
    supports_kv_cache=False,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=False,
    supports_training=True,
    fixed_num_frames=16,
    min_frames=16,
    max_frames=16,
)

HERMES_LLAVA_OV_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=False,
    supports_streaming=True,
    supports_kv_cache=True,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=True,
    supports_training=False,
    min_frames=1,
)


def register_builtin_integrations(registry: Any = ENCODER_REGISTRY) -> None:
    """Install built-ins idempotently without resolving optional targets."""

    if "videomaev2" not in registry:
        registry.register_lazy(
            "videomaev2",
            "vadbench.integrations.videomaev2:VideoMAEv2Adapter",
            capabilities=VIDEOMAEV2_CAPABILITIES,
            metadata={
                "family": "fixed_clip",
                "upstream_lock": "integrations/videomaev2/upstream.lock.yaml",
            },
        )
    if "hermes_llava_ov" not in registry:
        registry.register_lazy(
            "hermes_llava_ov",
            "vadbench.integrations.hermes:HermesLlavaOVAdapter",
            capabilities=HERMES_LLAVA_OV_CAPABILITIES,
            metadata={
                "family": "streaming_vlm",
                "cache_owner": "language_model_decoder",
                "upstream_lock": "integrations/hermes/upstream.lock.yaml",
            },
        )


register_builtin_integrations()


__all__ = [
    "HERMES_LLAVA_OV_CAPABILITIES",
    "VIDEOMAEV2_CAPABILITIES",
    "register_builtin_integrations",
]
