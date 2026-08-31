"""Lazy-friendly long-video/VLM integration declarations.

Importing this package only imports the dependency-free adapter facade.  The
actual upstream model packages are loaded only when an adapter is constructed
with an explicit local loader, checkout entrypoint, injected worker, or worker
command.
"""

from .base import (
    DEFAULT_NEUTRAL_PROMPT,
    ExternalAssetError,
    ExternalFixedVideoAdapter,
    ExternalPythonWorker,
    ExternalStreamingVideoAdapter,
    ExternalWorkerError,
    LongVideoAssetError,
    LongVideoWorkerError,
    MissingAssetError,
    MissingLongVideoAssetError,
    StructuredLongVideoError,
    build_token_timeline,
    normalize_long_video_output,
)
from .infinipot_v import InfiniPotVAdapter
from .longvu import LongVUAdapter
from .ma_lmm import MALMMAdapter
from .moviechat import MovieChatAdapter
from .mukv import MuKVAdapter
from .streaming_vlm import StreamingVLMAdapter
from .videochat import VideoChatAdapter
from .videochat_flash import VideoChatFlashAdapter
from .videochat_online import VideoChatOnlineAdapter

__all__ = [
    "DEFAULT_NEUTRAL_PROMPT",
    "build_token_timeline",
    "ExternalAssetError",
    "ExternalFixedVideoAdapter",
    "ExternalPythonWorker",
    "ExternalStreamingVideoAdapter",
    "ExternalWorkerError",
    "InfiniPotVAdapter",
    "LongVUAdapter",
    "LongVideoAssetError",
    "LongVideoWorkerError",
    "MALMMAdapter",
    "MissingAssetError",
    "MissingLongVideoAssetError",
    "normalize_long_video_output",
    "StructuredLongVideoError",
    "MovieChatAdapter",
    "MuKVAdapter",
    "StreamingVLMAdapter",
    "VideoChatAdapter",
    "VideoChatFlashAdapter",
    "VideoChatOnlineAdapter",
]
