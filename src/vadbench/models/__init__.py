"""Task heads exposed without making PyTorch a base-package dependency."""

from .heads import (
    TORCH_AVAILABLE,
    AttentionMILHead,
    MILAttentionHead,
    MILHeadOutput,
    TemporalClassificationHead,
    TemporalSupervisedHead,
    TopKMILHead,
    mil_ranking_loss,
    ranking_loss,
    temporal_bce_loss,
    temporal_supervised_loss,
    weakly_supervised_ranking_loss,
)

__all__ = [
    "AttentionMILHead",
    "MILAttentionHead",
    "MILHeadOutput",
    "TORCH_AVAILABLE",
    "TemporalClassificationHead",
    "TemporalSupervisedHead",
    "TopKMILHead",
    "mil_ranking_loss",
    "ranking_loss",
    "temporal_bce_loss",
    "temporal_supervised_loss",
    "weakly_supervised_ranking_loss",
]
