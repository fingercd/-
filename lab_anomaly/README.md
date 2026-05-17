<div align="right">

[中文介绍](README-CN.md)

</div>

# lab_anomaly — VideoMAE v2 Anomaly Detection Training

> Core training space for **VideoMAE v2 + MIL** based video anomaly detection. This directory houses the model definitions, training loops, and inference runtime used to learn discriminative spatiotemporal representations for anomaly recognition.

---

## What Is Trained Here

The primary objective of this module is to **fine-tune a VideoMAE v2 backbone** (pretrained on large-scale unlabeled video) for the downstream task of **weakly-supervised video anomaly detection**.

### Architecture

- **Backbone**: `OpenGVLab/VideoMAEv2-Base` — a 12-layer spatiotemporal Transformer pretrained via masked autoencoding on millions of video clips.
- **Head**: MIL (Multiple Instance Learning) Attention Pooling — aggregates clip-level features into a video-level anomaly score without requiring frame-level annotations.
- **Loss**: Cross-Entropy classification loss + Temporal Ranking loss — jointly optimizes bag-level classification and temporal anomaly margin.

### Training Strategy

Progressive three-stage fine-tuning is employed for stable transfer learning:

1. **Head-only** (epochs 0–N): Freeze the entire VideoMAE v2 backbone, train only the MIL head.
2. **Partial unfreeze** (epochs N–M): Gradually thaw the top Transformer blocks.
3. **Full unfreeze** (final epochs): Fine-tune the entire network end-to-end with a reduced learning rate.

This staged approach prevents catastrophic forgetting of the rich self-supervised pretraining while adapting the model to anomaly-specific patterns.

---

## Performance Snapshot

Evaluation metrics are automatically serialized to JSON after training. Below are the results from the current best checkpoint:

**End-to-End Classifier — Evaluation Metrics**

```json
{
  "accuracy": 0.9266,
  "precision_anomaly": 0.8897,
  "recall_anomaly": 0.9365,
  "f1_anomaly": 0.9125,
  "auc_binary": 0.9805
}
```

| Metric | Value |
|--------|-------|
| **Accuracy** | **92.66%** |
| **Precision (Anomaly)** | 88.97% |
| **Recall (Anomaly)** | 93.65% |
| **F1-Score (Anomaly)** | 91.25% |
| **AUC (Binary)** | **98.05%** |

Per-category breakdown:
- **Normal**: 734 / 798 correct (91.98%)
- **Steal**: 377 / 401 correct (94.01%)
- **Violent Conflict**: 139 / 150 correct (92.67%)

> 📁 *Raw metrics are saved in `lab_dataset/derived/end2end_classifier/eval_report/eval_metrics.json` and training curves in `history.json`.*

---

## Training History

Key validation milestones from the training log:

| Epoch | Stage | Val Accuracy | Val AUC (Binary) |
|-------|-------|--------------|------------------|
| 0 | unfreeze_2 | 87.41% | 95.28% |
| 1 | unfreeze_2 | 85.56% | 95.58% |

The model converges to strong discriminative performance early in the partial-unfreeze stage, indicating that VideoMAE v2 features are highly transferable to anomaly detection.

---

## What This Enables

Once trained, the checkpoint can be consumed by:

- **Offline evaluation** — scoring pre-recorded videos and generating frame-level anomaly curves.
- **Real-time inference** — sliding-window clip scoring integrated into streaming pipelines.
- **Encoder reuse** — the fine-tuned VideoMAE v2 weights can serve as initialization for other VAD datasets or downstream heads.

---

## Scope Note

This directory is strictly concerned with **model training and inference execution**. Data ingestion, clip preprocessing, and dataset curation are handled upstream.
