<div align="center">

# Video Anomaly Detection — Research Space

**Exploring 3D Video Encoders & Machine Learning Methods for Robust Anomaly Recognition**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="docs/assets/vad_banner.png" alt="VAD Banner" width="800">
</p>

**Higher Accuracy · Better Generalization · Clearer Interpretability**

[中文介绍](README-CN.md)

</div>

---

## What We Do

This project is a dedicated research space for **Video Anomaly Detection (VAD)**. We systematically explore how modern **3D video encoders** and **machine learning paradigms** can push the boundaries of anomaly recognition accuracy on public benchmarks.

Unlike standard action recognition, anomaly detection demands understanding the **joint evolution of spatial semantics and temporal dynamics** over continuous video streams. Anomalies are rare, diverse, and deeply context-dependent — making this one of the most challenging open problems in computer vision.

Our mission is simple: **build a flexible, extensible training framework where encoders can be swapped, heads can be plugged in, and ideas can be benchmarked quickly.**

---

## Research Directions

### 3D Video Encoder Benchmarking

The choice of spatiotemporal backbone is the single most impactful decision in a VAD pipeline. We evaluate and compare:

- **Self-supervised transformers** (VideoMAE v2, Video Swin) — rich transferable features from large-scale unlabeled video pretraining.
- **Hybrid architectures** (UniFormerV2) — combining local inductive biases with global attention for peak accuracy.
- **Classic baselines** (I3D, SlowFast, R(2+1)D) — ensuring academic rigor through historical comparisons.
- **Next-generation models** (Video Mamba, state-space models) — tackling long-video modeling with linear complexity.
- **Vision-language encoders** (UMT-L, InternVid, Video-LLaVA) — enabling zero-shot and open-vocabulary anomaly detection.

### Weakly-Supervised Learning

Most real-world surveillance data only provides video-level labels (normal vs. abnormal), without frame-level annotations. We focus on:

- Multiple Instance Learning (MIL) and its variants — learning to attend to anomalous snippets within untrimmed videos.
- Ranking and margin losses — separating normal and abnormal temporal dynamics.
- Pseudo-labeling and self-training — iteratively refining frame-level predictions from coarse video-level supervision.

### Transfer & Generalization

- **Cross-dataset evaluation** — training on UCF-Crime, testing on XD-Violence or custom surveillance streams.
- **Pretraining strategies** — leveraging Kinetics, InternVid, and video-text contrastive learning before domain adaptation.
- **Progressive fine-tuning** — stage-wise backbone unfreezing for stable transfer from pretrained weights to target domains.

### Emerging Paradigms

- **Multimodal fusion** — integrating audio cues (explosions, screams) with video for richer anomaly signatures.
- **Explainable VAD** — using vision-language models to generate textual explanations for detected anomalies.
- **Long-range temporal modeling** — moving beyond 16-frame clips to capture slowly unfolding anomalous events.

---

## Datasets of Interest

We benchmark primarily on the following standard VAD datasets:

| Dataset | Setting | Key Characteristics |
|---------|---------|---------------------|
| **UCF-Crime** | Weakly-supervised | 1,900 real-world surveillance videos, 13 anomaly categories |
| **XD-Violence** | Weakly-supervised | 4,754 videos with audio, multi-scene violence detection |
| **ShanghaiTech** | Frame-level GT | 437 videos across 13 campus scenes |
| **CUHK Avenue** | Frame-level GT | 37 videos, pedestrian-focused anomalies |
| **UBnormal** | Synthetic GT | Virtually generated diverse anomalies for data augmentation |

---

## Current Status

- **Baseline established**: VideoMAE v2 + MIL attention with progressive three-stage fine-tuning.
- **Next steps**: Integrate Video Swin Transformer, UniFormerV2, and Video Mamba backbones for head-to-head comparison.

---

## Roadmap

- [x] VideoMAE v2 + MIL baseline
- [ ] Video Swin Transformer integration
- [ ] UniFormerV2 backbone benchmark
- [ ] Video Mamba for long-video anomaly detection
- [ ] UMT-L / InternVid zero-shot evaluation
- [ ] XD-Violence audio-visual fusion
- [ ] Vision-language model explainable VAD

---

## Acknowledgements

This project draws inspiration from the broader VAD research community, including VideoMAE, RTFM, MGFN, VERA, and the Awesome Video Anomaly Detection survey repositories.

---

## License

MIT License.

---

<div align="center">

**⭐ Star this repo if you find the research direction interesting! ⭐**

</div>
