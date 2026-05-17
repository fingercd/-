<div align="center">

# 视频异常检测 —— 研究空间

**探索三维视频编码器与机器学习方法，实现更精准的异常识别**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="docs/assets/vad_banner.png" alt="VAD Banner" width="800">
</p>

**更高准确率 · 更强泛化性 · 更清晰可解释性**

[English Version](README.md)

</div>

---

## 我们在做什么

本项目是一个专注于**视频异常检测（Video Anomaly Detection, VAD）**的研究空间。我们系统性地探索现代 **3D 视频编码器** 与 **机器学习范式**，在公开基准数据集上不断突破异常识别的准确率上限。

与标准的动作识别不同，异常检测需要理解连续视频流中**空间语义与时序动态的联合演变**。异常事件稀少、种类多样、且高度依赖场景上下文——这使其成为计算机视觉中最具挑战性的开放问题之一。

我们的目标很简单：**构建一个灵活、可扩展的训练框架，让编码器可以随意更换，检测头可以即插即用，新想法能够快速得到验证。**

---

## 研究方向

### 3D 视频编码器基准评测

时空骨干网络的选择是 VAD 流程中最关键的决定。我们评估并对比以下方向：

- **自监督 Transformer**（VideoMAE v2、Video Swin）—— 从大规模无标注视频预训练中获得丰富的可迁移特征。
- **混合架构**（UniFormerV2）—— 结合局部归纳偏置与全局注意力，追求极限精度。
- **经典基线**（I3D、SlowFast、R(2+1)D）—— 通过历史对比确保学术严谨性。
- **下一代模型**（Video Mamba、状态空间模型）—— 以线性复杂度解决长视频建模难题。
- **视觉-语言编码器**（UMT-L、InternVid、Video-LLaVA）—— 实现零样本与开放词汇的异常检测。

### 弱监督学习

大多数真实监控数据仅提供视频级标签（正常 vs. 异常），没有帧级标注。我们聚焦于：

- **多示例学习（MIL）及其变体** —— 在未经修剪的视频中学习关注异常片段。
- **排序与边界损失** —— 拉开正常与异常时序动态的差距。
- **伪标签与自训练** —— 从粗糙的视频级监督中迭代精炼帧级预测。

### 迁移与泛化

- **跨数据集评估** —— 在 UCF-Crime 上训练，在 XD-Violence 或自定义监控流上测试。
- **预训练策略** —— 利用 Kinetics、InternVid 和视频-文本对比学习，再进行领域自适应。
- **渐进式微调** —— 分阶段逐步解冻骨干网络，实现从预训练权重到目标领域的稳定迁移。

### 新兴范式

- **多模态融合** —— 将音频线索（爆炸、尖叫）与视频结合，获得更丰富的异常特征。
- **可解释 VAD** —— 利用视觉-语言模型为检测到的异常生成文本解释。
- **长程时序建模** —— 超越 16 帧片段，捕捉缓慢展开的异常事件。

---

## 关注的数据集

我们主要在以下标准 VAD 数据集上进行基准评测：

| 数据集 | 设定 | 关键特点 |
|--------|------|----------|
| **UCF-Crime** | 弱监督 | 1,900 段真实监控视频，13 类异常事件 |
| **XD-Violence** | 弱监督 | 4,754 段视频，含音频，多场景暴力检测 |
| **ShanghaiTech** | 帧级真值 | 437 段视频，覆盖 13 个校园场景 |
| **CUHK Avenue** | 帧级真值 | 37 段视频，聚焦行人异常 |
| **UBnormal** | 合成真值 | 虚拟生成的多样化异常，用于数据增强 |

---

## 当前进展

- **基线已建立**：VideoMAE v2 + MIL 注意力，配合三阶段渐进式微调。
- **下一步**：接入 Video Swin Transformer、UniFormerV2 和 Video Mamba 骨干网络进行直接对比。

---

## 路线图

- [x] VideoMAE v2 + MIL 基线
- [ ] Video Swin Transformer 接入
- [ ] UniFormerV2 骨干网络基准评测
- [ ] Video Mamba 长视频异常检测
- [ ] UMT-L / InternVid 零样本评估
- [ ] XD-Violence 音视频融合
- [ ] 视觉-语言模型可解释 VAD

---

## 致谢

本项目深受视频异常检测研究社区的启发，包括 VideoMAE、RTFM、MGFN、VERA，以及 Awesome Video Anomaly Detection 综述仓库等优秀工作。

---

## 许可证

MIT 许可证。

---

<div align="center">

**⭐ 如果你对这个研究方向感兴趣，请给本仓库点一颗 Star！⭐**

</div>
