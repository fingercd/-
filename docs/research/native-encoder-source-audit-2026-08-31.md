# 剩余路线原生接入来源审计（2026-08-31）

本审计只记录“可作为原生接入起点”的一手来源和当前风险，不把代码仓库存在等同于权重可下载。每个 checkpoint 在服务器下载后必须固定 revision、文件清单、SHA256 和许可证；没有这些证据不能标记 `verified`。

## 已确认可执行的来源

| ID | 官方代码 | 原生权重/入口 | 计划中的真实输出 | 关键风险 |
|---|---|---|---|---|
| `c3d` | [facebookarchive/C3D](https://github.com/facebookarchive/C3D) | [MMAction C3D model zoo](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) 的 Sports-1M 转换 checkpoint，或 Dartmouth 原始模型 | C3D `fc6/fc7`，`[B,S,D]` | Caffe/旧版运行时；CC-BY-NC；必须核验转换权重确实为 C3D |
| `timesformer` | [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer) | [facebook/timesformer-base-finetuned-k400](https://huggingface.co/facebook/timesformer-base-finetuned-k400)，8 帧、224、约 486 MB | `TimesformerModel` 的 hidden state 或 classifier 前特征 | 官方仓库归档；HF 权重 CC-BY-NC-4.0；不能只读 logits |
| `videomae` | [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base)，94.2M 参数 | `last_hidden_state`/CLS，`[B,S,D]` | 预训练模型默认可能调用 reconstruction decoder；需明确只导出 encoder 表征 |
| `uniformerv2` | [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2) | 官方 [MODEL_ZOO](https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md) 的 K400 B/16 8×3×4 checkpoint（OSS 链接） | UniFormerV2 backbone/projector 前特征 | OSS 链接、实际运行 config 和许可证需逐个冻结 |
| `umt` | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) | 官方 `single_modality/MODEL_ZOO.md` 当前链接 | UMT backbone token/pooled | 官方 Issue [#50](https://github.com/OpenGVLab/unmasked_teacher/issues/50) 报告 checkpoint 链接失效；失效则保留 blocked，不替代 |
| `internvideo2` | [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo) 的 `InternVideo2` | [OpenGVLab/InternVideo2-Stage2_1B-224p-f4](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) | 只调用 vision encoder，输出 backbone token/pooled | 1B 模型和代码依赖较重；不能把 LLM logits 当视觉特征 |
| `videomamba` | [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba) | [OpenGVLab/VideoMamba](https://huggingface.co/OpenGVLab/VideoMamba) 的 `videomamba_t16_k400_f16_res224.pth` 或官方 Tiny 入口 | classifier 前的 Mamba video feature | CUDA/自定义 selective-scan；内部 SSM 不自动代表跨 chunk state |
| `vjepa2` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | Meta 直链 `https://dl.fbaipublicfiles.com/vjepa2/vitl-256.pt` 或 [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | encoder 输出（不使用 predictor/head） | 需要 decord/timm/einops；官方 README 明确要求本地下载 checkpoint |
| `longvu` | [Vision-CAIR/LongVU](https://github.com/Vision-CAIR/LongVU) | [Vision-CAIR/LongVU_Qwen2_7B](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B)，或预检确认可用的较小官方视频变体 | DINOv2/SigLIP→projector 的视觉 token | 官方 README 写明本地 demo 最低约 40 GB GPU；必须拆出 visual path，不把生成文本作为 encoder |
| `videochat` | [OpenGVLab/Ask-Anything](https://github.com/OpenGVLab/Ask-Anything) | 以官方 README 的 VideoChat/VideoChat2 模型链接为准，先确认实际 checkpoint 文件 | InternVideo/UMT 视觉输出或 VideoChat projector 输出 | 原始 VideoChat 依赖 Vicuna/旧栈；若官方链接只有 Space、无可下载权重，直接 blocked |
| `videochat_online` | [MCG-NJU/VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online) | [MCG-NJU/VideoChatOnline-4B](https://huggingface.co/MCG-NJU/VideoChatOnline-4B)，8.29 GB，HF license unknown；README 仅有 MIT badge，仓库根目录无 LICENSE 文件 | 官方 ViT features + Pyramid Memory Bank，`visual_memory` | README 明确当前开源实现基于 ViT feature，KV-cache 版本尚未提供；不能标成 decoder KV |
| `videochat_flash` | [OpenGVLab/VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash) | [VideoChat-Flash-Qwen2_5-2B_res448](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448)，Apache-2.0 | `get_vision_tower()`/projected visual，关闭其压缩开关 | 官方示例需要 `trust_remote_code`、Transformers 4.40.1 和可选 flash-attn |
| `ma_lmm` | [boheumd/MA-LMM](https://github.com/boheumd/MA-LMM) | 官方 `saved_model.tar` + InstructBLIP/LAVIS/Vicuna 依赖 | Q-Former memory bank，`visual_memory` | 权重分散在外部链接；多份上游许可证，必须分别登记 |
| `moviechat` | [wenhaochai/MovieChat](https://github.com/wenhaochai/MovieChat) | 官方 README 的 MovieChat/MovieChat-Onevision 权重及显式 base model | short/long visual memory | BSD、LAVIS、MiniGPT-4、VideoLLaMA 等多许可证；不能自动下载 |
| `streaming_vlm` | [mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm) | [mit-han-lab/StreamingVLM](https://huggingface.co/mit-han-lab/StreamingVLM)，8B BF16、4 shard | 官方 streaming compact decoder KV | 约 16.6 GB 权重；官方推理环境与 SFT 环境分开 |
| `infinipot_v` | [aiha-lab/InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V) | 官方代码支持 Qwen2-VL/Qwen2.5-VL；使用对应公开 base checkpoint | 原生 continual KV compression 前后的 decoder KV | 官方仓库自称研究复现且不含论文全部组件；当前仓库未见明确 LICENSE，法律门禁优先 |
| `mukv` | [IMBALDY/MuKV](https://github.com/IMBALDY/MuKV) | 官方 README 指定 [LLaVA-OneVision 0.5B](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) base | `kvcache_utils.py` 的 multi-grained decoder KV | 仓库当前未显示清晰 LICENSE；先记录授权，再运行/提交其代码 |

## 结论

- 17 条路线都存在真实的“原生接入路径”，但可执行性不是均等的：HF 权重（TimeSformer、VideoMAE、VideoChat-Online、VideoChat-Flash、StreamingVLM、LongVU、V-JEPA2）优先；OSS/Google Drive/Caffe/旧 CUDA 路线必须逐项预检。
- VideoChat-Online 的当前公开代码只能作为 `visual_memory`；当前原生两 chunk 已通过，但因仓库根目录无 LICENSE 文件进入 `blocked`；只有许可审计和未来官方 KV 实现均满足时才可升级。
- InfiniPot-V 和 MuKV 即使代码可见，也不能在许可证未确认时复制进主树或标记可再分发。
- 兼容桥只能保留为独立的框架契约测试，不能出现在原生 smoke 的 checkpoint、状态或 PASS 统计中。
