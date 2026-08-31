# 25 路视频模型/VLM 原生接入矩阵（纠正版）

> 更新日期：2026-08-31
> 新计划：`docs/plans/2026-08-31-native-encoder-integration-correction.md`
> 来源审计：`docs/research/native-encoder-source-audit-2026-08-31.md`
> 当前视频：`data/smoke/mlvu-surveil-8.mp4`

## 纠正说明

此前 `outputs/encoder-integration/current-video-final/` 中的 25 项矩阵包含 17 条 R(2+1)D 兼容桥结果。它们只属于 `contract_only`，不能证明目标模型已接入，也不再计入原生 PASS。

## 当前汇总

| 原生状态 | 数量 |
|---|---:|
| `smoke_pass` | 14 |
| `planned` | 10 |
| `failed` | 0 |
| `blocked` | 1（VideoChat-Online 无 LICENSE 文件） |

## 原生接入矩阵

| # | ID | 当前原生状态 | 已验证/下一动作 |
|---:|---|---|---|
| 1 | `r2plus1d_18` | `smoke_pass` | TorchVision K400 原生权重，当前视频通过 |
| 2 | `x3d` | `smoke_pass` | PyTorchVideo X3D-S 原生权重，当前视频通过 |
| 3 | `mvitv2` | `smoke_pass` | TorchVision MViTv2-S 原生权重，当前视频通过 |
| 4 | `slowfast` | `smoke_pass` | PyTorchVideo SlowFast-R50 原生权重，当前视频通过 |
| 5 | `c3d` | `planned` | 获取原始/官方转换 Sports-1M C3D 权重；导出 fc6/fc7 |
| 6 | `i3d` | `smoke_pass` | PyTorchVideo I3D-R50 原生权重，当前视频通过 |
| 7 | `timesformer` | `smoke_pass` | `facebook/timesformer-base-finetuned-k400`，固定 8 帧；当前视频真实前向通过 |
| 8 | `video_swin` | `smoke_pass` | TorchVision Swin3D-T 原生权重，当前视频通过 |
| 9 | `videomae` | `smoke_pass` | `MCG-NJU/videomae-base`，输出 `[1,1568,768]`；当前视频真实前向通过 |
| 10 | `videomaev2` | `smoke_pass` | OpenGVLab VideoMAE V2 Base 原生权重，当前视频通过 |
| 11 | `uniformerv2` | `planned` | 固定官方 K400 B/16 model-zoo checkpoint 与运行 config |
| 12 | `umt` | `planned` | 复核官方 model-zoo 失效链接；无权重则 blocked |
| 13 | `internvideo2` | `planned` | 下载 Stage2 1B 224p-f4，构造真实 vision encoder |
| 14 | `videomamba` | `smoke_pass` | 官方 Tiny K400 16-frame checkpoint + pinned checkout；输出 `[1,1,192]`；CPU reference selective scan |
| 15 | `vjepa2` | `smoke_pass` | Meta `facebook/vjepa2-vitl-fpc64-256`，输出 `[1,8192,1024]`；当前视频真实前向通过 |
| 16 | `longvu` | `smoke_pass` | 官方 Qwen2 7B + SigLIP SO400M + DINOv2-Giant + SVA connector；输出 `[1,144,3584]` |
| 17 | `videochat` | `planned` | 核验 Ask-Anything 官方可下载 checkpoint；仅 demo 则 blocked |
| 18 | `videochat_online` | `blocked` | 原生两 chunk 已通过（`[1,304,3072]` → `[1,608,3072]`）；官方仓库无 LICENSE 文件，许可审计未过 |
| 19 | `videochat_flash` | `smoke_pass` | 官方 2B res448 snapshot；只加载视觉塔/projector，关闭 `mm_llm_compress`；输出 `[1,64,1536]` |
| 20 | `ma_lmm` | `planned` | 获取官方 saved_model.tar + InstructBLIP/LAVIS/Vicuna 资产 |
| 21 | `moviechat` | `planned` | 固定 MovieChat/Onevision checkpoint、base model 和多份许可证 |
| 22 | `streaming_vlm` | `planned` | 下载官方 8B BF16 四分片权重，接真实 compact decoder KV |
| 23 | `infinipot_v` | `planned` | 官方研究复现 + Qwen2.5-VL base；许可证未确认则 blocked |
| 24 | `hermes_llava_ov` | `smoke_pass` | HERMES + LLaVA-OneVision 0.5B 原生权重，两 chunk 通过 |
| 25 | `mukv` | `planned` | 使用官方 MuKV 代码 + LLaVA-OneVision 0.5B base；许可证先行 |

## 原生 PASS 门禁

1. 目标自己的上游代码与 checkpoint；
2. checkpoint revision、license、SHA256 与文件大小可追溯；
3. `aux.native_route_available=true`、`aux.implementation_source=native_upstream`；
4. 当前视频真实前向，fixed 至少一 clip，streaming 至少两 chunk；
5. shape、dtype、finite、timeline、state 和 JSON schema 全部通过；
6. `contract_only`、mock、随机权重、其他模型 alias 永远不能计入。
