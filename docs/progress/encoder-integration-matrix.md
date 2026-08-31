# 25 路视频模型/VLM 接入矩阵

> 更新日期：2026-08-31
>
> 计划：`docs/plans/2026-08-31-25-video-model-integration.md`
>
> 统一真实视频：`data/smoke/mlvu-surveil-8.mp4`
>
> 状态只允许：`planned / preflight_pass / acquiring / integrated / smoke_pass / failed / blocked`。

## 当前汇总

| 状态 | 数量 |
|---|---:|
| `smoke_pass` | 1 |
| `integrated` | 1 |
| `planned` | 23 |
| `failed` | 0 |
| `blocked` | 0 |

## 接入矩阵

| # | ID | 路线 | 运行时 | 首选 feature stage | 当前状态 | 当前证据/下一动作 |
|---:|---|---|---|---|---|---|
| 1 | `c3d` | fixed | `external_python` | `pooled`/`fc_features` | `planned` | 冻结可运行 Sports1M 权重和 legacy worker |
| 2 | `i3d` | fixed | `in_process` | `pooled` | `planned` | 接入 PyTorchVideo I3D-R50 代表实现 |
| 3 | `r2plus1d_18` | fixed | `in_process` | `pooled` | `planned` | 接入 TorchVision 官方权重 |
| 4 | `slowfast` | fixed | `in_process` | `pooled` | `planned` | 接入 PyTorchVideo 双路径预处理 |
| 5 | `x3d` | fixed | `in_process` | `pooled` | `planned` | 接入最小 X3D 公开变体 |
| 6 | `timesformer` | fixed | `in_process` | `backbone_tokens` | `planned` | 接入 Transformers 官方公开权重 |
| 7 | `video_swin` | fixed | `in_process` | `backbone_tokens` | `planned` | 接入 VideoSwinModel/K400 权重 |
| 8 | `mvitv2` | fixed | `in_process` | `backbone_tokens`/`pooled` | `planned` | 接入 TorchVision MViT V2 |
| 9 | `videomae` | fixed | `in_process` | `backbone_tokens` | `planned` | 接入 VideoMAEModel 公开权重 |
| 10 | `videomaev2` | fixed | `in_process` | `observed_backbone`/`pooled` | `smoke_pass` | `outputs/server-smoke/videomaev2-mlvu-cpu.json`，当前视频真权重已通过 |
| 11 | `uniformerv2` | fixed | `external_python` | `pooled` | `planned` | 冻结官方最小公开 checkpoint |
| 12 | `umt` | fixed | `external_python` | `backbone_tokens` | `planned` | 冻结 UMT-B 或最小可运行公开权重 |
| 13 | `internvideo2` | fixed | `external_python` | `backbone_tokens` | `planned` | 选择服务器可承受的最小 Stage2 视觉 checkpoint |
| 14 | `videomamba` | fixed | `external_python` | `pooled` | `planned` | 冻结 Tiny/Small checkpoint 与 CUDA 依赖 |
| 15 | `vjepa2` | fixed | `external_python` | `backbone_tokens` | `planned` | 冻结 Meta 最小公开 encoder checkpoint |
| 16 | `longvu` | long/fixed | `external_python` | `projected_visual` | `planned` | 预检最小公开变体、显存和依赖 |
| 17 | `videochat` | fixed | `external_python` | `projected_visual` | `planned` | 冻结 Ask-Anything 代码与公开权重 |
| 18 | `videochat_online` | streaming | `external_python` | `visual_memory` | `planned` | 核验官方代码/权重可得性后接 PMB |
| 19 | `videochat_flash` | long/fixed | `external_python` | `projected_visual` | `planned` | 冻结最小公开 checkpoint |
| 20 | `ma_lmm` | streaming | `external_python` | `visual_memory` | `planned` | 冻结 BLIP2/Q-Former/LLM 资产 |
| 21 | `moviechat` | streaming | `external_python` | `visual_memory` | `planned` | 核验维护仓与公开权重 |
| 22 | `streaming_vlm` | streaming | `external_python` | `decoder_contextual` | `planned` | 冻结官方 SFT checkpoint；压缩关闭 |
| 23 | `infinipot_v` | streaming | `external_python` | `decoder_contextual` | `planned` | 核验许可证与公开复现范围；压缩关闭 |
| 24 | `hermes_llava_ov` | streaming | `in_process` | `projected_visual`/`decoder_contextual` | `integrated` | 真权重两 chunk 已通过旧 static smoke；需补当前视频 `off/identity` 统一 smoke v2 |
| 25 | `mukv` | streaming | `external_python` | `decoder_contextual` | `planned` | 核验许可证，复用 0.5B 基座；压缩关闭 |

## 每项 `smoke_pass` 的五道门禁

1. registry/catalog 可发现并能解析配置；
2. adapter/worker 契约测试通过；
3. 真实公开 checkpoint 已锁定并校验；
4. 当前真实视频前向退出码为 0；
5. shape、dtype、finite、timeline 和 JSON schema 全部通过。

mock、随机权重或仅配置加载不计入 `smoke_pass`。`blocked` 必须附上可复核的上游、许可证、资产、访问或硬件证据。
