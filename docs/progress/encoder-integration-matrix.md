# 25 路视频模型/VLM v2 接入矩阵

> 更新日期：2026-09-03
> 环境实施记录：docs/progress/encoder-environment-v2.md
> 当前视频：data/smoke/mlvu-surveil-8.mp4

## 当前汇总

| v2 状态 | 数量 |
|---|---:|
| smoke_pass | 14 |
| blocked_license | 2 |
| manual_required | 5 |
| unregistered | 4 |

## 分类与验证

| # | ID | 环境组 | v2 状态 | 说明 |
|---:|---|---|---|---|
| 1 | r2plus1d_18 | classic-video-v2 | smoke_pass | CPU 与 GPU 1 真实权重 smoke 通过 |
| 2 | x3d | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 3 | mvitv2 | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 4 | slowfast | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 5 | c3d | classic-video-v2 | manual_required | 代码已在 external-v2；等待官方 checkpoint |
| 6 | i3d | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 7 | timesformer | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 8 | video_swin | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 9 | videomae | classic-video-v2 | smoke_pass | CPU 真实权重 smoke 通过 |
| 10 | videomaev2 | foundation-video-v2 | smoke_pass | Transformers 4.56.1 overlay，CPU smoke 通过 |
| 11 | uniformerv2 | foundation-video-v2 | unregistered | 官方权重链接失效，不满足注册条件 |
| 12 | umt | foundation-video-v2 | unregistered | 官方权重链接失效，不满足注册条件 |
| 13 | internvideo2 | foundation-video-v2 | manual_required | gated checkpoint 与代码 checkout 待人工准备 |
| 14 | videomamba | foundation-video-v2 | smoke_pass | 原生 VideoMamba + CPU reference selective scan |
| 15 | vjepa2 | foundation-video-v2 | smoke_pass | Meta checkpoint，CPU smoke 通过 |
| 16 | longvu | visual-vlm-v2 | smoke_pass | LongVU 视觉路径 CPU smoke 通过 |
| 17 | videochat | visual-vlm-v2 | manual_required | 代码和权重待人工准备 |
| 18 | videochat_online | visual-vlm-v2 | blocked_license | 两 chunk visual_memory 技术前向通过 |
| 19 | videochat_flash | visual-vlm-v2 | smoke_pass | 视觉塔/projector CPU smoke 通过 |
| 20 | ma_lmm | visual-vlm-v2 | manual_required | 代码已在 external-v2；等待官方权重 |
| 21 | moviechat | visual-vlm-v2 | manual_required | 代码、base model 和 checkpoint 待人工准备 |
| 22 | streaming_vlm | stream-kv-v2 | blocked_license | 两 chunk decoder KV 技术前向通过 |
| 23 | infinipot_v | stream-kv-v2 | unregistered | 无可校验目标 checkpoint |
| 24 | hermes_llava_ov | stream-kv-v2 | smoke_pass | 固定 Transformers commit overlay，两 chunk 通过 |
| 25 | mukv | stream-kv-v2 | unregistered | 无 MuKV 专用 checkpoint |

## PASS 门禁

1. 目标自己的代码和 checkpoint。
2. checkpoint revision、许可证、SHA256 与文件大小可追溯。
3. 真实视频前向，fixed 至少一个 clip，streaming 至少两个 chunk。
4. 新 Python executable 必须位于 .encoder-envs/v2。
5. 未核验资产、mock、随机权重和其他模型 alias 永远不能计入。
