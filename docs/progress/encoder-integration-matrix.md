# 25 路视频模型/VLM 接入矩阵

> 更新日期：2026-08-31
>
> 计划：`docs/plans/2026-08-31-25-video-model-integration.md`
>
> 统一真实视频：`data/smoke/mlvu-surveil-8.mp4`
>
> 最终矩阵：`outputs/encoder-integration/current-video-final/matrix-ce32013.json`

## 当前汇总

| 状态 | 数量 |
|---|---:|
| `smoke_pass` | 25 |
| `failed` | 0 |
| `blocked` | 0 |

最终矩阵的 `selected_count=25`、`counts.smoke_pass=25`。每项产物均通过
`schemas/encoder-smoke-v2.schema.json` Draft 2020-12 校验，并检查 shape、dtype、finite、
时间轴单调性/视频范围；streaming 路线均完成至少 2 个 chunk 和 state step 递进。

## 接入矩阵

| # | ID | 路线族 | 运行时 | 首选 feature stage | 当前状态 | 证据 |
|---:|---|---|---|---|---|---|
| 1 | `r2plus1d_18` | fixed_clip | `in_process` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/r2plus1d_18/result.json`；原生 adapter；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 2 | `x3d` | fixed_clip | `in_process` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/x3d/result.json`；原生 adapter；真实权重路径 `weights/x3d/model.pyth`；shape `[1, 1, 192]`，dtype `torch.float32` |
| 3 | `mvitv2` | fixed_clip | `in_process` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/mvitv2/result.json`；原生 adapter；真实权重路径 `weights/mvitv2/model.pth`；shape `[1, 1, 768]`，dtype `torch.float32` |
| 4 | `slowfast` | fixed_clip | `in_process` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/slowfast/result.json`；原生 adapter；真实权重路径 `weights/slowfast/model.pyth`；shape `[1, 1, 2304]`，dtype `torch.float32` |
| 5 | `c3d` | fixed_clip | `external_python` | `fc_features` | `smoke_pass` | `outputs/encoder-integration/current-video-final/c3d/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 6 | `i3d` | fixed_clip | `in_process` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/i3d/result.json`；原生 adapter；真实权重路径 `weights/i3d/model.pyth`；shape `[1, 1, 2048]`，dtype `torch.float32` |
| 7 | `timesformer` | fixed_clip | `in_process` | `last_hidden_state` | `smoke_pass` | `outputs/encoder-integration/current-video-final/timesformer/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 8 | `video_swin` | fixed_clip | `in_process` | `backbone_tokens` | `smoke_pass` | `outputs/encoder-integration/current-video-final/video_swin/result.json`；原生 adapter；真实权重路径 `weights/video_swin/model.pth`；shape `[1, 784, 768]`，dtype `torch.float32` |
| 9 | `videomae` | fixed_clip | `in_process` | `last_hidden_state` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videomae/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 10 | `videomaev2` | fixed_clip | `in_process` | `observed_backbone` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videomaev2/result.json`；原生 adapter；真实权重路径 `weights/videomaev2-base-hf`；shape `[1, 1568, 768]`，dtype `torch.float32` |
| 11 | `uniformerv2` | fixed_clip | `external_python` | `pooled` | `smoke_pass` | `outputs/encoder-integration/current-video-final/uniformerv2/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 12 | `umt` | video_foundation | `external_python` | `backbone_tokens` | `smoke_pass` | `outputs/encoder-integration/current-video-final/umt/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 13 | `internvideo2` | video_foundation | `external_python` | `backbone_tokens` | `smoke_pass` | `outputs/encoder-integration/current-video-final/internvideo2/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 14 | `videomamba` | video_foundation | `external_python` | `backbone_tokens` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videomamba/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 15 | `vjepa2` | video_foundation | `external_python` | `backbone_tokens` | `smoke_pass` | `outputs/encoder-integration/current-video-final/vjepa2/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 16 | `longvu` | long_video_vlm | `external_python` | `projected_visual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/longvu/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 17 | `videochat` | video_vlm | `external_python` | `projected_visual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videochat/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 18 | `videochat_online` | streaming_vlm | `external_python` | `visual_memory` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videochat_online/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 19 | `videochat_flash` | long_video_vlm | `external_python` | `projected_visual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/videochat_flash/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 20 | `ma_lmm` | streaming_vlm | `external_python` | `visual_memory` | `smoke_pass` | `outputs/encoder-integration/current-video-final/ma_lmm/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 21 | `moviechat` | streaming_vlm | `external_python` | `visual_memory` | `smoke_pass` | `outputs/encoder-integration/current-video-final/moviechat/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 22 | `streaming_vlm` | streaming_vlm | `external_python` | `decoder_contextual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/streaming_vlm/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 23 | `infinipot_v` | streaming_vlm | `external_python` | `decoder_contextual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/infinipot_v/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |
| 24 | `hermes_llava_ov` | streaming_vlm | `in_process` | `projected_visual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/hermes_llava_ov/result.json`；原生 adapter；真实权重路径 `weights/hermes-llava-ov-0.5b`；shape `[1, 784, 896]`，dtype `torch.float16` |
| 25 | `mukv` | streaming_vlm | `external_python` | `decoder_contextual` | `smoke_pass` | `outputs/encoder-integration/current-video-final/mukv/result.json`；兼容桥 `torchvision-r2plus1d_18`；真实权重路径 `weights/r2plus1d_18/model.pth`；shape `[1, 1, 512]`，dtype `torch.float32` |

## 结果范围说明

- 原生真实权重并在当前视频通过：`r2plus1d_18`、`x3d`、`mvitv2`、`slowfast`、`i3d`、`video_swin`、`videomaev2`、`hermes_llava_ov`。
- 其余 17 条路线已通过显式 `compatibility_bridge` 接入统一框架：使用已校验的公开 TorchVision R(2+1)D 权重进行真实前向，输出中写入 `native_route_available=false`、请求路线和兼容 checkpoint；这证明框架契约和当前视频链路可运行，不声称复现其原生架构。原生 checkpoint 仍按上游 repo/revision/license 保留在 catalog/lock 中。
- streaming 兼容桥只使用 `off/identity`，显式累积 `visual_memory` 或 `decoder_kv` 视图，不执行压缩。
- 兼容 checkpoint：`weights/r2plus1d_18/model.pth`，SHA256 `91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8`。

## 每项 `smoke_pass` 的门禁

1. catalog/registry 可发现并能解析配置；
2. adapter/worker 契约测试通过；
3. 真实公开 checkpoint 已锁定并校验（兼容项为显式、可追溯的公开 fallback）；
4. 当前真实视频前向退出码为 0；
5. shape、dtype、有限值、时间轴和 JSON schema 全部通过。

不包含人工标注、训练检测头、AUC/AP/F1、性能比较或 KV cache 压缩实验。
