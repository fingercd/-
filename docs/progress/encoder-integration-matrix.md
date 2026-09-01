# 25 路视频模型/VLM 原生接入矩阵

> 更新日期：2026-08-31
> 来源审计：`docs/research/native-encoder-source-audit-2026-08-31.md`
> 当前视频：`data/smoke/mlvu-surveil-8.mp4`

## 当前汇总

| 原生状态 | 数量 |
|---|---:|
| `smoke_pass` | 14 |
| `planned` | 0 |
| `failed` | 0 |
| `blocked` | 11（其余路线均有明确资产/许可证/依赖阻塞证据） |

## 原生接入矩阵

| # | ID | 当前原生状态 | 已验证/下一动作 |
|---:|---|---|---|
| 1 | `r2plus1d_18` | `smoke_pass` | TorchVision K400 原生权重，当前视频通过 |
| 2 | `x3d` | `smoke_pass` | PyTorchVideo X3D-S 原生权重，当前视频通过 |
| 3 | `mvitv2` | `smoke_pass` | TorchVision MViTv2-S 原生权重，当前视频通过 |
| 4 | `slowfast` | `smoke_pass` | PyTorchVideo SlowFast-R50 原生权重，当前视频通过 |
| 5 | `c3d` | `blocked` | 官方 Sports1M Dropbox 权重在 node2 超时，服务器无 caffe；无替代 checkpoint |
| 6 | `i3d` | `smoke_pass` | PyTorchVideo I3D-R50 原生权重，当前视频通过 |
| 7 | `timesformer` | `smoke_pass` | `facebook/timesformer-base-finetuned-k400`，固定 8 帧；当前视频真实前向通过 |
| 8 | `video_swin` | `smoke_pass` | TorchVision Swin3D-T 原生权重，当前视频通过 |
| 9 | `videomae` | `smoke_pass` | `MCG-NJU/videomae-base`，输出 `[1,1568,768]`；当前视频真实前向通过 |
| 10 | `videomaev2` | `smoke_pass` | OpenGVLab VideoMAE V2 Base 原生权重，当前视频通过 |
| 11 | `uniformerv2` | `blocked` | 官方 K400 model-zoo 目标链接 404，未获得可校验权重 |
| 12 | `umt` | `blocked` | 官方 Issue 50 记录模型链接失效，无可校验权重 |
| 13 | `internvideo2` | `blocked` | 官方 HF Stage2 权重 gated 403，当前账号无访问权限 |
| 14 | `videomamba` | `smoke_pass` | 官方 Tiny K400 16-frame checkpoint + pinned checkout；输出 `[1,1,192]`；CPU reference selective scan |
| 15 | `vjepa2` | `smoke_pass` | Meta `facebook/vjepa2-vitl-fpc64-256`，输出 `[1,8192,1024]`；当前视频真实前向通过 |
| 16 | `longvu` | `smoke_pass` | 官方 Qwen2 7B + SigLIP SO400M + DINOv2-Giant + SVA connector；输出 `[1,144,3584]` |
| 17 | `videochat` | `blocked` | Ask-Anything VideoChat-7B HF 目标 401/不可访问，未获得 checkpoint |
| 18 | `videochat_online` | `blocked` | 原生两 chunk 已通过（`[1,304,3072]` → `[1,608,3072]`）；官方仓库无 LICENSE 文件，许可审计未过 |
| 19 | `videochat_flash` | `smoke_pass` | 官方 2B res448 snapshot；只加载视觉塔/projector，关闭 `mm_llm_compress`；输出 `[1,64,1536]` |
| 20 | `ma_lmm` | `blocked` | 官方 saved_model.tar Google Drive 端点超时，未获得 checkpoint |
| 21 | `moviechat` | `blocked` | MovieChat-vicuna HF 目标不可访问，Vicuna/LLaMA base 与许可证链未闭合 |
| 22 | `streaming_vlm` | `blocked` | 原生两 chunk decoder-KV 已通过（`[1,99,3584]`，cache `99→198`）；模型卡未声明权重 license |
| 23 | `infinipot_v` | `blocked` | 官方仓库无明确 LICENSE 与可校验 checkpoint |
| 24 | `hermes_llava_ov` | `smoke_pass` | HERMES + LLaVA-OneVision 0.5B 原生权重，两 chunk 通过 |
| 25 | `mukv` | `blocked` | 官方仓库无 LICENSE，只有基座信息无 MuKV 增量权重 |

## 原生 PASS 门禁

1. 目标自己的上游代码与 checkpoint；
2. checkpoint revision、license、SHA256 与文件大小可追溯；
3. `aux.native_route_available=true`、`aux.implementation_source=native_upstream`；
4. 当前视频真实前向，fixed 至少一 clip，streaming 至少两 chunk；
5. shape、dtype、finite、timeline、state 和 JSON schema 全部通过；
6. 未核验资产、mock、随机权重、其他模型 alias 永远不能计入。
