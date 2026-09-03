# Encoder 四组隔离环境 v2 实施记录

> 日期：2026-09-03
> 服务器根：/users/fotile/VAD
> 当前视频：data/smoke/mlvu-surveil-8.mp4
> 视频 SHA256：5c7dd43429c5e556de67489920a799af8fdb614a089ab52c04b1c3b044703963

## 结论

本次迁移建立了四个全新环境和模型覆盖层，四个旧环境保持只读且前后指纹一致。25 路研究候选被完整分类，其中 21 路进入运行 catalog，4 路只保留候选记录。

最终 native v2 状态：

| 状态 | 数量 |
|---|---:|
| smoke_pass | 14 |
| blocked_license | 2 |
| manual_required | 5 |
| unregistered | 4 |

VideoChat-Online 与 StreamingVLM 在新环境中的两 chunk 原生技术前向通过，但许可证仍未闭合，因此只记为 blocked_license，不进入 14 路 PASS。

## 四组环境

| 组 | 新路径 | 核心版本 | 新环境验证 |
|---|---|---|---|
| classic-video-v2 | .encoder-envs/v2/classic-video-v2 | Python 3.10.20；Torch 2.3.0+cu121；TorchVision 0.18；Transformers 4.37.2；PyTorchVideo 0.1.5 | 8 路 CPU smoke 通过；R(2+1)D 另有 GPU 1 smoke |
| foundation-video-v2 | .encoder-envs/v2/foundation-video-v2 | Python 3.10.20；Torch 2.8.0/CUDA 12.9；TorchVision 0.23；Transformers 4.57.3 | VideoMAE V2、VideoMamba、V-JEPA2 通过 |
| visual-vlm-v2 | .encoder-envs/v2/visual-vlm-v2 | Python 3.11.15；Torch 2.5.1+cu124；TorchVision 0.20.1；模型覆盖层 | LongVU、VideoChat-Flash 通过；VideoChat-Online 技术通过但许可 blocked |
| stream-kv-v2 | .encoder-envs/v2/stream-kv-v2 | Python 3.11.15；Torch 2.5.1+cu124；模型覆盖层 | HERMES 通过；StreamingVLM 技术通过但许可 blocked |

foundation clone 需要恢复原种子中的 28 个 CUDA 动态库软链，原因是 Conda clone 会把原环境的兼容覆盖恢复成需要更高 GLIBC 的包版本。visual/stream 两组补入 OpenCV 和 Transformers 间接需要的 libsndfile/FLAC/opus/vorbis/mpg123/lame；所有修复均只发生在新环境并写入环境 marker。

classic 的 pip check 仍报告 decord wheel 平台元数据警告；foundation 继承了 InternNav/Habitat 的无关依赖冲突。四组核心 Torch/CUDA/import 均通过，且所有 14 路真实模型 smoke 通过；这些 pip check 偏差保留在 outputs/environment-migration-v2/pip-check，不伪称 clean。

## 资产与代码

- 16 路现有 checkpoint 共约 54 GB，重新计算 SHA256 后全部匹配；未复制、未覆盖。
- 现有 external checkout 只读复制到 external-v2，adapter 不再修改旧 external。
- node2 在执行下载阶段无法连接跳板机，因此没有重复尝试；缺失资产按规则转入人工清单。
- 人工下载 5 路：C3D、InternVideo2、VideoChat、MA-LMM、MovieChat。
- 不注册 4 路：UniFormerV2、UMT、InfiniPot-V、MuKV。
- VideoChat、InternVideo2、MovieChat 的 external-v2 checkout 尚缺，已与 checkpoint 一同列入人工交接信息。

## 可追溯产物

- 环境矩阵：outputs/environment-migration-v2/environment-matrix.json
- 旧环境基线：outputs/environment-migration-v2/old-envs-before.json
- 资产矩阵：outputs/environment-migration-v2/asset-matrix.json
- 人工下载清单：outputs/environment-migration-v2/manual-download-manifest.json
- 覆盖层矩阵：outputs/environment-migration-v2/overlay-matrix.json
- 最终 smoke 矩阵：outputs/environment-migration-v2/native-smoke-matrix.json

根卷当前仍高于 400 GiB 剩余空间门禁。下载器使用 .cache-v2 和 weights-v2，不覆盖现有权重。
