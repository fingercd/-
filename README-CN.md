# VADBench：可插拔视频编码器与 UCF-Crime 基准框架

VADBench 用同一套数据、时间轴和产物协议编排 25 条视频模型/VLM 路线，比较两类视频建模路径：

- **无状态固定 clip 编码器**：首个实现是 VideoMAE V2 Base。每个 clip 独立前向，不支持跨 clip KV cache。
- **长视频流式上下文方法**：首个实现是 HERMES + LLaVA-OneVision-Qwen2-0.5B。它缓存并压缩的是语言模型 **decoder KV**，同时导出 decoder 前的视觉 token；它不是“带 KV cache 的视觉编码器”。

首个 benchmark 是 UCF-Crime。框架覆盖官方 split 导入、32 段兼容采样、冻结特征抽取、弱监督 MIL、显式时序强监督、帧级 ROC-AUC/AP、缓存压缩注入和可追溯 JSON/JSONL 产物。原来的 `lab_anomaly/` VideoMAE V2 + MIL 代码仍保留，新的实验从 `src/vadbench/` 进入。

[English](README.md) · [编码器调研](docs/research/video-encoder-survey-2026-08-31.md) · [UCF-Crime 协议](docs/research/ucf-crime-protocol.md) · [来源审计](docs/research/native-encoder-source-audit-2026-08-31.md) · [当前进度](docs/progress/2026-08-31-current-progress.md) · [四组环境迁移](docs/progress/encoder-environment-v2.md)

## 当前实现状态

| 能力 | 状态 | 说明 |
|---|---:|---|
| 统一 `BTHWC uint8 → features[B,S,D]` | ✅ | 每个 token 保留秒/帧范围时间轴 |
| Encoder 注册与能力协商 | ✅ | 不支持的 streaming/cache/梯度请求直接失败 |
| UCF-Crime 官方清单与标注导入 | ✅ | 自动阻止 train/test 泄漏 |
| VideoMAE V2 adapter | ✅ | 稳定 pooled 输出；可选内部 hook 序列 |
| HERMES adapter | ✅ | decoder KV、position IDs、原生层次压缩与遥测 |
| 25 路候选 / 21 路运行 catalog | ✅ | 四组新环境；缺少目标权重的 4 路不注册 |
| 四组新环境 native smoke | ✅ | 14 PASS、2 许可阻塞、5 人工下载、4 未注册 |
| 特征仓和运行产物 | ✅ | 内容寻址 NPZ/NPY + 版本化 JSONL |
| 弱监督/强监督训练 | ✅ | Attention/Top-k MIL 与 temporal head |
| UCF 帧级评测 | ✅ | micro frame ROC-AUC/AP |
| 本地 mock/合成测试 | ✅ | PyTorch 与无 PyTorch 路径均覆盖 |
| VideoMAE V2 真权重冒烟 | ✅（本地与 node3 CPU） | [本地证据](docs/evidence/local-videomaev2-smoke-2026-08-31.json) · [服务器证据](docs/evidence/server-videomaev2-smoke-2026-08-31.json) |
| HERMES 真权重冒烟 | ✅（node3 CPU） | [证据](docs/evidence/server-hermes-smoke-2026-08-31.json)；四组环境结果见 outputs/environment-migration-v2/native-smoke-matrix.json |
| train/evaluate 微型闭环 | ✅ | [证据](docs/evidence/pipeline-smoke-2026-08-31.json)；合成特征，不是 benchmark 分数 |
| 真实 UCF 全量结果 | 尚未声称 | 仓库不含数据，必须使用官方视频与完整清单运行 |

HERMES 公开工作面向 VideoQA，并没有证明在 UCF-Crime 上优于固定 clip encoder。这里把它作为“真实 decoder-KV 流式路径”的首个高风险研究基线，而不是现成的 VAD SOTA。

2026-09-04 完成代码精简：生产代码净删 1,675 行，Windows/node3 全量测试通过，14 路真实权重 smoke 复核通过；删减边界、提交和验证证据见[精简计划与执行结果](docs/plans/2026-09-03-vadbench-code-simplification.md)。

## 三类缓存/压缩

| 类型 | 代表 | 框架标识 | 能否直接称视觉 encoder KV cache |
|---|---|---|---:|
| 视觉 token 压缩 | LongVU、VideoChat-Flash | `vision_tokens` | 否 |
| 外部视觉记忆 | MA-LMM、MovieChat | `visual_memory` | 否 |
| 语言模型 decoder KV | HERMES、InfiniPot-V、MuKV | `decoder_kv` | 否 |

VideoMAE V2 属于第四种：固定 clip 的无状态表征编码器。它每次 forward 内部当然会计算注意力 K/V，但没有可复用的跨调用 `past_key_values`。

## 工程结构

```text
configs/                 实验与 encoder 配置
docs/                    调研、协议和实施计划
integrations/            上游 repo/commit/license 锁
registry/                checkpoint revision、许可证与 SHA256
schemas/                 manifest、feature index、prediction JSON Schema
scripts/                 上游同步与服务器离线部署脚本
src/vadbench/
  contracts.py           encoder、时间轴、stream/cache 契约
  data/                   UCF manifest、采样、视频 I/O、特征数据集
  integrations/          21 路运行 adapter、四组环境与 worker protocol
  engine/                特征抽取、训练 runner、评测
  features.py            二进制特征仓与 JSONL 索引
  artifacts.py           provenance、预测、指标、cache telemetry
  cli.py                 `vadbench` 命令行
tests/                    单元、契约与编排测试
lab_anomaly/              旧版训练/推理代码，保留兼容
```

`data/`、`weights/`、`outputs/`、`external/` 的大文件均被 Git 忽略。仓库只提交占位、锁文件、schema 和代码。

## 快速开始

要求 Python 3.10–3.12。推荐 `uv`：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,train,video]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m vadbench doctor
```

Linux：

```bash
uv sync --extra dev --extra train --extra video
uv run pytest
uv run vadbench doctor
```

HERMES 应使用独立环境或服务器离线环境；不要把它的上游完整 pinned requirements 强塞进其他 encoder 环境。

## 冻结上游代码和权重

上游代码按精确 commit 获取：

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_upstreams.py --verify-only
```

权重下载要求显式确认各自许可证：

```bash
vadbench weights fetch videomaev2-base-hf weights/videomaev2-base-hf \
  --accept-license cc-by-nc-4.0

vadbench weights fetch hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b \
  --accept-license apache-2.0

vadbench weights verify videomaev2-base-hf weights/videomaev2-base-hf
vadbench weights verify hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b
```

VideoMAE V2 的代码仓库是 MIT，但本项目锁定的 HF 权重是 CC-BY-NC-4.0；许可证不能混为一谈。

## 构建 UCF-Crime manifest

官方 `Anomaly_Train.txt` 含完整 1,610 个训练视频。官方 `Temporal_Anomaly_Annotation.txt` 的 290 行可确定测试视频和时序标注，因此测试列表可以省略：

```bash
vadbench manifest import-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-split data/splits/Anomaly_Train.txt \
  --temporal-annotations data/splits/Temporal_Anomaly_Annotation.txt \
  --output-dir data/manifests/ucf_crime \
  --require-files \
  --probe-video-info

vadbench manifest validate data/manifests/ucf_crime/test.jsonl \
  --dataset-root data/raw/ucf_crime \
  --require-files
```

官方端点是 MATLAB 1-based inclusive；导入器会转换为 zero-based half-open `[raw_start-1, raw_end)` 并保留原始端点。例如 `165..240` 变成 `[164,240)`，覆盖 76 帧。

UCA 的时间戳自然语言事件可以用 `--uca-captions` 附加，但 `is_anomaly` 保持 `null`。没有显式审计的语义映射，不能把 UCA 所有区间当异常强监督。截止 2026-08-31，FS-UCF-Crime Zenodo 条目仍只有 placeholder。

## 25 路原生接入计划

当前研究候选保留 25 路，运行 catalog 只登记满足代码/权重条件的 21 路。四组新环境已重新验证 14 路 PASS；VideoChat-Online 与 StreamingVLM 技术前向通过但许可证阻塞，5 路等待人工资产，4 路因缺少可校验目标 checkpoint 不注册。

详细的逐路线来源、真实 checkpoint、加载入口和阻塞条件见：

- `docs/research/native-encoder-source-audit-2026-08-31.md`
- `docs/progress/encoder-integration-matrix.md`

原生 smoke 的完成条件是：目标自己的上游代码、目标自己的公开权重、`native_route_available=true`
和当前视频真实前向全部通过。任何路线无法获取原生资产时标记 `planned/blocked`，不会用另一个
模型替代。

## 抽取特征

固定 clip 路径：

```bash
vadbench extract \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --split train
```

流式 decoder-KV 路径：

```bash
vadbench extract \
  -c configs/experiments/ucf_hermes_stream.yaml \
  --split train \
  --limit-videos 2
```

HERMES encoder 配置默认启用官方 `predict_and_compress()`。外部 `identity` 只是框架侧无损对照；`keep_recent` 属于另一条简单基线，不能称 HERMES 原生策略。

## 训练检测头

冻结特征训练不会加载大 encoder：

```bash
vadbench train \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --features outputs/ucf-videomaev2-weak/features \
  --max-steps 10
```

- `task.kind: weak_mil` 只需要视频级标签。
- `task.kind: temporal_supervised` 必须具有显式 frame/segment 二值标注。
- 只有整段视频的 normal/anomaly 标签仍属于时序定位的弱监督，不会因为标签确定就自动变成强监督。

## 帧级评测

预测必须符合 `schemas/prediction-v1.schema.json`，并带帧区间或秒区间：

```bash
vadbench evaluate \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --predictions outputs/<run>/predictions/predictions.jsonl \
  --manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/<run>/evaluation/metrics.json
```

框架把所有测试视频的帧标签和分数拼接后计算 micro frame ROC-AUC，同时报告 AP。32 段是 Sultani-compatible 基线协议，不是数据集规定的唯一采样方式。

## 真实权重冒烟

```bash
vadbench smoke \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --video path/to/clip.mp4 \
  --output outputs/smoke/videomaev2.json

vadbench smoke \
  -c configs/experiments/ucf_hermes_stream.yaml \
  --video path/to/long-clip.mp4 \
  --chunks 2 \
  --output outputs/smoke/hermes.json
```

冒烟 JSON 会记录输入视频信息、输出 shape/dtype、耗时、峰值显存，以及 HERMES 原生压缩是否被调用、压缩前后 KV token 和缓存字节数。

## 运行产物

```text
outputs/<run>/
  provenance/run.json
  features/index.jsonl
  features/blobs/*.npz
  cache_telemetry/events.jsonl
  predictions/predictions.jsonl
  metrics/metrics.json
  training/checkpoints/final.pt
  training/history.json
  evaluation/metrics.json
  evaluation/frame_scores.npz
```

JSON 只存索引和元数据；大 tensor 放 NPZ/NPY。encoder fingerprint 覆盖 adapter 配置、采样、权重 revision/checksum，避免误复用不兼容特征。

## node3 离线部署

服务器目标目录是 `/users/fotile/VAD`。node2 可用时应作为集群外网出口；node2 不可用时，可以在本地冻结代码/权重/轮子并上传到 node3，共享 `/users` 上只保留一份。

当前 node3 离线环境、167 项测试、两套权重 SHA256、上游 commit 与数据软链状态见 [部署证据](docs/evidence/server-deployment-2026-08-31.json)。两类真权重正确性冒烟均已通过；GPU 吞吐/峰值显存属于下一阶段性能基准。

```bash
export VAD_PROJECT_ROOT=/users/fotile/VAD
python scripts/fetch_upstreams.py --verify-only
python -m vadbench doctor --project-root "$VAD_PROJECT_ROOT"
python scripts/server/manage_encoder_envs_v2.py verify
python scripts/server/fetch_encoder_assets_v2.py
python scripts/server/prepare_encoder_overlays_v2.py

# 数据到位后，只创建显式、不可覆盖的软链接：
bash scripts/server/link_ucf_crime.sh /users/fotile/datasets/UCF-Crime

python scripts/server/run_native_encoder_matrix_v2.py \
  --video data/smoke/mlvu-surveil-8.mp4 \
  --device cuda:5 \
  --id videomaev2 \
  --id hermes_llava_ov
```

native runner 在启动前检查目标 GPU；不会抢占已使用超过 1 GiB 的设备。

## 测试与质量门禁

```bash
python -m pytest
python -m ruff check src tests scripts/fetch_upstreams.py
python -m ruff format --check src tests scripts/fetch_upstreams.py
python -m compileall src tests
```

真实权重通过是独立门禁，不能用 fake adapter 单测替代。

## 许可证

本仓库代码使用 MIT License。第三方上游代码、数据和权重遵循各自许可证；详见 `integrations/*/upstream.lock.yaml`、`registry/checkpoints.yaml` 与调研报告。
