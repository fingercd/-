# 可插拔视频编码器与 UCF-Crime 基准框架实施计划

> 日期：2026-08-31
>
> 分支：`feat/video-encoder-benchmark-framework`
>
> 本地工作区：`D:\PythonProject\VAD`
>
> 服务器部署根：`/users/fotile/VAD`

## 一句话目标与成功标准

构建一个可注册固定 clip 与流式视频编码器的统一框架，在官方 UCF-Crime 协议上完成 manifest、特征、训练、评测和缓存遥测闭环；用真实 VideoMAE V2 Base 权重跑通无缓存 clip 冒烟，用真实 HERMES + LLaVA-OneVision-Qwen2-0.5B 权重跑通至少两个 chunk 的 decoder-KV 流式冒烟，并把代码、配置、产物和服务器证据按功能提交并推送。

目标完成必须同时有以下证据，少任一项都不能标为完成：

- 两类候选的可追溯调研、UCF-Crime 协议和项目规则文档；
- 统一 `BTHWC uint8 -> features[B,S,D]` 契约、lazy registry 和可注入 cache policy；
- 官方 split 导入与防泄漏、32 段采样、frame ROC-AUC/AP；
- 版本化 feature/prediction/run/cache-telemetry 产物；
- `vadbench manifest|extract|train|evaluate|smoke` 的可执行入口；
- Windows 本地全量单元测试通过；
- `/users/fotile/VAD` 中固定 commit 的部署记录；
- 两个真实模型、真实视频的服务器冒烟日志，不以 mock 代替；
- 至少一条微型 train/evaluate 闭环；如果全量数据仍在传输，明确标为 smoke，不伪称全量 benchmark；
- 每个独立功能有中文规范提交且已推送到 GitHub。

## 架构决策

```text
video manifest -> sampler -> encoder registry -> EncoderOutput(features + timeline)
                                      | fixed: encode(ClipBatch)
                                      | stream: init_state -> encode_step(CachePolicy)
                                                           -> StreamStep + cache telemetry
EncoderOutput -> FeatureStore -> Task/Head -> train/evaluate -> ArtifactStore
```

关键边界：

1. `VideoEncoderAdapter` 与 `StreamingVideoEncoderAdapter` 分开，固定 clip 模型不能伪装成流式模型。
2. `CacheKind` 固定为 `decoder_kv`、`vision_tokens`、`visual_memory`；HERMES 暴露的是 LLM decoder KV，分类特征则来自 decoder 前的 projected visual token。
3. 通用 cache policy 注入 adapter；HERMES 等上游原生策略以独立 adapter 配置和遥测接通，不能用 keep-recent 冒充。
4. 原始视频、权重、第三方 checkout 与大特征不进 Git；Git 只保存 manifest/schema、revision、checksum、配置和小型运行证据。
5. UCF-Crime 弱监督与时间强监督是两个明确 task；强监督来源不足时不能从文件名或 caption 自动造真值。

## 2026-08-31 实施状态快照

本计划同时记录已落地模块与后续性能工作，不能把正确性冒烟理解为完整 benchmark：

- CLI 已接通 `doctor`、`config`、`encoders`、`manifest`、`weights`、`extract`、`train`、`evaluate` 与 `smoke`；
- 核心契约、两类 adapter、data、feature/artifact、training/metrics 与 head-only runner 已落地；HERMES adapter 已区分并接通官方 `predict_and_compress` 与框架外部 policy；
- 官方标注的 1-based inclusive → 0-based half-open 转换已有 `165..240 → [164,240)` 金标测试；
- `/users/fotile/VAD` 已完成离线部署、167 项测试、两套 SHA256/upstream 校验、数据软链、两条真权重 CPU 冒烟与服务器微型 train/evaluate 闭环；证据在 `docs/evidence/`。A100 `predict` 模式吞吐/峰值显存仍是性能跟进，不阻塞框架正确性交付。

## 文件责任图

| 路径 | 责任 |
|---|---|
| `AGENTS.md` | 项目协作、Git、服务器、数据、缓存术语和验证规则 |
| `docs/research/*.md` | 编码器证据、缓存分层、UCF-Crime 协议与限制 |
| `pyproject.toml` | 最小依赖、optional extras、`vadbench` console script |
| `src/vadbench/config.py` | 实验 YAML 加载与跨字段校验 |
| `src/vadbench/checkpoints.py`、`registry/checkpoints.yaml` | 固定 revision/许可证/SHA256 的权重下载与验证 |
| `src/vadbench/contracts.py` | batch、时间轴、输出、capabilities、stream/cache 契约 |
| `src/vadbench/registry.py` | `ENCODER_REGISTRY`、lazy register/create |
| `src/vadbench/compression.py` | identity 与最小 keep-recent 对照 policy |
| `src/vadbench/data/{manifest,ucf_crime,sampling}.py` | JSONL、防泄漏、官方 split/标注、32 段/clip 采样 |
| `schemas/video-manifest-v1.schema.json` | 数据 manifest 契约 |
| `src/vadbench/integrations/videomaev2.py` | 固定 clip adapter，无缓存声明 |
| `src/vadbench/integrations/hermes.py` | 真实 HERMES chunk 路径、decoder KV state、投影视觉 token |
| `configs/encoders/{videomaev2-base,hermes-llava-ov-0.5b}.yaml` | adapter 构造参数、能力与本地上游/权重路径 |
| `integrations/{videomaev2,hermes}/upstream.lock.yaml` | 第三方 repo commit、许可证、入口和权重身份 |
| `src/vadbench/features.py`、`src/vadbench/engine/extract.py` | `.npy/.npz` blob 与 `index.jsonl`、批量抽取 |
| `src/vadbench/artifacts.py` | run、prediction、metric、cache telemetry 的原子/追加写入 |
| `schemas/{feature-index-v1,prediction-v1}.schema.json` | 特征与预测 schema |
| `src/vadbench/models/heads.py`、`src/vadbench/tasks.py` | MIL/Top-K/时序监督 head 与 task/loss |
| `src/vadbench/metrics.py`、`src/vadbench/engine/{train,evaluate}.py` | 帧投影、AUC/AP、训练一步/checkpoint、评测 |
| `src/vadbench/cli.py` | doctor/config/weights/manifest/extract/train/evaluate/smoke 编排 |
| `configs/experiments/*.yaml` | 首批两个可复现实验 |
| `tests/` | 无网络、无真实大权重的契约与编排测试 |

## 任务 1：研究、协议和治理基线

**涉及文件：**

- 创建：`AGENTS.md`
- 创建：`docs/research/video-encoder-survey-2026-08-31.md`
- 创建：`docs/research/ucf-crime-protocol.md`
- 创建：`docs/plans/2026-08-31-video-encoder-benchmark-framework.md`

**实施内容：**

- 用论文/作者仓库区分固定 clip、视觉 token、视觉记忆、decoder KV。
- 明确 VideoMAE V2 + HERMES 0.5B 首批选型及 LongVU/VideoChat 后续路线。
- 固定 UCF-Crime 1,900/128h、1,610/290、32 段和 frame ROC-AUC；记录 UCA 与 FS placeholder 限制。
- 写明当前分支、中文提交、node2 外网和 `/users/fotile/VAD` 部署约束。

**验收：**

- 文档内关键数字和架构主张旁有一手链接；不存在把 visual memory 写成 decoder KV 的表述。
- `rg -n "1900|1,900|128|1,610|290|32|ROC-AUC|placeholder|decoder_kv" AGENTS.md docs` 命中对应规则。

**提交：** `docs: 完成视频编码器调研与UCF协议`

## 任务 2：包骨架、配置、环境诊断与权重供应链

**涉及文件：**

- 创建/修改：`pyproject.toml`
- 创建：`src/vadbench/{__init__,__main__,config,doctor,checkpoints,cli}.py`
- 创建：`registry/checkpoints.yaml`
- 创建：`configs/experiments/ucf_videomaev2_weak.yaml`
- 创建：`configs/experiments/ucf_hermes_stream.yaml`
- 测试：`tests/test_{config,doctor,checkpoints,cli}.py`

**接口与行为：**

- 最小安装不导入 torch/transformers/FlashAttention；模型依赖放 optional extras。
- `vadbench config validate` 当前负责结构与配置字段校验；adapter capabilities 的动态协商在 `encoders`/任务 8 实例化门禁执行，不能把结构校验成功当作模型可运行。
- `weights fetch` 要求显式接受登记许可证；固定 HF revision 与 allowlist；`weights verify` 校验 SHA256。
- `doctor` 当前只读输出平台/Python、依赖和关键目录状态，不读取秘密文件；GPU/CUDA/磁盘由任务 10 的服务器预检命令记录，除非后续显式扩展 doctor。

**验证：**

```powershell
.venv/Scripts/python.exe -m vadbench doctor --project-root .
.venv/Scripts/python.exe -m vadbench config validate configs/experiments/ucf_videomaev2_weak.yaml
.venv/Scripts/python.exe -m vadbench config validate configs/experiments/ucf_hermes_stream.yaml
.venv/Scripts/python.exe -m vadbench weights list
.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_doctor.py tests/test_checkpoints.py tests/test_cli.py
```

**验收：** 两个配置返回规范 JSON；登记项包含 repo/revision/license/checksum；最小环境 `weights list` 不触发模型下载。

**提交：** `feat(core): 建立配置诊断与权重注册基础`

## 任务 3：统一编码器、时间轴和缓存契约

**涉及文件：**

- 创建：`src/vadbench/contracts.py`
- 创建：`src/vadbench/registry.py`
- 创建：`src/vadbench/compression.py`
- 测试：`tests/test_contracts.py`、`tests/test_registry.py`、`tests/test_compression.py`

**接口：**

- `ClipBatch`：`frames` 必须为 `uint8 BTHWC`，带 `video_ids/timestamps_s/frame_indices`。
- `EncoderOutput`：`features[B,S,D]`、`pooled`、`TokenTimeline`、`aux`。
- `EncoderCapabilities`：固定 clip、streaming、KV/token/visual-memory cache、外部 policy、训练支持。
- `StreamState/StreamStep`：状态不可跨 `video_id`，step 与 timestamp 单调。
- `CacheView/CacheUpdate/CachePolicy`：明确 kind、sequence axis、时间轴、append/replace。
- `ENCODER_REGISTRY.register_lazy/create`：只在实例化时导入重依赖，并核对声明与实际 capabilities。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_contracts.py tests/test_registry.py tests/test_compression.py
```

**验收：** 覆盖非法 dtype/shape、非单调时间、跨视频 state、错误 cache kind、lazy import 与 keep-recent 时间轴同步裁剪。

**提交：** `feat(core): 定义编码器与缓存统一契约`

## 任务 4：UCF-Crime manifest、标注与采样

**涉及文件：**

- 创建：`src/vadbench/data/{__init__,manifest,ucf_crime,sampling}.py`
- 创建：`schemas/video-manifest-v1.schema.json`
- 测试：`tests/test_manifest.py`、`tests/test_ucf_crime.py`、`tests/test_sampling.py`
- 修改：`src/vadbench/cli.py`，实现：
  - `vadbench manifest import-ucf`
  - `vadbench manifest validate`

**实施内容：**

- `import_ucf_crime` 导入官方 train/test 身份、视频级标签、官方 TXT temporal annotation，可选原样导入 UCA caption interval；TXT 的 1-based inclusive `[start,end]` 必须转换为内部 0-based half-open `[start-1,end)`。
- `load_manifest_jsonl/write_manifest_jsonl/validate_manifest/validate_manifest_pair/assert_no_split_leakage` 执行 schema、重复 ID、规范化 ID/路径跨 split、文件存在性与区间边界检查；内容 hash/视觉近重复列为完整数据到位后的独立审计。
- `uniform_segments/sample_fixed_clip/sample_uniform_segment_clips/sample_32_segments` 对 `N>=32` 构造 32 个无交叠 segment；`N<32` 明确复用/夹紧部分帧并仍返回 32 个 instance。
- UCA 字段不自动转二值标签；FS placeholder 不注册为可训练 annotation provider。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_manifest.py tests/test_ucf_crime.py tests/test_sampling.py
.venv/Scripts/python.exe -m vadbench manifest import-ucf --help
.venv/Scripts/python.exe -m vadbench manifest validate --help
```

有完整数据时额外验收：导入报告必须是 train 800 normal + 810 abnormal，test 150 normal + 140 abnormal；抽样 GT 与官方 `.mat` 一致。

**提交：** `feat(data): 接入UCF-Crime官方协议与防泄漏校验`

## 任务 5：VideoMAE V2 与 HERMES adapter

**涉及文件：**

- 创建：`src/vadbench/integrations/{__init__,videomaev2,hermes}.py`
- 创建：`configs/encoders/videomaev2-base.yaml`
- 创建：`configs/encoders/hermes-llava-ov-0.5b.yaml`
- 创建：`integrations/videomaev2/upstream.lock.yaml`
- 创建：`integrations/hermes/upstream.lock.yaml`
- 测试：`tests/test_encoder_integrations.py`

**VideoMAE V2：**

- `VideoMAEv2Adapter.encode` 复用现有 `lab_anomaly.models.vit_video_encoder.VideoMAEv2Encoder`。
- 捕获可用 backbone token；若上游只返回 pooled embedding，规范成 `[B,1,D]` 并在 `aux.sequence_source` 记录降级。
- capabilities 明确 fixed 16、no streaming、no KV/token/visual-memory cache。

**HERMES：**

- `HermesLlavaOVAdapter` 只实现 streaming；从固定 checkout 加载官方 `inference/llavaov_hermes.py` 并校验模块来源。
- `encode_step` 调用真实 `encode_video_chunk`，在同一次前向捕获 `get_video_features` 的 projected visual token，避免重复 vision compute。
- 每层 decoder KV 以 `CacheView(kind=decoder_kv)` 暴露；外部 policy 后同步位置重编号；不同视频必须 reset。
- `upstream.lock.yaml` 固定 VideoMAE V2 commit `29eab1e...` 与 HERMES commit `8d699b1...`，并记录许可证和入口。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_encoder_integrations.py
```

**验收：** fake upstream 测试证明 shape、timeline、两步 state、cache append/policy、跨视频隔离和错误 checkout；这些只算集成单测，真实权重证据在任务 11/12。

**提交：** `feat(encoder): 接入VideoMAE V2与HERMES适配器`

## 任务 6：特征、运行产物与缓存遥测

**涉及文件：**

- 创建：`src/vadbench/features.py`
- 创建：`src/vadbench/artifacts.py`
- 创建：`src/vadbench/engine/extract.py`
- 创建：`schemas/feature-index-v1.schema.json`
- 创建：`schemas/prediction-v1.schema.json`
- 测试：`tests/test_features.py`、`tests/test_artifacts.py`

**产出：**

- `FeatureStore`：`blobs/*.npy|npz` + `index.jsonl`，每条含 video/segment/timeline/shape/dtype/encoder/weight identity。
- `ArtifactStore`：`outputs/runs/<run_id>/provenance/run.json`、`metrics/metrics.json`、`predictions/predictions.jsonl`、`cache_telemetry/events.jsonl`。
- 写入必须是原子文件或带进程锁的 JSONL append；FeatureStore 的重复键遵循显式 overwrite/upsert 语义并保持 index 与 blob 一致。shape/dtype/checksum/schema mismatch 为硬错误；当前 FeatureStore 不单独拒绝 NaN/Inf，正式训练/评测入口需做非有限值门禁。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_features.py tests/test_artifacts.py
```

**验收：** round-trip 不改变特征/时间轴；并发追加不产生半行；由正式 CLI 创建的 run 必须主动填入 commit/config/data/weight identity（`ArtifactStore` 本身不强制这些字段齐全）；cache telemetry 区分 `kv/token/visual_memory`。

**提交：** `feat(artifact): 统一特征产物与缓存遥测`

## 任务 7：MIL、强监督、训练和 UCF 评测

**涉及文件：**

- 创建：`src/vadbench/models/{__init__,heads}.py`
- 创建：`src/vadbench/tasks.py`
- 创建：`src/vadbench/metrics.py`
- 创建：`src/vadbench/engine/{train,evaluate}.py`
- 测试：`tests/test_heads.py`、`tests/test_tasks.py`、`tests/test_metrics.py`

**接口：**

- `AttentionMILHead`、`TopKMILHead`、`TemporalSupervisedHead`。
- `WeaklySupervisedMILTask`、`TemporalSupervisedTask`、`build_task`。
- `build_temporal_targets` 只消费显式 frame/segment anomaly span；caption/video-only annotation 默认 ignore。
- `mil_ranking_loss`、`temporal_supervised_loss`。
- `train_one_step/save_checkpoint/load_checkpoint`。
- `project_intervals_to_grid/project_intervals_to_frames`、纯 NumPy `roc_auc_score/average_precision_score/ucf_frame_metrics`。
- `prediction_records_to_temporal/evaluate_ucf_prediction_records/evaluate_ucf_predictions` 消费 ArtifactStore prediction JSONL，输出官方 micro frame ROC-AUC 和辅助 AP。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_heads.py tests/test_tasks.py tests/test_metrics.py
```

**验收：** 合成完美排序 AUC/AP=1；ties、单类输入、边界区间有明确行为；MIL loss 可反向；checkpoint round-trip；强监督不接受未映射 UCA interval。

**提交：** `feat(train): 建立MIL与时序监督训练评测链路`

## 任务 8：CLI 端到端编排

**涉及文件：**

- 修改：`src/vadbench/cli.py`
- 修改：`tests/test_cli.py`

**已实现的最终命令：**

```text
vadbench manifest import-ucf
vadbench manifest validate
vadbench extract -c/--config
vadbench train -c/--config
vadbench evaluate -c/--config
vadbench smoke --encoder <id>
```

`smoke` 的最终常用拼写固定为 `vadbench smoke -c <config> --video <path> [--encoder <id>] [--chunks 2]`。模型与 checkout 路径来自 encoder 配置或 `external/<id>` 默认解析，不把 upstream checkout 做成常用 CLI 参数。HERMES 模式必须支持至少两 chunk，并把机器可读摘要写入 run artifact。参数最终以 `vadbench smoke --help` 为准，README 与服务器命令必须从实际 `--help` 复制，不能维护第二套拼写。

**验证：**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_cli.py
.venv/Scripts/python.exe -m vadbench manifest --help
.venv/Scripts/python.exe -m vadbench extract --help
.venv/Scripts/python.exe -m vadbench train --help
.venv/Scripts/python.exe -m vadbench evaluate --help
.venv/Scripts/python.exe -m vadbench smoke --help
```

**验收：** 缺文件/依赖/权重返回非零且错误可行动；`--dry-run` 或等价预检不加载大模型；正常完成打印 run ID 与 artifact 绝对路径。

**提交：** `feat(cli): 编排数据特征训练评测与冒烟命令`

## 任务 9：本地质量门禁

**涉及文件：** 全仓；只修正本功能引入的问题，不重构 `lab_anomaly/` 无关代码。

**验证：**

```powershell
git status --short
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m compileall src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m vadbench config validate configs/experiments/ucf_videomaev2_weak.yaml
.venv/Scripts/python.exe -m vadbench config validate configs/experiments/ucf_hermes_stream.yaml
```

**验收：** 测试全绿；无网络的最小环境能 import `vadbench`、列 registry 和跑 mock 测试；`git diff --check` 无空白错误；大文件未进入 index。

**提交：** `test: 补齐框架端到端质量门禁`

## 任务 10：node2 部署与存储预检

`node2` 是唯一外网出口。以下命令是执行模板；软链接源路径未知，必须先只读定位并替换成经 `realpath` 确认的绝对路径，**禁止原样执行占位路径**。

```bash
ssh node2
export VAD_ROOT=/users/fotile/VAD
hostname
git -C "$VAD_ROOT" status --short --branch
git -C "$VAD_ROOT" remote get-url origin
df -h "$VAD_ROOT"
du -sh "$VAD_ROOT"/weights "$VAD_ROOT"/external "$VAD_ROOT"/outputs 2>/dev/null
nvidia-smi
python --version
```

若 `/users/fotile/VAD` 尚不存在，在 `/users/fotile` 下从 GitHub clone 当前分支；若已存在，必须先确认没有未提交用户改动，再 `fetch`、`checkout feat/video-encoder-benchmark-framework`、`pull --ff-only`。不使用 reset/clean 覆盖服务器文件。

主框架环境优先依据已提交的 `uv.lock` 建立；安装后先跑 Linux 门禁：

```bash
cd "$VAD_ROOT"
uv sync --extra dev --extra video --extra train --extra videomaev2
.venv/bin/python -m pytest
.venv/bin/python -m vadbench doctor --project-root .
```

数据软链接流程：

1. 只读定位 UCF-Crime 已传到服务器的目录，检查样例可解码；
2. `realpath` 确认源不在 Git 工作树内且有足够权限；
3. 确认 `$VAD_ROOT/data/raw/ucf_crime` 不会覆盖现有真实目录；
4. 创建软链接并用 `readlink -f`、`find ... -type f | head` 验证；
5. 数据未完整时生成 `smoke.jsonl`，不要伪造完整 1,900 条 manifest。

**存储门禁：**

- UCF-Crime 原视频、两个模型、HERMES checkout、venv/pip/HF cache 和 FlashAttention 编译临时文件可能同时占空间；下载前按远端实际文件清单估算，并至少保留“下载临时副本 + 最终文件”的空间。
- HF/pip/编译 cache 指向明确的数据盘变量，例如 `VAD_CACHE_ROOT`、`VAD_TMP_ROOT`；不用系统盘或未核验的宽泛目录。
- 每个权重下载后立即运行 registry SHA256 验证；失败文件隔离，不反复堆积多个 snapshot。
- `node3` 无外网。若计算转到 node3，先确认 `/users/fotile/VAD` 是否共享文件系统；不共享时从 node2 通过集群内部传输完整环境/权重，运行时设置 offline mode。

**验证与证据：** 保存 `outputs/runs/<deployment-run>/provenance/server.json`，包含 hostname、commit、branch、Python/PyTorch/CUDA/GPU、磁盘、数据链接解析和权重校验结果。

**提交：** 不提交服务器环境本身；若新增可复用部署脚本，则提交 `ops: 增加node2可复现部署与预检脚本`。

## 任务 11：真实 VideoMAE V2 固定 clip 冒烟

**准备：**

- 在 node2 显式接受 `CC-BY-NC-4.0` 后下载 registry 的 `videomaev2-base-hf` 到 `$VAD_ROOT/weights/videomaev2-base-hf`；
- 校验 pinned revision 和 `model.safetensors` SHA256；
- 选择一个真实 UCF-Crime 视频，解码确定的 16 帧，记录 frame index/timestamp。

```bash
cd "$VAD_ROOT"
.venv/bin/python -m vadbench weights fetch videomaev2-base-hf weights/videomaev2-base-hf --accept-license cc-by-nc-4.0
.venv/bin/python -m vadbench weights verify videomaev2-base-hf weights/videomaev2-base-hf
```

**运行：** 使用 `vadbench smoke -c configs/experiments/ucf_videomaev2_weak.yaml --video <真实视频> --encoder videomaev2`；模型本地路径由 `configs/encoders/videomaev2-base.yaml`/实验配置解析。

**验收：**

- 输出 `features[1,S,D]` 和 `pooled[1,D]` 全部有限；
- `aux.sequence_source` 明确是 `observed_backbone` 或 `pooled_singleton`；后者 `S=1` 是合法 fallback，前者依赖 pinned private hook；两者的 token-to-frame timeline 都标为近似策略；
- capabilities 显示 `supports_streaming=false`、三类 cache 均 false；
- 产物写明 `cache_kind=none`，不能出现伪造 KV 命中率；
- 保存 model revision/SHA、实际 frame indices、shape/dtype/device、耗时、峰值显存和日志；
- 相同 seed/输入再次运行 shape 与 pooled 数值在精度容差内一致。

**提交：** `test(smoke): 记录VideoMAE V2真实权重冒烟`

## 任务 12：真实 HERMES 0.5B 流式 KV 冒烟

**准备：**

- node2 clone `https://github.com/haowei-freesky/HERMES` 到 `$VAD_ROOT/external/hermes`，checkout lock 中的 `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789`；
- 建立与主框架隔离但可调用的 HERMES 环境；固定官方 Transformers commit。FlashAttention 可作为 GPU 性能优化，正确性 fallback 必须仍可运行并记录实现；
- 下载并校验 pinned `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` 到 `$VAD_ROOT/weights/hermes-llava-ov-0.5b`；
- 不在 node3 尝试联网补包。

```bash
cd "$VAD_ROOT"
.venv/bin/python -m vadbench weights fetch hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b --accept-license apache-2.0
.venv/bin/python -m vadbench weights verify hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b
python3.12 -m venv .venv-hermes
. .venv-hermes/bin/activate
python -m pip install -e '.[dev,video,train,videomaev2]'
python -m pip install -r external/hermes/requirements_llava.txt
python -m pip install flash-attn --no-build-isolation
```

HERMES 冒烟在 `.venv-hermes` 内运行，既保留上游依赖隔离，又让同一进程能 import 当前 VADBench editable package。

**运行：** 使用 `vadbench smoke -c configs/experiments/ucf_hermes_stream.yaml --video <真实视频> --encoder hermes_llava_ov --chunks 2`。checkout 与本地模型路径由 `configs/encoders/hermes-llava-ov-0.5b.yaml` 或默认 `external/<id>` 解析。先跑 `identity`，再跑 HERMES/预算 policy；两次输入完全相同。

**验收：**

- 每个 chunk 输出 projected visual `features[1,S,D]`；chunk 时间范围递进，token 近似时间轴整体单调不减并允许同帧 token 时间相等；
- 第 2 步前存在第 1 步 decoder KV，且 `StreamState.step_index` 从 0 到 2；
- 每层 cache metadata 为 `owner=language_model_decoder`、`is_vision_encoder_kv=false`；
- identity 的 KV 长度随 chunk 增长；预算 policy 发生可观察的 replace/evict/aggregate，且不超过声明预算；
- 保存每层/汇总 KV tokens、bytes、update action、encode/cache-policy time、峰值显存和端到端 FPS；
- 新视频重新 `init_state` 后缓存为空或仅含初始化 prompt，不能携带前一视频状态；
- 真实运行没有使用 fake model；日志能追溯 HERMES commit 与 0.5B 权重 SHA。

**提交：** `test(smoke): 记录HERMES 0.5B真实流式缓存冒烟`

**2026-08-31 证据：** node3 CPU 上以 2 帧/chunk、2 chunks、64 visual-token 预算运行真实 0.5B 权重；第二步 `cache_hit=true`、复用 77 tokens，官方 `static_pseudo` 两次把每层 KV 压到 `13+64=77`。详见 `docs/evidence/server-hermes-smoke-2026-08-31.json`。GPU `predict` 模式只作为后续性能基准。

## 任务 13：微型训练/评测闭环

如果完整 UCF-Crime 仍在传输，选至少一段 normal 和一段 abnormal 真实训练视频，再选带官方时间 GT 的测试异常视频和正常测试视频。明确 `protocol=smoke-subset`，不得发布成完整 benchmark AUC。

执行顺序：

1. `vadbench manifest import-ucf` 或构建经相同 schema 校验的 smoke manifest；
2. `vadbench manifest validate`；
3. 对 VideoMAE V2 运行 `vadbench extract -c ...`；
4. `vadbench train -c ...` 跑最小 epoch/step并保存 checkpoint；
5. `vadbench evaluate -c ...` 产生 predictions、frame projection、metrics；
6. 对 HERMES 至少抽取同一测试视频的时序特征并验证 head/evaluator 可消费。

**验收：** FeatureStore index 可读取；checkpoint 可 round-trip；prediction 数与 timeline 对齐；metrics JSON 含 AUC/AP 与 `protocol=smoke-subset`；run provenance 没有 test GT 进入训练的证据。

完整 1,900 视频到位后，再另开正式 run，必须先通过 1,610/290 计数门禁。

**提交：** `feat(pipeline): 跑通UCF-Crime微型训练评测闭环`

## 任务 14：最终审计、README、提交与推送

**涉及文件：**

- 修改：`README-CN.md`、`README.md`（由最终实现者按实际 `--help` 更新）
- 可选创建：`docs/results/<run-id>.md`，只引用已存在的 artifact 和服务器日志

**完成审计：**

- 按本计划开头的成功标准逐项找到文件、测试、命令输出、服务器运行和 Git 远端证据；
- `git status --short` 仅包含本功能预期改动；`git diff --check` 通过；
- `git log --oneline origin/feat/video-encoder-benchmark-framework..HEAD` 为空，证明每个本地提交已推送；
- `git ls-remote --heads origin feat/video-encoder-benchmark-framework` 返回分支；
- README 不宣称未完成的全量训练、真实 AUC 或未下载的 encoder；
- 权重、视频、外部仓库、大日志均未被 Git 跟踪。

**最终提交：** `docs: 更新框架使用方法与服务器验证证据`，随后推送当前分支。

## 风险、默认假设与停止条件

| 风险/未知 | 默认处理 | 会阻塞什么 |
|---|---|---|
| UCF-Crime 仍在传输 | 使用明确的真实 smoke subset；不报告完整 benchmark | 不阻塞框架/真实模型冒烟；阻塞正式 1,900 视频 AUC |
| 数据实际服务器路径未知 | 先只读定位、realpath 和解码验证，再建软链接 | 阻塞 manifest 实体校验和服务器 smoke |
| node2 GPU/磁盘不足 | node2 只下载；确认共享盘后在有 GPU 的节点离线运行 | 可能阻塞真实 HERMES smoke，但不允许用 mock 冒充 |
| FlashAttention 构建失败 | 记录 Python/CUDA/compiler；使用官方兼容版本或预构建 wheel，不随意改上游算法 | 阻塞 HERMES 真实 smoke |
| HERMES 上游接口漂移 | 只使用 lock commit；provenance 校验模块来源 | 阻塞非 pinned `main`，不影响锁定版本 |
| HERMES cache budget 语义与外部 policy 冲突 | 先跑 identity，逐层对照 before/update/after；拒绝双重静默压缩 | 阻塞压缩结论，不阻塞基础视觉 token 抽取 |
| UCA 无二值映射 | 保留 caption + interval，label=`ignore`；另做审计映射版本 | 阻塞“强监督 UCA”正式训练 |
| FS Zenodo 只有 placeholder | 每次正式研究前复查文件清单；当前禁用 provider | 阻塞 FS 强监督，不阻塞弱监督主线 |
| 第三方许可证不明 | 不 vendor、不再分发；等待明确许可证 | 阻塞 InfiniPot-V/MuKV 主树集成，不阻塞文献登记 |

只有真实外部资产或权限成为连续多轮无法绕开的唯一阻塞时才请求用户；在此之前继续完成本地框架、单元测试、文档、node2 下载预检和可运行的其他链路。
