# 25 路视频模型/VLM 统一接入与当前视频冒烟实施计划

> 日期：2026-08-31
>
> 服务器工作区：`/users/fotile/VAD`
>
> 工作分支：`feat/video-encoder-benchmark-framework`
>
> 当前验证视频：`data/smoke/mlvu-surveil-8.mp4`
>
> 视频 SHA256：`5c7dd43429c5e556de67489920a799af8fdb614a089ab52c04b1c3b044703963`

## 1. 一句话目标与成功标准

在现有 VADBench 上建立面向 25 条视频模型/VLM 路线的统一、可注册、依赖隔离的接入框架；所有路线都通过同一视频输入、同一输出契约和同一机器可读冒烟产物进行真实权重前向验证。

本计划不把“是否严格属于纯 video encoder”作为接入门槛。每个命名路线都视为一个 integration target，由 adapter 明确它导出的表征阶段。

成功标准：

1. 25 个目标均在版本化 catalog 中有唯一 ID、运行模式、adapter/backend、输入 profile、feature stage、上游代码、权重、许可证、依赖环境和状态。
2. registry 能在不导入重依赖的情况下发现全部 25 个目标，且配置均可解析。
3. 所有 adapter 消费统一 `BTHWC uint8` 与视频/帧/时间元数据，输出统一 `EncoderOutput(features[B,S,D], pooled[B,D], TokenTimeline, aux)`；流式目标输出 `StreamStep`。
4. 使用当前真实视频和真实公开 checkpoint 执行真实前向；mock、随机权重和只加载配置不能记为 `PASS`。
5. 每个目标产生统一 JSON 产物，至少记录输入身份、上游/权重身份、环境、feature stage、shape、dtype、有限值、时间轴、状态、日志和错误。
6. 不要求人工标注、训练检测头、AUC/AP/F1、模型间性能比较、缓存压缩算法或压缩消融。
7. 可关闭的 decoder-KV 压缩统一设为 `off` 或 `identity`；流式状态允许正常增长和传递，但不验压缩收益。
8. 所有项目修改、Git 检查和 commit 只在服务器完成；未经用户明确允许不执行 `git push`。

## 2. 范围与非范围

### 2.1 本计划接入的 25 个目标

| 批次 | ID | 显示名称 | 默认运行形态 | 首选表征阶段 | 首选后端/上游方向 |
|---|---|---|---|---|---|
| A | `r2plus1d_18` | R(2+1)D | fixed | `pooled` | TorchVision |
| A | `x3d` | X3D | fixed | `pooled` | PyTorchVideo |
| A | `mvitv2` | MViTv2 | fixed | `backbone_tokens`/`pooled` | TorchVision/SlowFast |
| A | `slowfast` | SlowFast | fixed | `pooled` | PyTorchVideo/PySlowFast |
| B | `c3d` | C3D | fixed | `fc_features` | 官方 legacy checkout；必要时隔离兼容实现 |
| A | `i3d` | I3D | fixed | `pooled` | PyTorchVideo I3D-R50 代表实现；记录实际 variant |
| C | `timesformer` | TimeSformer | fixed | `last_hidden_state` | Transformers/官方权重 |
| C | `video_swin` | Video Swin | fixed | `last_hidden_state` | Transformers/官方权重 |
| C | `videomae` | VideoMAE | fixed | `last_hidden_state` | Transformers/官方权重 |
| C | `videomaev2` | VideoMAE V2 | fixed | `observed_backbone`/`pooled` | 已接入的 pinned 上游 |
| C | `uniformerv2` | UniFormerV2 | fixed | `backbone_tokens`/`pooled` | 官方 checkout |
| D | `umt` | UMT | fixed | `backbone_tokens` | 官方 checkout |
| D | `internvideo2` | InternVideo2 | fixed | `backbone_tokens`/`pooled` | 官方 InternVideo checkout |
| D | `videomamba` | VideoMamba | fixed | `backbone_tokens`/`pooled` | 官方 checkout |
| D | `vjepa2` | V-JEPA 2 | fixed | `backbone_tokens`/`pooled` | Meta 官方代码/权重 |
| E | `longvu` | LongVU | fixed/long | `projected_visual` | 官方 checkout |
| E | `videochat` | VideoChat | fixed | `projected_visual` | Ask-Anything 官方 checkout |
| E | `videochat_online` | VideoChat-Online | streaming | `visual_memory` | 官方 checkout |
| E | `videochat_flash` | VideoChat-Flash | fixed/long | `projected_visual` | 官方 checkout |
| E | `ma_lmm` | MA-LMM | streaming | `visual_memory` | 官方 checkout |
| E | `moviechat` | MovieChat | streaming | `visual_memory` | 官方 checkout |
| E | `streaming_vlm` | StreamingVLM | streaming | `decoder_contextual` | 官方 checkout |
| E | `infinipot_v` | InfiniPot-V | streaming | `decoder_contextual` | 公开研究 checkout |
| E | `hermes_llava_ov` | HERMES + LLaVA-OneVision | streaming | `projected_visual`/`decoder_contextual` | 已接入的 pinned 上游 |
| E | `mukv` | MuKV | streaming | `decoder_contextual` | 官方 checkout |

上述表格是工程接入目标，不是模型学术分类结论。后续预检可以调整 backend、checkpoint 或 feature stage，但不能静默删除目标。

### 2.2 明确不纳入本 Goal

- UCF-Crime 全量 1,610/290 导入与完整性审计。
- 人工标签、UCA 映射、弱监督 MIL 或时序强监督训练。
- AUC、AP、F1、准确率或异常检测效果结论。
- encoder 间吞吐、延迟、显存、参数量或精度比较。
- KV cache 压缩、token budget 搜索、压缩消融或性能收益。
- 修改上游算法以声称新的压缩方法。
- push、合并、release 或历史改写。

## 3. 核心架构

```text
current video
  -> probe/decode
  -> SamplingProfile
  -> ClipBatch(BTHWC uint8 + video/frame/time metadata)
  -> EncoderRegistry
       -> shared backend adapter
       -> custom upstream adapter
       -> streaming adapter
  -> EncoderOutput / StreamStep
  -> SmokeValidator
  -> encoder-smoke-v2 JSON + log + environment + asset fingerprints
  -> 25-row integration matrix
```

### 3.1 保持并扩展的公共契约

- 输入继续使用 `ClipBatch`：`frames[B,T,H,W,C]`、`uint8`、`timestamps_s`、`frame_indices`、`video_ids`。
- 固定模型继续实现 `VideoEncoderAdapter.encode()`。
- 流式模型继续实现 `StreamingVideoEncoderAdapter.init_state()/encode_step()/finalize()`。
- 输出统一为 `features[B,S,D]`，必要时把只有 `[B,D]` 的模型规范为 `[B,1,D]`，同时保留 `pooled[B,D]`。
- `TokenTimeline` 必须有与 `S` 对齐的时间/帧范围；无法从上游得到逐 token 时间时，使用显式标记的近似映射，不能留空或伪称精确。
- `aux` 至少记录 `feature_stage`、`sequence_source`、`preprocess_profile`、`model_output_type`。
- 对多路输入（如 SlowFast）或 query/prompt 模型，转换逻辑只存在 adapter 内部，不改变公共输入。
- 需要 prompt 的 VLM 使用固定、类别无关的中性 prompt；prompt 进入配置和产物，不携带异常类别或真值。

### 3.2 数据驱动 catalog 与共享 backend

避免为 25 个目标复制 25 套近似加载逻辑：

- 用版本化 catalog 保存目标清单、definition、backend、profile、env、asset 和状态字段。
- 同一库/接口的模型复用 backend adapter：TorchVision、PyTorchVideo、Transformers。
- 上游接口特殊的模型使用独立 adapter，但仍复用输出规范化、有限值、时间轴和冒烟校验工具。
- registry 允许多个 ID 指向同一 lazy adapter class，并通过 `default_kwargs`/definition 选择模型变体。
- 列表和预检命令不能 import Torch、Transformers、上游 checkout 或加载权重。
- 当某条路线的原生 checkout/权重在服务器上不可得时，definition 可以显式声明
  `compatibility_bridge`。该桥只允许使用已校验的公开 checkpoint，结果的 `aux` 必须写入
  `native_route_available=false`、请求路线和实际 checkpoint；这证明统一框架和真实权重前向
  契约，不宣称复现原生架构。原生 repo/revision/license 仍保留在 catalog/lock，后续可替换。

### 3.3 两级运行时

为了避免 25 个上游的 Torch、CUDA、Transformers、timm、Caffe 等依赖互相污染，catalog 为每个目标声明一种 runtime：

- `in_process`：兼容主环境或共享 family 环境的模型直接实现 adapter。
- `external_python`：冲突上游由其隔离 Python 环境启动单模型 worker；主进程通过版本化 JSON request/response 和 NPY/NPZ sidecar 传递 `ClipBatch` 与接收 `EncoderOutput/StreamStep`。

两级运行时最终都必须经过同一 `SmokeValidator` 和 JSON schema。外部 worker 不是绕开契约的任意脚本：请求必须包含视频/帧/时间、配置与输出目录，响应必须包含 feature/timeline/aux/错误身份，sidecar 必须校验 shape、dtype、checksum 和相对路径。

### 3.4 缓存策略

- 本 Goal 不开发、调优或比较缓存压缩。
- HERMES、InfiniPot-V、MuKV 等可关闭的原生 KV 压缩统一关闭。
- 框架外部 policy 使用 `identity` 或不注入。
- 流式模型只验第二个 chunk 能消费第一步 `StreamState`、时间递进、输出契约和 state 结构一致。
- 若某个上游无法在关闭其内部机制时运行，保留官方最小默认前向，但产物必须写明该事实；不把它解释为压缩实验。

## 4. 目录与文件责任

计划中的精确文件可随源代码审计微调，但职责不变：

| 路径 | 责任 |
|---|---|
| `registry/encoder-integrations.yaml` | 25 条目标 catalog、静态资产/环境/状态字段 |
| `schemas/encoder-integration-catalog-v1.schema.json` | catalog 契约 |
| `schemas/encoder-smoke-v2.schema.json` | 单模型真实冒烟结果契约 |
| `src/vadbench/integrations/catalog.py` | 加载、校验 catalog 并惰性注册 |
| `src/vadbench/integrations/common.py` | 输出归一化、pooling、timeline、finite 检查辅助函数 |
| `src/vadbench/integrations/worker_protocol.py` | external Python 请求/响应与 sidecar 契约 |
| `src/vadbench/integrations/worker.py` | 隔离环境单模型 worker 入口 |
| `src/vadbench/integrations/torchvision_video.py` | TorchVision 模型族 adapter |
| `src/vadbench/integrations/pytorchvideo.py` | PyTorchVideo/SlowFast/X3D 模型族 adapter |
| `src/vadbench/integrations/transformers_video.py` | Transformers 视频模型族 adapter |
| `src/vadbench/integrations/legacy.py` | C3D/I3D 的隔离调用与输出桥接 |
| `src/vadbench/integrations/foundation/` | UMT、InternVideo2、VideoMamba、V-JEPA 2 特殊适配 |
| `src/vadbench/integrations/long_video/` | LongVU、VideoChat 系列、memory/streaming VLM 特殊适配 |
| `src/vadbench/smoke.py` | 统一冒烟、验证和 v2 产物 |
| `src/vadbench/engine/integration_matrix.py` | 单目标/批次/全矩阵编排与状态汇总 |
| `src/vadbench/cli.py` | `integrations list|preflight|smoke|matrix` 入口 |
| `configs/encoders/*.yaml` | 每个目标构造参数、profile、权重与 feature stage |
| `configs/smoke/encoder-matrix.yaml` | 当前视频、批次、环境、chunk 与输出目录 |
| `integrations/<id>/upstream.lock.yaml` | 上游 URL、commit、许可证、入口与依赖说明 |
| `registry/checkpoints.yaml` | checkpoint repo/revision/license/checksum/size |
| `scripts/server/prepare_encoder_assets.py` | 只做显式目标的资产预检/获取编排，不删除文件 |
| `scripts/server/run_encoder_matrix.sh` | 空闲 GPU 检查、tmux/log/exit code、单目标运行 |
| `tests/test_integration_catalog.py` | 25 项完整性、唯一性、lazy import、definition/lock 存在 |
| `tests/test_integration_common.py` | shape/pooling/timeline/finite 规范化 |
| `tests/test_encoder_family_*.py` | 各 backend 与特殊 adapter 契约测试 |
| `tests/test_integration_matrix.py` | PASS/FAILED/BLOCKED 状态和产物聚合 |
| `docs/progress/encoder-integration-matrix.md` | 25 行人工可读状态和证据索引 |

大权重、视频、环境、第三方 checkout、运行日志和 feature blobs 继续位于 Git ignore 边界内。

## 5. 状态与证据模型

每个目标只有以下运行状态：

- `planned`：catalog 已登记。
- `preflight_pass`：代码/许可证/权重/环境/资源路径已确定。
- `acquiring`：正在获取上游或权重。
- `integrated`：adapter、配置、测试已落地。
- `smoke_pass`：当前视频真实权重前向与 JSON 校验通过。
- `failed`：资产齐备但执行失败；保存 traceback、命令、环境、日志和下一动作。
- `blocked`：官方无可用权重、许可证不允许、访问受限、硬件/依赖不可满足等外部条件阻塞；必须保存可复核证据。

`blocked` 不是 `smoke_pass`，也不能用 mock 替代。Goal 默认持续推进，只有确认无法在现有授权和公开资产范围内解决时才向用户报告具体阻塞。

兼容桥的 `smoke_pass` 仅表示当前视频上的统一输入/输出/状态/产物契约通过；矩阵和进度文档必须单独列出兼容项与原生项，禁止把二者混写。

单模型 smoke JSON 至少包括：

- schema/version、run ID、status、encoder ID/display name/backend/run mode；
- Git commit/dirty、配置快照 hash；
- 上游 repo/commit/license、checkpoint repo/revision/SHA256；
- Python/Torch/CUDA/关键依赖与环境 profile；
- 视频相对路径、SHA256、帧数、FPS、时长、采样索引；
- input shape/dtype；
- feature stage、features/pooled shape/dtype、finite；
- timeline token 数、单调性、范围检查；
- streaming 的 chunks、state step、cache kind/presence；
- wall time/peak memory 仅作运行诊断，不用于横向比较；
- 日志路径、退出码、错误分类和 message。

## 6. 服务器、依赖与资产策略

1. node2 是集群唯一外网出口：获取 Git/Hugging Face 资产；node3 只运行已同步资产。
2. `/users` 共享，权重与 checkout 只保留一份；共享基座模型用显式引用避免 VideoChat/HERMES/MuKV 重复占盘。
3. 每个依赖族使用独立环境，如 `.venv-encoders/torchvision`、`.venv-encoders/transformers-video`、`.venv-encoders/<special>`；不升级现有主 `.venv` 或 `.venv-hermes` 破坏已通过链路。
4. 每次下载前记录 `df -h`、预计下载/解压/临时空间；当前服务器约有 736 GiB 可用但根卷已使用约 90%，不能无预算并发下载全部权重。
5. 权重获取后立即记录 revision、文件大小和 SHA256；checksum 不匹配即隔离为失败资产，不加载。
6. GPU 任务启动前检查实时 GPU 使用、进程用户和命令；不抢占他人 GPU。
7. 长任务使用 tmux 或项目脚本，记录节点、GPU、session、PID、日志、命令和 exit file。
8. 不读取或输出 `.env`、token、私钥或认证缓存。

## 7. 按依赖排序的执行任务

### 任务 0：冻结计划与服务器基线

**涉及文件：**

- 创建：`docs/plans/2026-08-31-25-video-model-integration.md`
- 更新：`docs/progress/encoder-integration-matrix.md`（随后创建）

**实施内容：**

- 保存本计划；记录服务器 hostname、分支、dirty 项、Python 环境、磁盘、已有视频/权重/上游和已有 smoke。
- 保留当前未跟踪的 `docs/progress/`，不覆盖或删除。
- 确认本分支不切换、不重建；不 push。

**验收与验证：**

- 计划文件位于服务器目标路径且 UTF-8 可读。
- `git status --short --branch` 只出现已知进度文档与本计划。

**提交边界：** `docs(plan): 制定25路视频模型统一接入计划`

### 任务 1：建立 25 项 catalog、schema 与 lazy registry

**涉及文件：**

- 创建：`registry/encoder-integrations.yaml`
- 创建：`schemas/encoder-integration-catalog-v1.schema.json`
- 创建：`src/vadbench/integrations/catalog.py`
- 修改：`src/vadbench/integrations/__init__.py`
- 修改：`src/vadbench/orchestration.py`
- 创建：`tests/test_integration_catalog.py`

**接口与产出：**

- `load_integration_catalog(path) -> IntegrationCatalog`
- `register_catalog_integrations(catalog, registry) -> None`
- `BUILTIN_ENCODER_CONFIGS` 由 catalog 生成或由 catalog 查询替代。

**实施内容：**

- 登记 25 个唯一目标；已有两个 ID 保持兼容。
- schema 强制每项声明 definition、backend、run mode、feature stage、environment、upstream lock、checkpoint 和 smoke profile。
- registry 列表操作维持纯轻量 import；不存在 definition/lock 的项 fail-closed。

**验收与验证：**

- `vadbench encoders list` 或新 `vadbench integrations list` 输出恰好 25 项。
- 在禁用 torch/transformers import 的测试中仍可列出 catalog。
- 所有 ID、配置和 lock 路径唯一且可重定位。
- 运行：`python -m pytest tests/test_registry.py tests/test_integration_catalog.py`。

**提交边界：** `feat(core): 建立25路模型接入目录与惰性注册`

### 任务 2：统一输出规范、冒烟校验和 v2 产物

**涉及文件：**

- 创建：`src/vadbench/integrations/common.py`
- 创建：`src/vadbench/integrations/worker_protocol.py`
- 创建：`src/vadbench/integrations/worker.py`
- 创建：`schemas/encoder-smoke-v2.schema.json`
- 修改：`src/vadbench/smoke.py`
- 创建：`src/vadbench/engine/integration_matrix.py`
- 修改：`src/vadbench/cli.py`
- 创建：`tests/test_integration_common.py`
- 创建：`tests/test_integration_worker.py`
- 创建：`tests/test_integration_matrix.py`

**实施内容：**

- 将 `[B,D]`、`[B,S,D]` 和模型输出对象规范为公共输出，并显式记录来源。
- 实现 `in_process` 与 `external_python` 两级运行时；worker 只处理一个模型/一次请求，NPY/NPZ sidecar 使用相对路径、shape/dtype/checksum 校验。
- 增加 feature/pooled 有限值、timeline 长度/单调/范围、stream step 递进检查。
- 失败运行也写状态、日志、环境和错误，不只抛出异常后丢失上下文。
- 加入 `integrations preflight/smoke/matrix` 编排；单次只运行一个目标，矩阵 runner 负责状态汇总，不把 25 个模型加载进同一进程。

**验收与验证：**

- 合成输出覆盖 `[B,D]`、`[B,S,D]`、NaN、timeline mismatch、stream step mismatch。
- smoke v2 JSON 通过 Draft 2020-12 schema。
- 旧 VideoMAE V2/HERMES smoke 命令保持可用。
- 运行：`python -m pytest tests/test_smoke.py tests/test_integration_common.py tests/test_integration_worker.py tests/test_integration_matrix.py`。

**提交边界：** `feat(smoke): 统一模型输出健康检查与冒烟产物`

### 任务 3：资产、依赖环境和运行脚本

**涉及文件：**

- 创建：`configs/smoke/encoder-matrix.yaml`
- 创建：`scripts/server/prepare_encoder_assets.py`
- 创建：`scripts/server/run_encoder_matrix.sh`
- 修改：`registry/checkpoints.yaml`
- 创建/修改：各目标 `integrations/<id>/upstream.lock.yaml`
- 创建：对应脚本测试或 dry-run 测试。

**实施内容：**

- 为每项目标登记下载来源、revision、许可证、尺寸预估、环境 profile 和共享基座引用。
- `prepare` 默认 dry-run；只有显式 ID 和显式许可证接受值才获取资产。
- runner 在 node3 检查空闲 GPU，启动 tmux/session，保存 command/log/exit file；不得自动选择已占用卡。
- 执行顺序为单目标串行，避免并发下载/加载把根卷或 GPU 占满。

**验收与验证：**

- 25 项 preflight 均产生机器可读状态。
- dry-run 不联网、不创建大文件。
- 已有 VideoMAE V2/HERMES 权重校验保持通过。
- shell 使用 `bash -n`；Python 使用 pytest、ruff、compileall。

**提交边界：** `ops(encoder): 增加模型资产预检与矩阵运行脚本`

### 任务 4：批次 A——TorchVision/PyTorchVideo

**目标：** I3D、R(2+1)D、X3D、MViTv2、SlowFast。

**涉及文件：**

- 创建：`src/vadbench/integrations/torchvision_video.py`
- 创建：`src/vadbench/integrations/pytorchvideo.py`
- 创建：5 个 encoder definition、checkpoint 条目和 upstream lock。
- 创建：`tests/test_encoder_family_torchvideo.py`

**实施内容：**

- adapter 内处理 `BTHWC -> BCTHW`、归一化、resize/crop、SlowFast 双路径。
- 捕获分类头前表征，统一 `[B,S,D]` 和 pooled。
- 先以 R(2+1)D 锁定公共接口，再扩展其他四个模型；I3D 使用 PyTorchVideo I3D-R50 代表实现，并在产物中明确实际 variant。

**验收与验证：**

- 5 个 registry/config 契约测试通过。
- 5 个模型均用当前视频真实公开权重生成 smoke v2 PASS。
- 单模型运行后释放模型进程；不做性能比较。

**提交边界：** `feat(encoder): 接入TorchVision与PyTorchVideo模型族`

### 任务 5：批次 B——C3D legacy

**目标：** C3D。

**实施内容：**

- 官方旧栈优先；若旧 Caffe 无法在当前 CUDA/Python 环境直接运行，建立隔离环境或使用能加载官方/权威公开权重的兼容桥接。
- 产物必须明确 `implementation_source`，不能把兼容实现伪称原官方 runtime。
- 输出 fc/mixed block 或 pooled 表征，不使用最终类别预测充当 VAD 指标。

**验收与验证：**

- C3D 真实权重前向成功并输出统一产物；若外部资产/旧运行时不可恢复，状态保留为 blocked 并保存完整证据，继续推进其他目标。

**提交边界：** `feat(encoder): 接入C3D经典视频模型`

### 任务 6：批次 C——视频 Transformer

**目标：** TimeSformer、Video Swin、VideoMAE、VideoMAE V2、UniFormerV2。

**实施内容：**

- Transformers 兼容目标复用共享 adapter；特殊上游用轻薄 wrapper。
- 记录每个模型 processor 的帧数、resize、mean/std、输出字段和 feature stage。
- 保持现有 VideoMAE V2 真实 smoke 回归。

**验收与验证：**

- 5 个目标使用当前视频真实权重通过统一 smoke。
- Transformers adapter 的 lazy import 和输出选择有定向测试。

**提交边界：** `feat(encoder): 接入视频Transformer模型族`

### 任务 7：批次 D——视频基础模型与状态空间模型

**目标：** UMT、InternVideo2、VideoMamba、V-JEPA 2。

**实施内容：**

- 每个上游使用 pinned checkout 与独立环境；避免把相互冲突的 timm/transformers/flash-attn 强塞进同一环境。
- 从上游稳定可观察节点导出 backbone tokens 或 pooled；记录 hook/输出字段的稳定性。
- 权重较大时先选择作者提供的最小公开变体，但 model family 与 checkpoint variant 必须同时记录。

**验收与验证：**

- 4 个目标均有真实视频、真实 checkpoint 的统一产物；不能因选择小变体而删除 family 名称。

**提交边界：** `feat(encoder): 接入视频基础模型与VideoMamba`

### 任务 8：批次 E1——视觉 token 与 VideoChat 系列

**目标：** LongVU、VideoChat、VideoChat-Online、VideoChat-Flash。

**实施内容：**

- 使用固定类别无关 prompt；prompt 与采样写入产物。
- 导出 projected visual、visual memory 或官方可稳定取得的表示。
- 不研究 token 压缩参数；若 upstream forward 内部不可避免地执行默认 token reduction，只记录行为，不做比较。

**验收与验证：**

- 4 个目标当前视频前向成功，统一输出与时间轴可被 FeatureStore 消费。

**提交边界：** `feat(encoder): 接入LongVU与VideoChat系列`

### 任务 9：批次 E2——视觉记忆与 streaming VLM

**目标：** MA-LMM、MovieChat、StreamingVLM、InfiniPot-V、HERMES、MuKV。

**实施内容：**

- memory 模型导出 visual memory；decoder 模型导出 projected/contextual 表征。
- 可关闭 KV 压缩设为 off；外部 policy 为 identity。
- 当前视频至少处理两个连续 chunk，只验 state 传递和输出健康。
- 保持现有 HERMES adapter 行为兼容，但新 smoke 配置不要求 native compression 生效。

**验收与验证：**

- 6 个目标均产生至少两个 chunk 的统一产物；第二步时间和 state index 递进。
- 许可证或权重缺失必须保存 blocked 证据，不能 vendor 未授权代码或权重。

**提交边界：** `feat(encoder): 接入视觉记忆与流式VLM模型族`

### 任务 10：最终矩阵、回归与交付审计

**涉及文件：**

- 更新：`docs/progress/encoder-integration-matrix.md`
- 更新：README 的 integration 使用方式和限制。
- 生成：`outputs/encoder-integration/<run-id>/matrix.json` 及每模型 result/log。

**实施内容：**

- 汇总 25 行状态、资产、环境、命令、产物和错误。
- 所有 `smoke_pass` 必须能从矩阵定位到真实 JSON/log/checksum。
- 对 failed/blocked 逐项给出下一动作，不用模糊“待处理”。
- 检查 Git 未跟踪大文件、外部 checkout、权重、视频和日志没有进入 index。

**验证：**

```bash
cd /users/fotile/VAD
python -m pytest
python -m compileall src tests
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
git diff --check
python -m vadbench integrations list
python -m vadbench integrations matrix \
  --config configs/smoke/encoder-matrix.yaml \
  --validate-existing
```

**验收标准：**

- catalog 恰好 25 项。
- 每项有可追溯状态；PASS 都是真实权重/真实视频。
- 没有标签、训练、AUC/AP、性能横比或缓存压缩结论。
- Git 历史按功能拆分、中文规范 commit；本地/远端均未 push。

**提交边界：** `docs(encoder): 完成25路模型接入矩阵与验证记录`

## 8. 每次功能提交的门禁

1. 在服务器执行 `git status --short`，记录并绕开用户现有改动。
2. 只暂存本功能的 adapter、config、lock、schema、test、doc。
3. 跑最相关 pytest、compileall、ruff 和 `git diff --check`。
4. 真实模型接入提交必须已有当前视频 smoke 产物；产物本身若被 ignore，只在文档记录路径/hash。
5. 使用中文 Conventional Commit。
6. 提交后确认工作树只剩其他任务/用户已有改动。
7. 不执行 `git push`。

## 9. 风险与默认处理

| 风险 | 默认处理 | 完成影响 |
|---|---|---|
| 官方无公开 checkpoint | 查官方 repo/model card/release；不使用随机权重冒充 | 标 blocked，保留证据并向用户汇报 |
| 根目录无许可证或禁止再分发 | 不 vendor、不上传权重；只登记来源或等待授权 | 可能阻塞实际接入 |
| legacy 运行时与当前 CUDA/Python 不兼容 | 独立环境/容器或权威兼容实现，记录来源 | 不污染共享环境 |
| 权重总量超出安全空间预算 | 分批下载、共享重复基座、先小变体 | 不并行堆积资产 |
| 模型需要多卡或超过 A100 40GB | 使用作者最小变体、量化仅在官方支持时使用 | 否则记录硬件 blocker |
| query/prompt 会引入任务先验 | 固定类别无关 prompt | 只做表征冒烟 |
| 同名 family 有多个 checkpoint | 选择最小公开可运行变体并记录完整 variant | family 仍保留独立 ID |
| 上游输出没有 token 序列 | 规范 pooled 为 `[B,1,D]` 并记录来源 | 仍满足统一契约 |
| 上游默认包含压缩 | 能关闭则关闭；不能关闭则只记录默认前向 | 不做压缩结论 |
| node3 无外网 | node2 获取，node3 离线运行 | 资产未同步前不反复联网 |

## 10. Goal 完成判定

Goal 不是“代码里出现 25 个名字”。完成需要：

1. 统一 catalog/registry/input/output/smoke/schema 已落地并通过回归。
2. 25 项均有 adapter/config/upstream/checkpoint/environment 状态。
3. 所有能在公开资产和现有硬件范围内运行的目标，均用当前视频真实前向并产生 `smoke_pass`。
4. 外部不可解除的 blocked 项有来源级证据；若用户不接受 blocked 作为例外，Goal 保持未完成并继续等待资产/授权。
5. 结果矩阵、运行产物、日志和进度文档一致。
6. 服务器 commit 完成；未 push。
