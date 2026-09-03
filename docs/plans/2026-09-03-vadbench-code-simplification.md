# VADBench 代码精简与防御逻辑收敛计划

> 日期：2026-09-03  
> 分支：`qzt/refactor-vadbench-simplification`  
> 基线提交：`a40ae2b4d591f7e99dd6c85be718b518dd4fc59b`  
> 本地工作区：`D:\PythonProject\VAD`  
> 服务器工作区：`/users/fotile/VAD`

## 1. 目标

在不改变 21 路运行 catalog、14 路已通过真权重 smoke、UCF-Crime 协议、缓存语义和产物格式的前提下，把当前实现收敛为更短、更直接、更易读的代码：

- 删除内部重复校验、重复异常翻译、动态签名猜测、旧版入口和已被 v2 取代的服务器工具；
- 同一个不变量只在一个可信边界校验一次，进入内部流程后直接使用；
- 第三方差异只允许存在于模型加载、预处理、输出字段和缓存语义四处；
- 不为“看起来更抽象”而新增层次；任何重构提交的生产代码必须净减少；
- 不把 `RuntimeError` 换成另一种异常来伪造精简，只有删除分支或合并边界校验才计入成果。

本计划不是机械删除全部 `if`。跨越用户输入、数据集、磁盘、进程、第三方模型、权重身份或服务器写入边界的检查继续保留；删除这些检查会让错误静默进入科研结果，违背项目目标。

## 2. 当前基线与问题规模

### 2.1 Git 与服务器

- node3 原工作分支在 `a40ae2b`，工作树干净；GitHub 的 `feat/video-encoder-benchmark-framework` 仍停在 `0a05599`。
- `qzt/refactor-vadbench-simplification` 已从 node3 的真实 `a40ae2b` 建立，并在本地建立同名分支。
- 以下既有未跟踪文件不属于本计划，执行期间不得修改、暂存或删除：
  - `docs/plans/2026-08-31-real-data-gpu-benchmark.md`
  - `tmp_repair_foundation.sh`
- Goal 已于 2026-09-04 获得用户确认；执行结果见本文末尾。

### 2.2 代码规模

| 范围 | 文件数 | 行数 |
|---|---:|---:|
| `src/vadbench/` | 65 | 24,484 |
| `src/vadbench/integrations/` | 29 | 10,972 |
| `src/vadbench/engine/` | 7 | 2,834 |
| `src/vadbench/data/` | 9 | 3,360 |
| `scripts/server/` | 16 | 1,491 |
| `tests/` | 49 | 7,837 |

从 `0a05599` 到 `a40ae2b` 的 25 路接入阶段共改动 94 个文件，增加 22,129 行、删除 91 行。最大热点为：

| 文件 | 行数 | 主要问题 |
|---|---:|---|
| `integrations/long_video/base.py` | 1,647 | loader、反射调用、输出规范化、子进程和流状态混在一个模块 |
| `integrations/worker_protocol.py` | 1,486 | 手写序列化状态机很多，且出现平台不兼容实现 |
| `integrations/legacy.py` | 1,309 | 通用桥、Caffe 特例和输出搜索混合 |
| `integrations/pytorchvideo.py` | 1,039 | 重复 shape、NumPy、checkpoint、输出搜索逻辑 |
| `smoke.py` | 1,010 | v1/v2 两套相似执行路径 |
| `engine/integration_matrix.py` | 882 | 单函数承担筛选、预检、执行、异常、落盘和汇总 |
| `cli.py` | 870 | CLI 与业务编排边界不清 |

粗略语法统计仅用于定位热点，不作为删代码 KPI：

- `src/vadbench` 有 2,362 个 `if`、1,154 个 `raise`、263 个 `try`；
- 其中 811 个为 `if not`、`is None` 或 `not in` 形态；
- 集成层单独有 1,042 个 `if`、458 个 `raise`、148 个 `try`；
- 全仓只有 35 个显式 `raise RuntimeError`，问题主体其实是重复协议和重复兼容路径，而不只是异常类型名称。

### 2.3 验证基线

- Windows：`346 passed, 14 failed, 9 skipped`。14 个失败集中于 `worker_protocol.py`：
  - 原子替换后对只读文件句柄再次 `fsync`，Windows 报 `Bad file descriptor`；
  - NPY header 校验调用 NumPy 私有 `_read_array_header`，当前 NumPy 版本不存在该符号。
- node3/Linux：`368 passed, 1 skipped`。
- Windows `compileall` 与 Ruff 均通过。

这说明当前防御层已经出现“代码更多但可移植性更差”的反例。第一阶段必须先恢复跨平台绿色基线，再开展结构精简。

## 3. 精简规则

### 3.1 删除规则

满足任一条件即可列为删除候选：

1. 同一对象已经由构造函数或上游边界验证，内部函数再次检查同一类型、shape 或空值；
2. 同一错误被 adapter、worker、matrix 和 CLI 连续包装多次；
3. 为未登记、未发布或已弃用的调用方式保留动态签名猜测、别名或 fallback；
4. v2 已完整覆盖的旧脚本、旧 smoke writer 或旧矩阵入口；
5. 纯声明 adapter 只重复类属性，catalog 已能提供全部参数；
6. `if ...: pass`、未生效参数、重复 Git 查询、重复路径转换等无行为代码；
7. 测试只锁定“错误输入被偷偷修正后继续执行”，而新契约应当要求 canonical 输入。

### 3.2 保留规则

以下检查是系统边界，不因代码精简而删除：

| 边界 | 必须保留的行为 |
|---|---|
| UCF-Crime | 官方 split 防泄漏、MATLAB 端点转换、GT 越界、caption 不自动二值化 |
| 特征与产物 | schema、run/fingerprint、SHA256、原子写、并发 JSONL 完整性、非有限值 |
| worker | root/path/symlink/no-overwrite、dtype/shape/size/hash、结构化错误响应 |
| adapter | 本地资产、固定 revision、禁止隐式联网、真实 output stage、有限值与 timeline |
| streaming | `video_id` 隔离、step 单调、cache kind/owner/axis、压缩预算与状态更新 |
| 服务器 | protected roots、旧环境指纹、磁盘底线、GPU 归属、软链接目标和禁止覆盖 |
| 权重 | 许可证接受、repo/revision、文件大小和校验值 |

保留检查也要集中：例如 `ClipBatch` 已验证 BTHWC、dtype 和时间戳后，内部 adapter 不再重复验证输入；worker 接收端完成 sidecar 校验后，执行函数直接使用数组。

### 3.3 每个提交的硬约束

- 生产代码 `src/ + scripts/` 必须净删行；测试和文档不能掩盖生产代码膨胀。
- 新 helper 只有在同一提交至少删除两份实现并产生明显净删行时才允许加入。
- 不引入新的兼容 shim、通用框架、状态机或自定义 schema DSL。
- 不通过压行、多语句同行、难懂推导式或删除类型信息制造“少行”。
- 不把异常改名当成果；统计时同时看分支、调用路径和净行数。
- 每次只改一个职责面，相关测试通过后才进入下一阶段。

## 4. 目标结构

精简后仍保留现有顶层目录，不做无收益的搬家：

```text
contracts/config/registry
  只定义公共契约、配置与注册；内部代码信任已验证对象

integrations/common.py
  唯一的 tensor shape、NumPy 转换、feature 选择、pooling、timeline、health 实现

integrations/<family>.py
  只保留模型构造、预处理、checkpoint key 和上游输出字段

integrations/worker_protocol.py
  只保留真正跨进程的 JSON/sidecar 边界和一次序列化/反序列化

smoke.py + engine/integration_matrix.py
  一条 smoke 执行路径；matrix 只做串行调度和汇总

scripts/server/*_v2.py
  一套环境、资产、overlay、native smoke 和结果汇总入口
```

不再维护三套 `_shape/_to_numpy/_load_entrypoint/_filtered_kwargs/_find_feature`，也不再让 loader 轮流猜 `batch/frames/video/x/pixel_values`。pinned 路线使用明确参数；只有一个集中兼容函数可以按签名过滤 kwargs，且不得在捕获 `TypeError` 后换签名重试。

## 5. 分阶段执行计划

### 阶段 0：恢复跨平台绿色基线

**涉及：**

- `src/vadbench/integrations/worker_protocol.py`
- `tests/test_integration_worker.py`

**修改：**

1. 删除临时文件已 `flush + fsync` 后，对替换后只读文件再次 `fsync` 的冗余代码；保留原子 `os.replace`、no-overwrite 和权限收紧。
2. 用 NumPy 公共 header reader 替代私有 `_read_array_header`；继续在 `np.load` 前拒绝超大 shape、dtype/shape/nbytes/file-size 不一致。
3. 消除 `descriptor: Optional + assert` 这类由正常控制流已证明的内部防御，直接在成功路径返回。
4. 不弱化 path、symlink、checksum、zip member 和 object dtype 检查。

**验收：**

- Windows 当前测试集达到预期 `360 passed, 9 skipped`；
- node3 当前测试集仍为 `368 passed, 1 skipped`；
- `compileall`、Ruff、`git diff --check` 通过；
- 生产代码净减少。

**提交建议：** `fix(runtime): 精简并修复跨平台sidecar写入`

### 阶段 1：删除被 v2 取代的服务器工具

**优先删除或合并：**

- `scripts/server/prepare_encoder_assets.py`
- `scripts/server/bootstrap_offline.sh`
- `scripts/server/bootstrap_hermes_offline.sh`
- `scripts/server/run_smokes.sh`
- `scripts/server/verify_deployment.sh`
- `scripts/server/check_h3_stack.sh`
- `scripts/server/check_modules.sh`
- `scripts/server/inspect_candidate_envs.sh`

`inventory_runtime.sh` 保留为唯一只读环境巡检入口；环境、资产、overlay 和矩阵分别由现有 v2 工具承担。同步更新 README、旧计划和帮助文本，禁止保留只打印“请使用新命令”的兼容脚本。

**保留：**

- `link_ucf_crime.sh` 的路径和不覆盖门禁；
- `install_bundle.sh` 的仅 EOL 修复限制；
- v2 的 protected-root、400 GiB、license、manual asset 和 new-env 门禁。

**验收：**

- `rg` 不再命中已删除入口；
- 服务器工具测试与 `bash -n` 通过；
- `snapshot-old → bootstrap/verify → assets → overlay → native matrix` 的文档路径唯一；
- 预计净删 280–380 行。

**提交建议：** `refactor(ops): 删除旧服务器入口并统一v2工具`

### 阶段 2：统一集成层公共逻辑

**涉及：**

- `integrations/common.py`
- `integrations/legacy.py`
- `integrations/pytorchvideo.py`
- `integrations/torchvision_video.py`
- `integrations/transformers_video.py`
- `integrations/foundation/base.py`
- `integrations/long_video/base.py`
- `integrations/videomaev2.py`
- `integrations/hermes.py`

**修改：**

1. 以 `common.py` 为唯一实现，删除其余模块重复的 `_shape`、`_to_numpy`、feature/pooled 搜索、uniform timeline 和 output normalization。
2. 合并 legacy/foundation/long-video 三套 entrypoint 路径解析与签名过滤；保留 checkout 越界、非本地资产、非 callable 和 upstream 内部异常原样传播。
3. pinned adapter 明确声明输入方式和输出字段；删除自动尝试多个参数名、短/长输入自动修复、捕获 `TypeError` 后改签名重试。
4. `legacy.py` 只保留 C3D/Caffe 特有的 BTHWC 转换、prototxt/checkpoint/blob 与运行时逻辑。
5. `long_video/base.py` 只保留外部进程调用、fixed/streaming 公共状态和 cache 语义；不把拆文件本身当成果。

**验收：**

- 相同 fake 与真模型输入的 shape、dtype、pooled、timeline、aux、cache telemetry 不变；
- loader 内部抛出的 `TypeError` 不被兼容回退吞掉；
- 所有模型只允许本地已冻结资产；
- 预计净删 650–1,000 行。

**提交建议：**

1. `refactor(integration): 统一输出规范化与入口调用`
2. `refactor(encoder): 收窄legacy与长视频公共适配层`

### 阶段 3：删除纯声明 adapter 与兼容别名

**修改：**

1. 让 catalog 直接把 base adapter 与 `default_kwargs` 绑定，删除只重复 `integration_id/backend/stage/capabilities/path` 的小模块。
2. 优先处理没有特殊 loader 的 InternVideo2、UMT、UniFormerV2、VideoChat、MA-LMM、MovieChat、InfiniPot-V、MuKV 等声明型入口。
3. LongVU、VideoChat-Flash、VideoChat-Online、StreamingVLM、VideoMamba、V-JEPA2 保留真实上游特例，但移除可由 catalog 提供的重复类属性。
4. 删除仓库内无调用者的 `run_matrix`、`build_integration_matrix` 等别名；若外部 API 尚未发布，不新增弃用 shim。
5. 更新 tests 使其验证 catalog 行为，而不是强迫每个 ID 都有一个手写 Python 类。

**验收：**

- 候选仍为 25、运行 catalog 仍为 21；
- 14/2/5/4 状态分类不变；
- lazy list 不导入 Torch、Transformers 或上游 checkout；
- 预计净删 180–300 行。

**提交建议：** `refactor(encoder): 用catalog替代声明型适配器`

### 阶段 4：合并 smoke、worker 与 matrix 编排

**修改：**

1. CLI `smoke` 直接走 v2 核心路径；删除 v1 的重复 probe/encode/write 流程，只保留同一命令名和必要输出兼容。
2. `run_integration_matrix` 改为单一 item 流程：预检产生 outcome，执行产生 outcome，最后只构造一次 item。
3. 删除当前未生效的 `validate_existing` 参数或实现唯一明确语义；本计划默认删除参数和 CLI flag，不保留死接口。
4. 每次矩阵只读取一次 Git identity，删除重复 result/log 相对路径和 item dict 构造。
5. worker 只在最外层捕获一次异常并写结构化 response；matrix 不重复把同一异常包装两次。
6. worker protocol 保留 exact fields、sidecar 安全和大小限制，但用直接的 dataclass `to_dict/from_dict` 共用函数消除重复字段拆装；不引入 schema 生成器。

**验收：**

- fixed、streaming、blocked、failed、reused 五条路径均有回归；
- 单模型失败不终止矩阵，错误仍含 stage/type/message；
- smoke v2 schema 与结果路径不变；
- 预计净删 300–500 行。

**提交建议：**

1. `refactor(smoke): 合并为单一冒烟执行路径`
2. `refactor(runtime): 精简worker与矩阵状态流`

### 阶段 5：收敛 engine、core 与数据层的小重复

**修改：**

- `benchmark_plan.py` 直接调用单 case benchmark，不再构造单元素 suite；
- `engine/runner.py` 统一 train/eval epoch 汇总；`predict.py` 复用同一 task-name 解析；
- `engine/extract.py` 合并 fixed/stream 的 validate→encode→persist 公共尾部；
- `config.py` 删除仅函数内使用的 `CapabilityRequest`，用局部值表达；
- `registry.py` 合并 lazy/factory 的 spec 构造，但不增加多层 builder；
- `manifest.py` 复用已有严格原子 JSONL writer；
- `features.py` 合并 NPY resolve→hash→load；
- `models/heads.py` 复用相同 classifier 构造；
- 修正 `doctor.py` 把 `bool(stat())` 当作可写性的错误，用直接权限检查替代。

**不改：** contracts 主契约、UCF 导入/审计、metrics 单类语义、特征 fingerprint、cache kind/axis 和原子产物边界。

**验收：**

- CLI 返回码、配置、checkpoint、prediction、evaluation 和 artifact schema 不变；
- 预计净删 150–260 行。

**提交建议：** `refactor(core): 合并执行层与数据层重复流程`

### 阶段 6：合并重复测试，不降低行为覆盖

**修改：**

- 合并 smoke v1/v2 重复 writer/schema 测试；
- 把 foundation、long-video、Transformers 的 fake loader/asset fixture 集中到 `conftest.py` 或一个现有测试 helper；
- 参数化 checkpoint catalog 元数据检查；
- 删除“错误 clip 长度被 adapter 自动修正”的测试，canonical 输入错误直接失败；
- 保留 worker sidecar、UCF split、坐标、hash、原子写、离线资产、stream/cache 和 server no-overwrite 故障测试。

**验收：**

- 行为覆盖清单逐项仍有至少一个测试；
- 不以删除失败用例来让测试变绿；
- 预计测试净删 100–180 行。

**提交建议：** `test(refactor): 合并重复夹具并保留边界覆盖`

### 阶段 7：服务器真权重回归与文档收口

1. 在 node3 记录 hostname、branch、commit、dirty、Python/PyTorch/CUDA、磁盘、GPU 进程和数据软链。
2. 全量运行 `.venv/bin/python -m pytest`、Ruff、compileall、schema 校验和 `git diff --check`。
3. 四组新环境分别执行受影响路线的 native smoke；公共 output/worker 路径变更后必须重跑全部 14 条 PASS，不能只跑 mock。
4. 对 VideoChat-Online 与 StreamingVLM 复核技术前向但继续标记 license blocked；不得因重构改写许可状态。
5. 核对最终矩阵仍为：14 `smoke_pass`、2 `blocked_license`、5 `manual_required`、4 `unregistered`。
6. 比较重构前后每路的 feature shape/dtype、timeline token 数、state steps、cache kind、checkpoint SHA 和关键 aux；任何差异必须解释或回退。
7. 更新 README、架构图和本计划的最终量化结果。

**提交建议：** `docs(refactor): 记录精简结果与服务器回归证据`

## 6. 量化完成标准

全部条件同时满足才算完成：

1. `src/ + scripts/` 净删至少 1,500 行；证据支持时争取 2,000–2,400 行，不为达到数字删除边界检查。
2. 集成层不再存在多份 `_shape/_to_numpy/_load_entrypoint/_find_feature` 通用实现。
3. 生产代码提交逐个净负增长；没有新增兼容框架或状态机。
4. generic `RuntimeError` 的减少来自分支消失，不是异常改名；最终报告删除了哪些调用路径。
5. Windows 与 node3 全量测试通过；Ruff、compileall、schema、`git diff --check` 通过。
6. 21 路 catalog、25 路候选、14/2/5/4 状态和所有已发布 schema 保持一致。
7. 14 路真权重 smoke 全部重新验证，HERMES/StreamingVLM 的两 chunk 状态和 cache kind 不变。
8. 未跟踪用户文件、数据、权重、旧环境和 outputs 不被修改、删除或纳入提交。
9. 每个独立功能使用中文 Conventional Commit；Goal 获得明确授权后，每次验证通过再从服务器推送新分支，不合并、不改写历史。

## 7. 风险与停止条件

| 风险 | 处理 |
|---|---|
| 删除兼容路径影响某个 pinned 上游 | 先用当前真权重记录调用签名与输出；出现差异立即回退该小提交 |
| worker 简化削弱 sidecar 安全 | path、symlink、size、dtype、shape、hash、no-overwrite 故障测试必须先通过 |
| 旧服务器脚本仍被外部任务调用 | 删除前用仓库引用、进程命令与现有自动化做只读核查；发现真实调用则迁移调用者后再删 |
| Windows 与 Linux 行为不同 | 阶段 0 先恢复双平台基线，之后每个跨进程提交都跑双平台测试 |
| 真权重回归耗时 | 按四环境串行运行并复用已冻结资产；不下载、不覆盖、不抢 GPU |
| 仅为少行数产生晦涩代码 | 评审以调用路径和可读性为主；压行、元编程和 schema DSL 直接拒绝 |

以下情况应停止当前阶段而不是继续硬删：

- 已有真权重输出、时间轴或 cache telemetry 发生未解释变化；
- UCF split/坐标、checksum、原子写、离线或服务器保护门禁被削弱；
- 某提交生产代码净增加；
- 唯一可行方案需要新增一套抽象框架；
- node3 工作树出现非本任务改动或目标 GPU 属于其他用户。

## 8. 建议执行 Goal

计划确认后，建议创建并开始以下 Goal：

> 在 `qzt/refactor-vadbench-simplification` 上，以 `a40ae2b` 为基线，按 `docs/plans/2026-09-03-vadbench-code-simplification.md` 分阶段完成行为保持、生产代码净负增长的重构；优先修复 worker 跨平台问题，删除旧服务器入口和重复集成/编排逻辑，保留数据、权重、worker、stream/cache 与服务器安全边界；每个功能提交完成本地与 node3 相关验证后推送新分支，最终以双平台全量测试、14 路真权重 smoke、状态矩阵不变和至少净删 1,500 行为完成门禁。

## 9. 执行结果（2026-09-04）

### 9.1 代码量

代码完成提交为 `72faada`。相对基线 `a40ae2b`：

| 范围 | 新增 | 删除 | 净变化 |
|---|---:|---:|---:|
| `src/ + scripts/` | 561 | 2,236 | **-1,675** |
| `tests/` | 510 | 114 | +396 |
| `src/ + scripts/ + tests/` | 1,071 | 2,350 | **-1,279** |

生产代码超过至少净删 1,500 行的门禁；新增测试用于锁定单次调用、边界错误和真模型兼容，代码与测试合计仍净删 1,279 行。

| 指标 | 基线 | 完成后 | 变化 |
|---|---:|---:|---:|
| `src/vadbench` Python 行数 | 24,484 | 23,180 | -1,304 |
| `integrations` Python 行数 | 10,972 | 10,101 | -871 |
| `scripts/server` 行数 | 1,491 | 1,284 | -207 |
| `src/vadbench` Python 文件 | 65 | 61 | -4 |
| server 工具文件 | 16 | 10 | -6 |

粗略语法计数从 `if=2362 / defensive-if=811 / raise=1154 / try=263` 降至 `2235 / 764 / 1137 / 249`。这些数字不是质量目标；主要收益来自整段删除多签名重试、旧入口、重复序列化和声明型模块，而不是把异常换名。

### 9.2 主要结果

- Windows sidecar 从 14 个失败恢复为全绿；NPY 1/2/3 header、hash、shape、dtype、size 和 path 安全继续覆盖。
- `smoke` 只保留一条 v2 执行路径；不再隐藏关闭压缩，stream telemetry 不再丢失，失败返回非零退出码。
- foundation、legacy 和 long-video worker 均改为明确单次调用；删除 frames/batch/x/pixel_values/opaque 等参数名猜测和 `TypeError` 后重试。
- 删除 4 个不在运行 catalog 的纯声明 adapter，同时保留它们的 candidate config、upstream lock 和状态。
- 删除 32 个无仓内消费者的兼容别名；协议文档明确承诺的 `ManifestRecord` 保留。
- 删除 6 个被 v2 覆盖的服务器入口；两个仍承担离线种子重建的 bootstrap 脚本保留。
- UCF split/坐标、feature fingerprint、权重 license/revision/hash、worker sidecar、cache kind/axis、原子写和服务器 no-overwrite 门禁未删除。

### 9.3 验证

- Windows：`366 passed, 9 skipped`。
- node3：`374 passed, 1 skipped`。
- node3 定向与全量 `compileall`、`git diff --check` 通过；Ruff lint 在 Windows 通过。
- 最终真权重矩阵：`outputs/refactor/final-native-72faada/matrix-v2.json`：
  - 14 `smoke_pass`；
  - 5 `manual_asset_missing` 与 2 `license_blocked` 按预期跳过；
  - 4 `candidate_only` 不进入 21 路运行 catalog；
  - 汇总仍为 14/2/5/4。
- foundation 专项：`outputs/refactor/foundation-6e0660b/`，VideoMamba `[1,1,192]`、V-JEPA2 `[1,8192,1024]`。
- long-video 专项：`outputs/refactor/long-video-signatures/` 与 `outputs/refactor/long-video-signatures-streaming/`；LongVU、VideoChat-Flash 通过，VideoChat-Online、StreamingVLM 技术前向通过且继续保持许可阻塞。

StreamingVLM 在合法结果写完后的进程退出期出现 `libgcc_s.so.1 must be installed for pthread_cancel to work` 并返回 `-6`。用未改动的 `c00d6eb` 代码快照、同一解释器/overlay/权重也复现 `exit 134`，证明它是当前 stream-kv 环境问题，不是本次代码回归；结果中没有把该退出码隐藏。最终 14 路 PASS 矩阵不包含许可证阻塞的 StreamingVLM。
