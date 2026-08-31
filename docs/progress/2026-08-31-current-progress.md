# VADBench 当前进度（2026-08-31）

## 0. 纠正声明

此前文档把 17 条 R(2+1)D 兼容桥结果写成目标模型的 `smoke_pass`，该结论错误。兼容桥只能证明统一输入输出框架可运行，不能证明对应模型已经接入。

当前可信状态是：**8 条原生路线真实权重通过，17 条原生路线待接入**。旧 `outputs/encoder-integration/current-video-final/` 矩阵视为 `contract_only` 历史产物，不再进入原生统计。

纠错计划：`docs/plans/2026-08-31-native-encoder-integration-correction.md`；来源审计：`docs/research/native-encoder-source-audit-2026-08-31.md`。

纠错提交：`b2dc918 fix(status): 纠正兼容桥冒烟状态`。

## 1. 当前任务范围

根据最新约定，当前 Goal 为：

1. 把调研中的 25 条固定 clip 视频模型与长视频/VLM 路线全部登记并接入 VADBench；
2. 建立统一 catalog、lazy registry、`BTHWC uint8` 输入、`EncoderOutput/StreamStep` 输出、隔离运行时和版本化 JSON 冒烟产物；
3. 使用服务器当前视频 `data/smoke/mlvu-surveil-8.mp4` 和真实公开权重逐项跑通；
4. 验证 shape、dtype、有限值、时间轴、stream state 和产物契约；
5. 不把重心放在各路线是否严格属于纯 encoder，而以工程接入与可运行为准；
6. 不依赖人工标注，不训练检测头，不计算 AUC/AP/F1，不做性能比较，也不开展 KV cache 压缩。

纠错后的详细执行计划：`docs/plans/2026-08-31-native-encoder-integration-correction.md`。

从本文件之后：

- 项目代码只在服务器 `/users/fotile/VAD` 修改；
- Git 检查、暂存和中文规范 commit 只在服务器执行；
- 未经用户明确允许，不执行 `git push`；
- 运行产生的 JSON、日志和 checkpoint 只写入 Git 忽略的 `outputs/`、`logs/`。

## 2. 服务器状态

- 节点：`ibnode3`（node3）
- 项目目录：`/users/fotile/VAD`
- 分支：`feat/video-encoder-benchmark-framework`
- 当前代码基线：`38a5f43`（25 路框架）；当前纠错提交：`b2dc918`；此前兼容桥历史提交：`ce32013`
- 主运行环境：`/users/fotile/VAD/.venv`
- HERMES 隔离环境：`/users/fotile/VAD/.venv-hermes`
- PyTorch/CUDA：`torch 2.5.1+cu124` / CUDA 12.4
- HERMES Transformers：官方锁定的 `4.45.0.dev0`（源码 commit `66bc4def9505fa7c7fe4aa7a248c34a026bb552b`）
- GPU 占用是动态状态；每次作业启动前重新检查 GPU、进程用户与完整命令，不沿用旧空卡快照。

GPU 运行时注意：

- HERMES GPU 环境不要继承旧的 `LD_LIBRARY_PATH`；该变量曾覆盖 CUDA 动态库并触发 `libcublasLt/torchao` ABI 错误。
- 正确做法是 `unset LD_LIBRARY_PATH` 后使用 `.venv-hermes`。

## 3. 已部署模型与资产

### VideoMAE V2

- Adapter：`videomaev2`
- 权重目录：`weights/videomaev2-base-hf`
- 权重 SHA256：`ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d`
- 上游 commit：`29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`
- 类型：固定 clip、无跨 clip cache。

### HERMES + LLaVA-OneVision-Qwen2-0.5B

- Adapter：`hermes_llava_ov`
- 权重目录：`weights/hermes-llava-ov-0.5b`
- 权重 SHA256：`07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e`
- HERMES 上游 commit：`8d699b16a6bedb9086c1b39ec4253c6a1d1ce789`
- 缓存类型：语言模型 decoder KV，不是视觉 encoder KV。
- 支持原生模式：`off / static_pseudo / predict`。
- 支持输出阶段：
  - `projected_visual`：decoder 前视觉 token，只适合链路/性能观察；
  - `decoder_contextual`：受历史 KV 及压缩策略条件影响，适合后续压缩算法的语义读出。

## 4. 当前视频

主要真实监控样例：

- 路径：`data/smoke/mlvu-surveil-8.mp4`
- 来源：InfiniPot-V 仓库随附的 MLVU surveillance anomaly 样例
- SHA256：`5c7dd43429c5e556de67489920a799af8fdb614a089ab52c04b1c3b044703963`
- 容器：6,716 帧，30 FPS，320×240，约 223.87 秒

另有确定性生成的短视频：

- `data/smoke/surveillance-smoke.mp4`
- 64 帧，1 FPS，320×240

## 5. 已完成验证

### 自动化测试

- 服务器全量测试：`360 passed, 1 skipped`
- 服务器 Ruff check/format、compileall、`git diff --check` 均通过。
- 原生当前状态：8 项 `smoke_pass`，17 项 `planned`。此前 25 项矩阵属于包含兼容桥的 `contract_only` 历史产物。

### VideoMAE V2 真权重

已在 node3 CPU 上使用真实监控样例跑通：

- 输入：`mlvu-surveil-8.mp4`
- 输出 feature：`[1, 1568, 768]`
- pooled：`[1, 768]`
- dtype：`torch.float32`
- 结果文件：`outputs/server-smoke/videomaev2-mlvu-cpu.json`

### HERMES 真权重与缓存

已在 node3 CPU 上连续处理 2 个 chunk：

- 统一 smoke v2 每 chunk 输出：`[1, 784, 896]`，FP16
- 第 2 步：`cache_hit=true`
- 复用 KV token：77
- 每层 raw KV：405 / 469
- 压缩后：77（13 个 protected prompt + 64 token budget）
- 原生 `static_pseudo` 两次均 `called=true, applied=true`
- 旧压缩验证结果：`outputs/server-smoke/hermes-cpu-static.json`；统一当前视频结果：`outputs/encoder-integration/current-video-final/hermes_llava_ov/result.json`

这证明真实 decoder-KV 状态可跨 chunk 复用并可被压缩。CPU 很慢，但正确性链路成立。

### 训练/评测编排

- `train → checkpoint → evaluate` 微型闭环已跑通；
- 结果：`outputs/server-pipeline-smoke/`
- 该结果使用合成特征，只证明工程链路，不代表 UCF-Crime 精度。


## 5.1 25 路原生接入现状

- 原生真实权重当前视频通过：`r2plus1d_18`、`x3d`、`mvitv2`、`slowfast`、`i3d`、`video_swin`、`videomaev2`、`hermes_llava_ov`。
- 其余 17 条默认配置已删除兼容桥并恢复 native checkpoint 路径，状态改回 `planned`。
- 后续严格按纠错计划逐条获取官方代码/权重；资产或许可证不满足时标记 `blocked`，不再使用其他模型替代。
- 原生矩阵见 `docs/progress/encoder-integration-matrix.md`。

## 6. UCF-Crime 数据状态

Canonical 软链：

```text
/users/fotile/VAD/data/raw/ucf_crime
  -> /users/fotile/datasets/UCF-Crime
```

当前目标目录文件数：`0`。

官方协议文件已冻结并验证：

- train：1,610（800 normal + 810 anomaly）
- test：290（150 normal + 140 anomaly）
- 140 条异常测试记录共 156 个合法 frame span

由于视频本体尚未到位：

- 当前不运行正式数据审计通过门禁；
- 不生成完整 1,900 视频 feature；
- 不报告 UCF-Crime frame ROC-AUC/AP。

这不阻塞当前 25 路模型使用现有视频的接入与冒烟任务。

## 7. 当前执行顺序

1. 纠正 17 条兼容桥的 catalog/checkpoint/config 状态；
2. 冻结剩余路线的官方代码、真实 checkpoint、许可证和环境；
3. 按 fixed、foundation、visual-memory、decoder-KV 四批原生接入；
4. 每条使用当前真实视频独立生成 native smoke；
5. 最终只统计 `native_upstream` 结果。

不在当前任务中进行标注、训练检测头、AUC/AP、性能比较或 KV 压缩算法开发。

## 8. 统一接入原则

25 条路线统一遵循：

- 公共输入始终是带视频/帧/时间元数据的 `BTHWC uint8`；
- 公共输出始终能归一化为 `features[B,S,D]`、`pooled[B,D]`、`TokenTimeline` 和 `aux`；
- 需要冲突依赖的上游走独立 Python worker，不污染已有 `.venv`；
- 可关闭的 KV 压缩设为 `off/identity`，只跑基础前向；
- mock、随机权重、兼容桥或只加载配置不能记为原生 `smoke_pass`；
- 上游缺权重、许可证或硬件不满足时保存可复核 `blocked` 证据，不伪造成功。

每个功能完成后在服务器运行定向测试并提交中文 Conventional Commit；未经用户授权不 push。
