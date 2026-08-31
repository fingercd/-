# VADBench 当前进度（2026-08-31）

## 0. 纠正声明

此前文档把 17 条 R(2+1)D 兼容桥结果写成目标模型的 `smoke_pass`，该结论错误。兼容桥只能证明统一输入输出框架可运行，不能证明对应模型已经接入。

当前可信状态是：**14 条可计入 PASS 的原生前向通过，9 条路线待接入，2 条路线因许可门禁 blocked**。旧 `outputs/encoder-integration/current-video-final/` 矩阵视为 `contract_only` 历史产物，不再进入原生统计。

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
- 当前代码基线：`38a5f43`（25 路框架）；当前纠错提交：`b2dc918`；最新原生提交：`f4fe41f`（VideoMamba）；此前兼容桥历史提交：`ce32013`
- 主运行环境：`/users/fotile/VAD/.venv`
- HERMES 隔离环境：`/users/fotile/VAD/.venv-hermes`
- PyTorch/CUDA：主 `.venv` 为 `torch 2.5.1+cu124` / CUDA 12.4；VideoMamba 原生 smoke 使用隔离的 `mllm-comp-internav`（torch 2.8.0 / CUDA 12.9），CPU reference selective-scan 路径
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

### VideoMamba

- Adapter：`videomamba`
- 官方 checkout：`external/videomamba`，commit `37355c26d0ae99ca2459f6d4044a5f509031a79f`
- 权重：`weights/videomamba/videomamba_t16_k400_f16_res224.pth`
- 权重 SHA256：`a335d728ae4dbe4f49a435022f95c6cf98108d20fe084120db1f18cb73e84f4a`，28,290,634 bytes
- 原生输出：`[1,1,192]`，`torch.float32`，CPU reference selective scan；不宣称跨 clip state/cache。

### VideoChat-Flash

- Adapter：`videochat_flash`；官方 HF snapshot：`OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448`，revision `878b4d86ab382a83b9353c33db89210aa459a735`。
- 原生 loader 只加载 `model.vision_tower`（303 keys）与 `model.mm_projector`（4 keys），关闭 `mm_llm_compress`；当前视频 4 帧 smoke 输出 `features=[1,64,1536]`、`pooled=[1,1536]`，`native_route_available=true`。

### VideoChat-Online

- Adapter：`videochat_online`；官方 checkout：`external/videochat-online`，HF 权重 revision `7373f325b9265527b9363f231b168a14523ac875`。
- 两 chunk 原生 visual-memory smoke 通过：第 1 步 `[1,304,3072]`，第 2 步 `[1,608,3072]`，`state_steps=[1,2]`；不声明 decoder KV。
- catalog 状态为 `blocked`，原因是官方仓库根目录无 LICENSE 文件；不是模型 forward 失败。

### V-JEPA 2

- Adapter：`vjepa2`；权重目录：`weights/vjepa2`；Meta 官方 `facebook/vjepa2-vitl-fpc64-256`。
- 当前视频输出：`features=[1,8192,1024]`、`pooled=[1,1024]`，`native_route_available=true`；固定 64 帧 profile，不宣称 streaming cache。

### LongVU

- Adapter：`longvu`；官方 Qwen2 7B checkpoint、SigLIP SO400M、DINOv2-Giant 与 LongVU SVA connector 均已同步。
- 当前视频 1 帧 smoke 输出 `features=[1,144,3584]`、`pooled=[1,3584]`，`native_route_available=true`；仅导出 projected_visual，不宣称 decoder KV。

### StreamingVLM

- Adapter：`streaming_vlm`；官方 Qwen2.5-VL 8B 四分片 checkpoint 已同步。
- 两 chunk 原生 decoder-KV smoke 通过：每步 `features=[1,99,3584]`，28 层 DynamicCache，cache sequence `99 → 198`，`cache_kinds=[decoder_kv]`；不执行压缩。
- catalog 状态为 `blocked`，原因是模型卡未给出明确权重 license；不是 forward 失败。

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

- VideoMamba 提交前定向测试：`55 passed, 1 warning`；Ruff check/format 与 `git diff --check` 通过。
- 最新全量测试基线仍需在本次四路原生变更后复跑；此前全量为 `360 passed, 1 skipped`。
- 原生当前状态：14 项 `smoke_pass`，9 项 `planned`，2 项 `blocked`（VideoChat-Online、StreamingVLM 许可审计）。此前 25 项矩阵属于包含兼容桥的 `contract_only` 历史产物。

### TimeSformer 真权重

已在 node3 CPU 使用 `facebook/timesformer-base-finetuned-k400` 真实权重跑通当前视频：输出 hidden state，固定 8 帧，`native_route_available=true`。

### VideoMAE 真权重

已在 node3 CPU 使用 `MCG-NJU/videomae-base` 真实权重跑通当前视频：输出 `[1,1568,768]`，`pooled=[1,768]`，`native_route_available=true`。

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

- 原生真实权重当前视频通过：`r2plus1d_18`、`x3d`、`mvitv2`、`slowfast`、`i3d`、`video_swin`、`videomaev2`、`hermes_llava_ov`、`timesformer`、`videomae`、`videomamba`、`vjepa2`、`videochat_flash`、`longvu`。
- 其余 9 条默认配置已删除兼容桥并恢复 native checkpoint 路径，状态保持 `planned`；VideoChat-Online 原生两 chunk 已运行通过但因无 LICENSE 文件进入 `blocked`；StreamingVLM 原生 decoder-KV 两 chunk 已运行通过但模型卡未提供明确权重 license，进入 `blocked`；InternVideo2 的官方 HF 权重当前返回 gated 403。
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
