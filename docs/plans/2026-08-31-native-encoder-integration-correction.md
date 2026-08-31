# 原生视频模型接入纠错与剩余 17 路实施计划

> 日期：2026-08-31（Asia/Hong_Kong）
> 服务器：`ibnode3:/users/fotile/VAD`
> 分支：`feat/video-encoder-benchmark-framework`
> 现有代码提交：`86cb4ac`
> 调研审计：`docs/research/native-encoder-source-audit-2026-08-31.md`

## 0. 这份计划为什么重新建立

之前的矩阵把 17 条路线的 R(2+1)D 兼容桥前向记成了目标模型的 `smoke_pass`。那只能证明公共框架能接收视频并产生 `EncoderOutput`，不能证明对应模型已经接入；该状态必须纠正。

当前事实：

- 原生代码 + 原生公开权重真实前向已确认：`r2plus1d_18`、`x3d`、`mvitv2`、`slowfast`、`i3d`、`video_swin`、`videomaev2`、`hermes_llava_ov`（8 条）。
- `c3d`、`timesformer`、`videomae`、`uniformerv2`、`umt`、`internvideo2`、`videomamba`、`vjepa2`、`longvu`、`videochat`、`videochat_online`、`videochat_flash`、`ma_lmm`、`moviechat`、`streaming_vlm`、`infinipot_v`、`mukv`（17 条）必须重新做原生接入。
- 现有 `compatibility_bridge` 结果只能移入 `contract_only` 目录，不能进入原生矩阵、checkpoint verified 统计或 Goal 完成条件。

本计划不再设置“替代模型也算通过”的路径。原生权重、原生代码或许可证无法满足时，状态只能是 `planned` 或 `blocked`。

## 1. 目标与不可妥协的完成条件

### 1.1 目标

在 VADBench 中为剩余 17 条路线建立真正的原生 adapter：加载该路线自己的上游代码和 checkpoint，消费统一 `ClipBatch(BTHWC uint8)`，输出统一 `EncoderOutput` 或真实 `StreamStep`，并使用 `data/smoke/mlvu-surveil-8.mp4` 生成可追溯的原生 smoke 产物。

### 1.2 完成条件

对每一个目标分别满足：

1. catalog ID、definition、upstream lock、native checkpoint registry 唯一且可解析；
2. adapter 实际导入该路线的上游实现，不调用兼容桥、不调用随机/空初始化权重；
3. checkpoint 文件来自官方仓库、官方模型卡、官方 model zoo 或可核验的官方转换物，revision/URL/license/SHA256 已冻结；
4. fixed 路线完成至少一个当前视频 clip；streaming 路线在同一模型实例中连续消费两个 chunk，第二步显式接收第一步状态；
5. 输出通过 shape、dtype、finite、timeline、视频范围和 v3 JSON schema；
6. `aux.implementation_source=native_upstream`、`aux.native_route_available=true`，并记录真实 model id 和 upstream commit；
7. 如果上游只有视觉 memory 而没有 decoder KV，必须注册为 `visual_memory`；不能凭 `use_cache=True` 或 attention 模块把它写成 `decoder_kv`；
8. 任何资产缺失、链接失效、许可证未确认、硬件不可满足都不得被替换成另一个模型来通过。

不包括：UCF-Crime 全量训练/标注/AUC/AP、模型横向性能比较、KV 压缩算法设计和压缩收益结论。

## 2. 统一架构调整

### 2.1 原生与 contract-only 分离

**涉及文件：**

- 修改：`registry/encoder-integrations.yaml`
- 修改：`registry/checkpoints.yaml`
- 修改：`schemas/encoder-integration-catalog-v1.schema.json`
- 新增：`registry/contract-smoke.yaml`（可选的兼容桥登记）
- 修改：`src/vadbench/integrations/catalog.py`
- 修改：`src/vadbench/engine/integration_matrix.py`
- 修改：`docs/progress/encoder-integration-matrix.md`

**实施：**

- 将 17 条目标从 `smoke_pass` 恢复为 `planned`；其 native checkpoint 状态恢复为 `planned` 或 `blocked`。
- 增加机器可验证的 `validation_scope: native | contract_only`。默认 `integrations matrix` 只选择 `native`；兼容桥只能通过显式 `contract-smoke` 命令运行。
- 将已有 R(2+1)D 兼容产物标记为 `contract_only` 并在文档中声明“不是目标模型结果”。不删除历史产物，避免审计线索丢失。
- native smoke v3 必须拒绝 `native_route_available=false` 的结果，即使输出 shape 健康也不能计数为 PASS。

**验收：**

- `vadbench integrations list` 显示 8 条原生已通过、17 条 native pending/blocked；不再显示 25 条原生 PASS。
- `vadbench integrations matrix` 不会自动加载兼容桥。
- 对兼容桥显式执行 `vadbench integrations contract-smoke --id ...` 才能产生 contract-only 产物。

### 2.2 资产获取和校验器

**涉及文件：**

- 修改：`registry/checkpoints.yaml`
- 新增：`scripts/server/prepare_native_assets.py`
- 新增：`scripts/server/verify_native_asset.py`
- 修改：`scripts/server/run_encoder_matrix.sh`
- 测试：`tests/test_native_asset_policy.py`

**实施：**

- node2 负责下载 Git/Hugging Face/OSS；node3 只执行已同步资产。每次下载前记录 `df -h`、预估体积和目标目录。
- Hugging Face 使用固定 `revision`、`allow_patterns`、`local_files_only` 复验；直链使用 `curl --fail --location`，下载后立即 SHA256 校验。
- 每个 native checkpoint 条目必须同时保存：`repo_id/repo_url/revision/variant/license/local_path/files/sha256/size_bytes`。不能把另一模型的文件写在该条目的 `local_path` 下。
- 许可证未确认的路线先保留 `blocked`；不能把“GitHub 可见”当作“允许复制/再分发”。
- 脚本只预检和下载明确目标，不自动 clone 全部仓库、不删除旧资产、不覆盖校验通过的文件。

**验收：**

- 资产预检输出每项 `native_model_id`、实际路径、大小、SHA256、许可证和状态。
- 一字节改动或路径指向其他模型时验证失败。

## 3. 17 条路线的真实接入方案

下表中的“首选”是实际实现路径；如果下载/许可证/运行时预检失败，保留该目标为 `blocked`，不更换模型。

| ID | 官方代码与原生权重 | 原生 adapter 做法 | 运行时/资源门禁 |
|---|---|---|---|
| `c3d` | [facebookarchive/C3D](https://github.com/facebookarchive/C3D)；优先核验 [MMAction C3D Sports-1M 转换 checkpoint](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) 或 Dartmouth 原始 Sports-1M 模型 | `LegacyVideoAdapter` 改为加载真实 C3D Caffe/转换 state；固定 16×112，导出 `fc6/fc7`，记录转换来源和层名 | 旧 Caffe/PyTorch 隔离环境；CC-BY-NC，转换物必须单独登记 |
| `timesformer` | [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer)；[facebook/timesformer-base-finetuned-k400](https://huggingface.co/facebook/timesformer-base-finetuned-k400)，8×224、约 486 MB | 固定 8 帧；使用官方/Transformers `TimesformerModel`，从 hidden state 或 classifier 前 hook 取特征，禁止只返回 logits | Python 3.10/3.11 + pinned Transformers/timm/einops；CC-BY-NC-4.0 |
| `videomae` | [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)；[MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | 移除 definition 中兼容桥，使用 `VideoMAEModel/VideoMAEForPreTraining` 的真实 encoder hidden state；16×224；不把 reconstruction decoder 当语言 decoder | Transformers 环境；CC-BY-NC-4.0；需下载 config、processor、safetensors |
| `uniformerv2` | [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)；官方 [MODEL_ZOO](https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md) 的 K400 B/16 8×3×4 OSS checkpoint | 固定 8 帧；加载官方 `UniFormerV2`，从 backbone/projector 前层取 `[B,S,D]`；严格记录实际 run/test config | OpenGVLab 专用环境；OSS 链接与许可证逐项冻结；不能用 HF gated 的不相关文件替代 |
| `umt` | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) 与其 `single_modality/MODEL_ZOO.md` | 使用官方 UMT video classification/backbone builder，加载真实 K400/SSv2 checkpoint，hook classifier 前特征 | 官方 Issue [#50](https://github.com/OpenGVLab/unmasked_teacher/issues/50) 报告模型链接失效；链接未恢复前保持 blocked |
| `internvideo2` | [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo) 的 `InternVideo2`；[OpenGVLab/InternVideo2-Stage2_1B-224p-f4](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) | 只构造 vision encoder，不能调用 LLM logits；导出 Stage2 visual tokens/pooled；16×224 | 1B 专用环境，优先单卡 CPU/GPU 小 batch；核验 vision state keys 和 model revision |
| `videomamba` | [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba)；[OpenGVLab/VideoMamba](https://huggingface.co/OpenGVLab/VideoMamba) 的 `videomamba_t16_k400_f16_res224.pth` | 使用官方 `videomamba_tiny(num_classes=400,num_frames=16)`，加载 K400 权重并导出 classifier 前状态；固定路线，不宣称跨 clip state | dedicated CUDA/selective-scan 环境；先跑 tiny；HF 模型页标 Apache-2.0 |
| `vjepa2` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)；Meta 直链 `https://dl.fbaipublicfiles.com/vjepa2/vitl-256.pt` 或 [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 使用官方 video encoder，不使用 predictor/action head；记录官方 preprocessor；固定 16/64 帧 profile | decord/timm/einops；ViT-L 约 300M，显存预检；Meta README 要求显式下载 checkpoint |
| `longvu` | [Vision-CAIR/LongVU](https://github.com/Vision-CAIR/LongVU)；[Vision-CAIR/LongVU_Qwen2_7B](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B)，若官方较小视频变体预检可用则优先较小者 | 调用 `load_pretrained_model`、官方 video processor 和 vision tower/projector；导出 projected visual，不生成回答文本；长视频 adapter 只负责真实 visual path | 官方 README 本地 demo 最低约 40 GB GPU；A100 40GB 需 batch=1/少帧预检，OOM 即 blocked |
| `videochat` | [OpenGVLab/Ask-Anything](https://github.com/OpenGVLab/Ask-Anything)；以官方 README 的 VideoChat/VideoChat2 checkpoint 链接为唯一来源 | 选择实际可下载的 VideoChat 版本后，加载其 InternVideo/UMT vision tower 和 projector；导出 projector 前后 feature，禁止用 Space 或随机初始化代替 | 旧 Vicuna/InternVideo 依赖隔离；若官方只提供 demo 无可下载 checkpoint，标 blocked |
| `videochat_online` | [MCG-NJU/VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online)；[MCG-NJU/VideoChatOnline-4B](https://huggingface.co/MCG-NJU/VideoChatOnline-4B)，8.29 GB、HF license unknown；README 仅有 MIT badge，仓库根目录无 LICENSE 文件 | 复现官方 ViT feature + Pyramid Memory Bank；实现真实 `visual_memory` state；当前公开代码明确未提供 KV-cache memory，不能注册为 decoder KV | Python 3.9/Transformers custom code；单卡 BF16/CPU 预检；必须连续两 chunk |
| `videochat_flash` | [OpenGVLab/VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash)；[VideoChat-Flash-Qwen2_5-2B_res448](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448)，2B、Apache-2.0 | `AutoModel.from_pretrained(..., trust_remote_code=True)`；调用官方 `get_vision_tower()`/projector，默认关闭 `mm_llm_compress`，导出 projected visual | Transformers 4.40.1/timm/av/decord；可选 flash-attn；2B A100 预检 |
| `ma_lmm` | [boheumd/MA-LMM](https://github.com/boheumd/MA-LMM)；官方 `saved_model.tar`，以及 InstructBLIP/LAVIS/Vicuna base | 复现官方 memory bank update/adjacent merge；输出 `visual_memory`，保留 Q-Former 输入与 memory 长度；不能称 decoder KV | LAVIS/fairscale/decord 隔离；Google Drive 权重、Vicuna 和多份许可证全部核验 |
| `moviechat` | [wenhaochai/MovieChat](https://github.com/wenhaochai/MovieChat) 的固定 revision/Onevision 分支；README 指定的 MovieChat checkpoint 和 base model | 复现 short/long memory 更新与合并；从真实 EVA/视觉编码器和 projector 取 memory；不以兼容桥伪造 state | BSD/LAVIS/MiniGPT-4/VideoLLaMA 等许可证分开登记；旧 Vicuna 权重需本地路径 |
| `streaming_vlm` | [mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)；[mit-han-lab/StreamingVLM](https://huggingface.co/mit-han-lab/StreamingVLM)，8B BF16、4 shard | 使用官方 `streaming_vlm/inference/inference.py` 的 compact KV 机制；同一模型实例返回真实 decoder KV/state；不生成额外压缩算法 | `env_infer.sh` 对应独立环境；约 16.6 GB 权重，A100 40GB batch=1；native cache kind=decoder_kv |
| `infinipot_v` | [aiha-lab/InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V)；官方支持 Qwen2-VL/Qwen2.5-VL base + repo 代码 | 调用官方 `qwen_inference_ovu.py`/`kvcache_utils.py`；分别记录 base visual tokens、native KV cache 和 compression off 对照；不把伪实现当 native | 官方仓库自称 research re-implementation 且不含论文全部组件；根目录未见明确 LICENSE，许可证确认前 blocked |
| `mukv` | [IMBALDY/MuKV](https://github.com/IMBALDY/MuKV)；README 指定 [LLaVA-OneVision 0.5B](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) base | 加载官方 MuKV `model/` 与 `kvcache_utils.py`，真实接通 multi-grained KV state；默认 smoke 关闭压缩但保留 native cache plumbing；不再使用 R(2+1)D | `prepare.sh` 独立环境；当前仓库未显示清晰 LICENSE，先做授权门禁 |

## 4. adapter 与输出实现细则

### 4.1 Fixed 路线

- `ClipBatch` 的采样由 VADBench 统一完成，adapter 内只做该模型官方要求的 resize/normalize/layout/temporal sampling。
- 若上游 forward 默认只返回分类 logits，使用官方允许的 `return_features`、`forward_features`、模块 hook 或替换 classifier 为 identity；记录 `feature_source` 和层名。
- 不把 logits 当作 `[B,S,D]` 视觉 token；若只能得到 `[B,D]`，规范成 `[B,1,D]` 并记录 `sequence_source=pooled_singleton`。
- fixed adapter 的 `supports_streaming=false`、所有 cache 能力为 false；Transformer 内部 K/V 不计作跨 clip cache。

### 4.2 Visual-memory 路线

- VideoChat-Online、MA-LMM、MovieChat 只在上游确实提供 memory bank/update API 时实现 `visual_memory`。
- state 必须包含可序列化的上游状态或同一进程中的明确对象句柄；第二个 chunk 必须真正消费第一个 chunk 更新后的 memory。
- 如果公开实现只提供离线视觉 token，没有在线 update API，则先标 fixed/blocked，不能手写一个“看起来像 memory”的平均向量冒充。

### 4.3 Decoder-KV 路线

- StreamingVLM、InfiniPot-V、MuKV、HERMES 只有在真实 causal LM `past_key_values`/等价 Cache 对象可观察、可传递时才声明 `decoder_kv`。
- smoke 默认 `native_compression_mode=off`、框架 policy=`identity`；只验证 state 递进，不验证压缩收益。
- 保存每层 KV shape、sequence axis、cache owner、第二步 cache hit/reuse；不得把 visual tower K/V 写成 decoder KV。

## 5. 执行任务与文件责任

### 任务 0：纠正历史状态和 schema

**涉及文件：**第 2.1 节文件、现有 `outputs/encoder-integration/` 说明文件、相关测试。

**验收：**17 条不再计入 native PASS；旧兼容结果带 `contract_only` 标记；全量 catalog 测试通过。

**提交边界：**`fix(status): 纠正兼容桥冒烟状态`

### 任务 1：冻结 native assets 和环境 profile

**涉及文件：**第 2.2 节文件、`integrations/<id>/upstream.lock.yaml`、`configs/encoders/*.yaml`。

**验收：**每条路线至少有一条真实来源记录；缺失项输出 `planned/blocked`，不创建 alias 文件；各环境 `import`/构造器预检通过。

**提交边界：**`feat(asset): 建立17路原生资产与环境清单`

### 任务 2：修复 fixed 原生 adapter

**目标：**`c3d`、`timesformer`、`videomae`。UniFormerV2/UMT 属于 foundation 批次，在任务 3 单独处理。

**测试：**`tests/test_legacy_c3d_integration.py`、`tests/test_transformers_video_integration.py`，新增各自 native state-key、feature-source、checkpoint identity 测试。

**真实验证：**每项用当前视频，保存 `outputs/encoder-integration/native/<id>/result.json`；不能引用 `contract-only` 结果。

**提交边界：**每个独立模型一个中文 commit，例如 `feat(encoder): 原生接入TimeSformer`。

### 任务 3：修复 foundation adapter

**目标：**`uniformerv2`、`umt`、`internvideo2`、`videomamba`、`vjepa2`。

**测试：**增加 loader state-key 检查、官方 preprocessor shape 检查、classifier 前 feature hook 检查。

**资源门禁：**显存或自定义 CUDA 不可满足时只记录 blocked；不降级到 R(2+1)D。

**提交边界：**`feat(encoder): 原生接入视频基础模型族`，或按模型拆分。

### 任务 4：修复固定/长视频 VLM adapter

**目标：**`longvu`、`videochat`、`videochat_flash`。

**实施重点：**将视觉 tower/projector 与文本生成解耦；smoke 只取视觉表征，不运行不必要的长文本 generation；记录 `visual_token_count`、projector dimension 和官方 model id。

**提交边界：**每条路线独立 commit；失败保留结构化日志。

### 任务 5：接入 visual-memory 路线

**目标：**`videochat_online`、`ma_lmm`、`moviechat`。

**实施重点：**依照各自原生 memory bank/update API；先验证 memory 类型和长度递进，再接统一 head；VideoChat-Online 只标 `visual_memory`。

**提交边界：**`feat(stream): 原生接入视觉记忆路线`

### 任务 6：接入 decoder-KV 路线

**目标：**`streaming_vlm`、`infinipot_v`、`mukv`；HERMES 只做回归。

**实施重点：**为每个上游写独立 worker/adapter，不共享假 cache；两个 chunk 在同一模型实例中执行，保存真实 KV 层数/shape/sequence axis。许可证不明的 MuKV/InfiniPot-V 不进入可发布代码，直到授权证据落地。

**提交边界：**每条路线独立 commit，例如 `feat(encoder): 原生接入StreamingVLM`

### 任务 7：矩阵编排和当前视频真实验证

**涉及文件：**`src/vadbench/smoke.py`、`src/vadbench/engine/integration_matrix.py`、`scripts/server/run_encoder_matrix.sh`、`schemas/encoder-smoke-v3.schema.json`。

**实施：**

- 为每个环境选择实际 Python executable；external runtime 必须启动 worker，不在主进程偷偷导入上游。
- 当前视频固定为 `data/smoke/mlvu-surveil-8.mp4`，保留 SHA256 `5c7dd43429c5e556de67489920a799af8fdb614a089ab52c04b1c3b044703963`。
- 分组运行但写入同一个 native matrix；结果路径、command、env、asset fingerprint 可重放。
- 单项失败继续，最终矩阵明确列出 `smoke_pass/failed/blocked/planned`，不能以“全部选中”代替成功。

**验收命令：**

```bash
cd /users/fotile/VAD
python scripts/server/prepare_native_assets.py --output outputs/native-assets-preflight.json
bash scripts/server/run_encoder_matrix.sh data/smoke/mlvu-surveil-8.mp4
python scripts/server/audit_native_matrix.py \
  --matrix outputs/encoder-integration/native/matrix.json \
  --schema schemas/encoder-smoke-v3.schema.json
```

### 任务 8：文档、审计和最终交付

**涉及文件：**

- 修改：`README-CN.md`
- 修改：`docs/progress/2026-08-31-current-progress.md`
- 修改：`docs/progress/encoder-integration-matrix.md`
- 更新：`docs/research/native-encoder-source-audit-2026-08-31.md`

**最终文档必须明确：**

- 原生通过数量和名单；
- 每条剩余路线的真实 checkpoint、许可证和运行环境；
- `planned/blocked` 的可复核证据；
- contract-only 结果不进入 native 统计；
- 不报告尚未运行的指标，不把 VQA 结果外推到 UCF-Crime。

## 6. 测试与 Git 规则

每项原生模型完成后，至少运行：

```bash
export PYTHONPATH=src
.venv/bin/python -m pytest -q tests/test_<target>_integration.py
/users/fotile/miniconda3/envs/mllm-comp-internav/bin/ruff check src tests scripts
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python scripts/server/audit_native_matrix.py ...
git diff --check
git status --short
```

每个独立功能完成后在服务器提交中文 Conventional Commit；不执行 `git push`。提交前必须确认：

- 没有把 alias/fallback checkpoint 写进 native registry；
- result provenance 的 `git_dirty=false`；
- 真实日志和错误文件保留；
- 未读取/提交密钥、token 或 `.env`。

## 7. 失败、回滚和阻塞政策

- 下载失败：检查 node2 出口、磁盘和代理一次；仍失败则保存日志并标 `blocked`，不反复轰击上游。
- 链接失效：寻找同一官方仓库的固定 tag/release/官方 HF mirror；找不到则 `blocked`。
- checkpoint state-key 不匹配：停止该项，修正官方 config/代码版本；禁止 `strict=False` 静默吞掉大量 missing keys。
- 许可证缺失或限制再分发：不复制上游代码/权重进入主树；只记录来源和 blocked 证据，等待用户/作者授权。
- GPU 被占用：先用 `nvidia-smi` + `ps` 映射用户和命令；不杀进程、不抢卡。若无可用资源，保留 blocked 并报告精确原因。
- 如需回滚，只回滚本计划引入的独立 commit；不使用 `git reset --hard`，不覆盖用户改动。

## 8. 交付后的可追溯关系

```text
native catalog id
  -> definition YAML
  -> upstream.lock.yaml + checkpoint registry
  -> isolated environment / worker
  -> current-video native smoke v3
  -> shape/dtype/finite/timeline/state audit
  -> JSON artifact + log + git commit
```

Goal 只有在剩余路线逐条满足第 1.2 节并且不存在未披露的 fallback 时才能重新标记完成。`contract_only` 结果永远不能替代 native smoke。

## 9. 2026-09-01 执行状态

当前计划已无 `planned` 项：14 条原生路线通过当前视频 smoke，11 条路线进入有证据的 `blocked`。`blocked` 项不得被兼容桥、随机权重、同族模型或未授权权重替换；解除门禁时必须补齐同一目标的官方资产、许可证、checksum 和真实 smoke。
