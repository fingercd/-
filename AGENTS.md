# VADBench 项目协作规则

## 项目目标

本仓库以 UCF-Crime 为首个基准，统一比较两类视频表征路径：

1. 固定时长 clip 编码器，例如 VideoMAE V2；每个 clip 独立前向，不宣称支持跨 clip KV cache。
2. 面向长视频或视频流的视觉 token、视觉记忆或语言模型 decoder KV cache 路径，例如 HERMES。

框架必须允许后续编码器通过注册机制接入，并让采样、编码、压缩、训练、评测和产物格式彼此解耦。研究结论必须能追溯到配置、权重 revision、数据 manifest、代码提交和运行产物。

## 进入项目后的必读顺序

1. `README-CN.md`
2. `docs/research/video-encoder-survey-2026-08-31.md`
3. `docs/research/ucf-crime-protocol.md`
4. 当前任务对应的 `docs/plans/` 文档和实验 YAML
5. 修改模块邻近的源码、测试和 schema

如果文档与可执行代码冲突，以测试覆盖的接口和 schema 为当前事实，同时在同一功能提交中修正文档。

## Git 与提交

- 只在独立功能分支工作，不直接改 `main`。当前课题分支为 `feat/video-encoder-benchmark-framework`；不要自行切换或重建分支。
- 修改前运行 `git status --short`，保留并绕开用户或其他协作者已有改动。
- 一个可独立验证的功能对应一个提交；提交信息使用中文 Conventional Commits，例如 `feat(encoder): 接入 VideoMAE V2 固定片段编码器`。
- 用户已明确要求本课题每个功能提交后推送；推送前必须先完成该提交对应的最小验证，并记录失败或未验证项。
- 未经当前任务授权，不合并分支、不改写历史、不创建 release。

## 目录与数据边界

- `src/vadbench/`：框架源码；`tests/`：自动化测试；`configs/experiments/`：可复现实验配置。
- `schemas/`：JSON/JSONL 契约；`registry/`：权重与编码器登记信息。
- `data/`、`weights/`、`outputs/`、`external/` 只保留占位、manifest、校验值或说明；禁止提交原始视频、模型大权重、特征 blob、缓存 dump 和第三方仓库副本。
- 服务器目标固定为 `/users/fotile/VAD`。真实 UCF-Crime 数据放在服务器已有数据盘，并通过显式软链接接入；禁止复制进 Git 工作树。
- 权重必须固定上游 repo、revision 和校验值；下载前检查许可证，不能用本仓库 MIT 许可证覆盖上游权重或第三方代码条款。
- 不读取、提交或输出 `.env`、token、认证缓存、私钥和 `.sandbox-secrets`。

## 服务器约束

- `node2` 是集群唯一外网出口：Git 拉取/推送、Hugging Face 权重下载和第三方仓库获取均在 `node2` 完成。
- `node3` 无外网：只运行已经同步好的代码、依赖、权重和数据；失败时先排除缺失离线资产，不反复尝试联网。
- 部署前后记录 `hostname`、Git commit、Python/PyTorch/CUDA 版本、GPU、可用磁盘和数据软链接解析结果。
- 下载前用 `df -h` 和目标文件预估体积检查空间；权重必须落到明确目录，禁止占满系统盘或用户家目录。
- 服务器上的删除、覆盖、移动或软链接替换必须先解析绝对路径并确认目标位于 `/users/fotile/VAD` 或明确的数据目录内。

## 编码器与缓存术语

- 统一输入是 `BTHWC uint8`，统一特征输出是 `features[B,S,D]`；时间轴和原视频坐标通过 `TokenTimeline` 保留。
- 固定 clip adapter 实现 `VideoEncoderAdapter`；真正跨 chunk 复用状态的 adapter 才实现 `StreamingVideoEncoderAdapter`。
- 必须分别标注 `vision_tokens`、`visual_memory` 和 `decoder_kv`。Q-Former 中充当 key/value 的视觉特征不等同于语言模型 decoder KV cache。
- 不得因为模型内部使用 Transformer、SSM，或推理时设置了 `use_cache=True`，就宣称视觉编码器支持跨视频片段缓存；必须有可观察的 `StreamState`、缓存更新和等价性/语义测试。
- 缓存策略通过 `CachePolicy` 注入，不在具体 adapter 内硬编码；`identity` 是无压缩对照，压缩实验必须同时记录预算、保留率、峰值显存、吞吐/延迟和精度。

## UCF-Crime 协议

- 使用官方视频级划分：训练 1,610，测试 290；禁止从 clip 级随机重划分。
- 弱监督训练只消费训练视频级标签；官方测试时间标注只用于最终评测。
- 默认复现 32 个时间段的 Sultani-compatible MIL bag（这是项目基线协议，不是数据集唯一采样法），并将片段分数投影回帧后计算 frame-level ROC-AUC。
- 官方 TXT/MAT 时间端点是 MATLAB 1-based inclusive；内部 `TemporalSpan` 必须转成 zero-based half-open `[raw_start-1, raw_end)` 并保留 raw provenance。
- UCA 是带时间戳的自然语言事件标注，不是现成的二值强监督真值；没有经过显式异常语义映射的 UCA 区间不得直接标成异常。
- 截至 2026-08-31，FS-UCF-Crime Zenodo 记录只含 placeholder，不能作为可用训练标注来源。
- 数据导入必须以 `video_id`/路径检测 train/test 重叠，检查重复 ID、越界时间区间、帧率/帧数缺失以及归一化统计对测试集的拟合泄漏；内容 hash/视觉近重复属于额外审计，未运行时必须如实记录。

## 产物与可复现性

- 特征索引、预测、指标和缓存遥测必须符合 `schemas/` 中的版本化 schema。
- 每次运行在 `outputs/runs/<run_id>/` 写入配置快照、代码提交、数据/权重指纹、环境信息和状态；失败运行也要保留错误与阶段信息。
- 路径字段应能在服务器部署根目录下重定位；不要把某台机器的临时绝对路径写成数据身份。
- 比较编码器时固定数据划分、采样和检测头；任何改变必须单独列为消融，不能与缓存收益混为一谈。

## 验证要求

- Windows 全量测试：`.venv/Scripts/python.exe -m pytest`
- Linux/服务器全量测试：`python -m pytest`
- 改动 Python 后运行相关测试和 `python -m compileall src tests`；配置或 schema 改动同时运行对应校验命令。
- 真实权重冒烟必须使用至少一个实际视频片段，不以 mock 代替，并保存输入、输出 shape、dtype、设备、耗时、峰值显存及缓存遥测。
- 无法完成真实验证时明确说明缺失的资产、硬件或权限，不能把单元测试通过表述为真实模型已跑通。
