# 四组 Encoder 隔离环境与原生资产迁移

> 日期：2026-09-03

## 目标

在不修改现有四个 Python 环境的前提下，建立四组新环境、模型覆盖层、21 路运行 catalog、25 路候选清单和 native-only 资产/冒烟产物。

## 固定边界

- node2 只负责网络资产，node3 负责真实权重验证。
- 旧环境、旧 external 和现有权重只读。
- 只有代码和目标 checkpoint 都存在的路线进入运行 catalog；人工下载路线可注册但在资产到位前 fail-closed。
- 无权重身份的路线只留候选记录。
- 未明确许可证的路线即使技术前向通过也保持 blocked。
- 不允许随机权重、替代 checkpoint 或同族模型冒名。
- 根卷预计剩余低于 400 GiB 时停止下载。

## 交付

1. registry/encoder-candidates.yaml 覆盖 25 路研究候选。
2. registry/encoder-integrations.yaml 只登记 21 路运行目标。
3. registry/encoder-environments-v2.yaml 固定四组环境、受保护旧路径和覆盖层。
4. server 工具负责旧环境指纹、环境 bootstrap、overlay、资产校验、人工清单、native runner 和结果聚合。
5. 所有可运行模型在 data/smoke/mlvu-surveil-8.mp4 上重新验证，不复用旧 smoke。
