# C500 Kernel Lab

[![CPU contracts](https://github.com/123123213weqw/c500-kernel-lab/actions/workflows/cpu-contracts.yml/badge.svg)](https://github.com/123123213weqw/c500-kernel-lab/actions/workflows/cpu-contracts.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

面向 **MetaX C500** 的大模型推理内核优化与可复现评测项目，目前聚焦 **Qwen3.5-27B 文本推理**。
通过 Triton 内核与分段执行图减少推理开销，同时以原生 Transformers 实现作为数值参考。

[快速开始](#快速开始) · [性能结果](#性能结果) · [支持范围](#支持范围) · [文档](#文档) · [参与贡献](#参与贡献)

## 项目特点

- **正确性优先**：检查全词表 logits、四类逻辑缓存、完整生成结果及跨请求状态重置。
- **内核与执行图协同**：提供 Gated DeltaNet（GDN）局部融合、精确归约及固定形状分段图实验。
- **可复现评测**：固定模型、精度、输入输出长度和 token 计数，保留至少三次同条件重复测量。
- **透明的实验记录**：提供冻结配置、权重校验清单、CPU 契约测试，以及失败、回退和未测试项。

> **项目状态：实验性。** 这是非官方研究项目，不是通用推理服务器。
> `segmented` 是默认复现路径；新归约融合仅在已验证的 B48 测试点准入，不代表所有尺寸或软件环境均可直接使用。

## 快速开始

### 获取源码与运行 CPU 测试

在 Linux / macOS 上，CPU 测试与审计工具仅依赖 Python 标准库，无需 GPU、模型权重或 MetaX SDK。
CI 使用 Python 3.10。

```bash
git clone https://github.com/123123213weqw/c500-kernel-lab.git
cd c500-kernel-lab

python3 -m unittest discover -s tests -v
python3 tools/check_repository.py
```

### 在 C500 上复现

GPU 实验需要预先配置已验证的 MetaX / PyTorch / Triton 环境，并自行准备模型权重。
仓库不分发 SDK、二进制 wheel 或 checkpoint，也不提供通用的 `pip install` 部署入口。

1. 按[环境说明](docs/ENVIRONMENTS.md)核对依赖与[冻结版本](strategies/segmented/FROZEN_CONFIG.json)。
2. 按[复现指南](docs/REPRODUCING.md)校验权重，设置隔离目录、共同 GPU 锁和统一截止时间。
3. 先运行微内核 oracle，再运行整模型正确性检查与端到端评测。

默认路径见[完整复现步骤](docs/REPRODUCING.md)；B48 新归约融合见[实验说明](experimental/fused_reduce_v2/README.md)。

## 性能结果

**测试条件：** 单张 C500 64 GiB，Qwen3.5-27B，BF16 权重 / 原生 FP32 recurrent state，
batch 48，输入 128 / 输出 128，greedy、无 padding。下表为每项 **n=3 中位数**，吞吐为整个 batch 的输出 token/s。

| 路径 | 输出 token/s ↑ | 整批延迟（秒）↓ | 峰值 allocated（GiB） |
|---|---:|---:|---:|
| 原生 HF（新候选同轮基线） | 118.14 | 52.007 | 60.79 |
| 默认 `segmented`（复测 1） | 222.96 | 27.557 | 58.95 |
| 默认 `segmented`（复测 2） | 222.69 | 27.590 | 58.95 |
| **`fused_reduce_v2`（B48 实验）** | **243.33** | **25.249** | **58.95** |

新候选相对其同轮原生 HF 基线为 **2.06×**，相对两轮已优化 `segmented` 复测再提升 **9.1%–9.3%**。
原路径两轮计时均先于新候选，并非随机交错或新候选前后夹测；不据此宣称普适提升或正式统计显著性。

**计时范围：** CPU 分词 → H2D → fresh-cache 全量 prefill 与生成 → D2H → 文本解码。
不含加载、编译、graph capture、首轮 warmup 和独立正确性检查；显存峰值包含 graph pool。
这不是 HTTP、排队、网络或冷启动测试，也不代表硬件物理绝对极限。

数据来源：[结构化结果](results/round2/paired-summary.json) · [验证记录](results/round2/validation.json) · [完整分析与限制](docs/OPTIMIZATION_ROUND2.md)。
历史轮次见[历史结果](results/historical/qwen35-27b.json)，不同硬件分配的结果不混算增益。

## 支持范围

| 组件 | 定位与当前边界 |
|---|---|
| `strategies/segmented` | 默认复现路径；冻结的 GDN 局部融合与 8 步分段图，固定形状评测 |
| `experimental/fused_reduce_v2` | 新精确归约融合；本轮仅 B48、128 入 / 128 出通过完整请求复测 |
| `strategies/generic` | Dense-layout GDN 研究路径；部分测试点存在性能回退 |
| `experimental/direct_state` | 直接状态写回研究；历史完整请求等价失败尚未解决，未准入 |

当前 GPU 路径使用 Linux、Python 3.10、MetaX 构建的 PyTorch 2.8 / Triton 3.0，以及 Transformers 5.3.0；
精确版本以冻结配置为准。新的 SDK、模型、batch、上下文、padding、多模态、多卡或并发请求均需重新验证。

**已知限制：** 已记录原生基线自身复测不一致的情况；B1 新候选没有合格端到端成绩，B8 未进行本轮完整请求测试。
失败、回退和未测试项不会记为 0 token/s，也不会因短测试通过而覆盖完整请求失败。
官方重建栈未通过相同数值准入，因此不提供与其的速度排名。详见[负结果记录](results/round2/negative-results.json)及[正确性协议](docs/CORRECTNESS.md)。

## 仓库结构

```text
benchmarks/      微内核、配置搜索与端到端评测入口
strategies/      已冻结的复现路径与独立研究实现
experimental/   尚有限定适用范围或未准入的候选
diagnostics/   原生参考与数值分歧诊断
tools/         截止控制、结果准入、权重校验与源码审计
tests/         无 GPU 依赖的契约测试
docs/          环境、复现、正确性与实验分析
results/       去标识的结构化结果和负结果摘要
models/        权重校验清单，不包含模型权重
```

## 文档

| 文档 | 内容 |
|---|---|
| [复现指南](docs/REPRODUCING.md) | 默认路径的准备、执行、结果汇总与退出检查 |
| [环境说明](docs/ENVIRONMENTS.md) | 已验证依赖及不同数值实现的边界 |
| [正确性协议](docs/CORRECTNESS.md) | Logits、缓存、完整生成与准入要求 |
| [最新优化分析](docs/OPTIMIZATION_ROUND2.md) | B48 增量优化、性能范围与完整负结果 |
| [早期实验记录](docs/RUN_2026-10-30.md) | 首轮复测及尚未解决的问题 |
| [贡献指南](docs/CONTRIBUTING.md) | 可贡献方向、测试要求与提交边界 |

## 参与贡献

欢迎通过 [Issues](https://github.com/123123213weqw/c500-kernel-lab/issues) 报告问题，或通过 Pull Request 提交改进。
建议优先从权重无关的最小复现、布局与缓存 oracle、数值分歧定位、评测工具和文档入手。

新增优化应附适用形状、stride、dtype、回退条件、正确性结果，以及至少三次同口径端到端复测。
只通过微内核测试的实现仍应保留为实验项。提交前请阅读[贡献指南](docs/CONTRIBUTING.md)，并运行上面的 CPU 测试与审计命令。

## 许可证

本仓库新增代码使用 [MIT License](LICENSE)。模型、SDK 及第三方组件分别遵循其自身许可证；
本项目的 MIT 许可不替代或重新授权这些外部组件。详见 [NOTICE](NOTICE)。
