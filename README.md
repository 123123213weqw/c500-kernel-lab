# C500 Kernel Lab

面向 **MetaX C500 / Qwen3.5-27B** 的精确计算与端到端推理实验仓库。
不是通用推理服务器，不宣称硬件物理极限，也不宣称超过官方栈。

## 已有实测（历史轮次，不能和新分配设备混算）

BF16 权重、原生 FP32 recurrent state，输入128 / 输出128，greedy、无 padding，
CPU 分词→H2D→全量 fresh-cache prefill+decode→D2H→解码；每项 n=3 中位数：

| Batch | 原生 HF token/s | segmented token/s | 同口径加速 |
|---:|---:|---:|---:|
| 1 | 9.37 | 16.42 | 1.75× |
| 8 | 51.18 | 97.50 | 1.90× |
| 48 | 112.50 | 222.80 | 1.98× |

加载、编译、graph capture、首轮 warmup 不计入以上请求耗时，另行记录。
显存峰值计入 graph pool。不是 HTTP、排队、网络或冷启动测试。
详细数值及失败项见 `results/historical/qwen35-27b.json`。

**不能只挑涨分：** generic GDN 局部融合，B1 +21.08%，B48 **−2.78%**。
官方0.19重建栈的短 holdout raw logits 未通过逐值完全相等门槛；
B17 有1/17完整序列不同。没有合格的官方速度排名，不能把它记成0 token/s。

## 目录

- `strategies/segmented/`：历史获胜路径的冻结 kernel + 8步分段 graph。
- `strategies/generic/`：支持 dense permutation 的独立 GDN；不是默认全场景加速。
- `experimental/direct_state/`：直接写最终 recurrent state，原先只有微内核验证，整模型以本轮结果为准。
- `benchmarks/micro.py`：30组输出/状态严格 oracle，显式记录 native fallback。
- `benchmarks/e2e.py`：固定形状端到端、完整输出校验、fresh-request A/B/A 防泄漏。
- `tools/qualify.py`：失败关闭的准入/配对统计；不以输出ID一致替代 full-vocab/caches 检查。
- `tools/launch_bounded.py`：单任务锁、统一截止、只终止自身启动的进程组、独立缓存。
- `tests/`：无需 GPU 的契约与截止测试；`docs/`：复现约束、贡献路径和来源。

## CPU 验收（不加载 Torch/GPU）

```bash
python3 -m unittest discover -s tests -v
python3 tools/check_repository.py
```

## GPU 复现

须在已授权且只有一张分配 C500 的环境执行。**不在本机跑 GPU/Rust 编译。**
环境精确版本见 `strategies/segmented/FROZEN_CONFIG.json`；不要用普通 CUDA
PyPI wheel 替换 MetaX wheel。仓库不提供自动改驱动、系统包、时钟或认证的脚本。

每次使用新工作目录、同一未来截止时间（最长1小时）。先微内核、后整模型：

```bash
# 在远端仓库内，先按现有部署激活已验证的 MetaX/HF 环境。
export MODEL_DIR=/path/to/read-only/Qwen3.5-27B
export RUN_DIR=/path/to/new-isolated-run
export DEADLINE_UTC=YYYY-MM-DDTHH:MM:SSZ
python tools/launch_bounded.py --deadline "$DEADLINE_UTC" --workdir "$RUN_DIR" -- \
  python benchmarks/micro.py --strategy segmented --workdir "$RUN_DIR"
python tools/launch_bounded.py --deadline "$DEADLINE_UTC" --workdir "$RUN_DIR" -- \
  python benchmarks/e2e.py --strategy segmented --model "$MODEL_DIR" --workdir "$RUN_DIR" --batches 1,8,48
```

存在其他控制器时，须给 launcher 添加其共同锁 `--shared-lock /path/to/shared.lock`；
不得绕过锁并发争用 GPU。launcher 的 workdir controller lock 与 worker GPU lock 分离。
已经在独立有截止控制器下运行时，不再嵌套获取同一共同锁。

`direct-state` 替换上述两处 `--strategy` 即可测试实验路径。
两个策略必须使用**不同全新进程**；不得在同一解释器热切换 `exact_gdn`。
新硬件/新软件版本、padding、长上下文、多模态、并发请求等均需重新验证；
不能从本矩阵外推。退出后须另行核验自身进程退出和 GPU 空闲。

## 当前仓库状态

本地与授权远端均为私有工作仓库；没有创建 GitHub 远程、公开推送或上传权重。
MIT 仅覆盖本仓库新增代码；外部组件/模型各遵其许可，见 `NOTICE`。
完整原始日志留在独立实验归档，本仓库只收代码、去标识摘要和校验信息。
