# 复现指南

本指南针对默认 `segmented` 路径。它是固定形状的本地请求评测，不是 HTTP 服务启动教程。
新归约融合的独立流程见 [B48 实验说明](../experimental/fused_reduce_v2/README.md)；不要在同一 Python 进程热切换策略。

## 1. 准备环境与权重

在已经配置好、只暴露一张分配 C500 的 Linux 环境中执行 GPU 实验。
先激活经过验证的 MetaX / HF 环境，核对[环境说明](ENVIRONMENTS.md)与[冻结配置](../strategies/segmented/FROZEN_CONFIG.json)。
脚本会检查关键依赖版本；版本不匹配应单独验证，不要删除断言继续跑。
仓库不负责安装系统驱动或替换 SDK，也不包含模型权重。

在仓库根目录设置现有的只读模型位置，并执行 CPU 校验：

```bash
export MODEL_DIR=/path/to/read-only/Qwen3.5-27B

python3 tools/verify_model.py \
  --model "$MODEL_DIR" \
  --manifest models/Qwen3.5-27B.sha256.json
```

校验只读取已有文件，不下载、不修改权重。清单不匹配时应确认模型版本与完整性，不能直接改清单掩盖差异。

## 2. 设置隔离目录与统一截止

为本次实验选择一个新的目录；同一策略的已有结果不得覆盖。
`COMMON_GPU_LOCK` 应指向部署中其他控制器共同使用的现有锁文件，不能用新随机锁绕过正在进行的实验。

```bash
export RUN_DIR=/path/to/new-isolated-run
export COMMON_GPU_LOCK=/path/to/existing-common-gpu.lock
export DEADLINE_UTC="$(python3 -c 'from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) + timedelta(minutes=55)).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
```

下面的辅助函数让各阶段共用同一个截止时间、工作目录和 GPU 锁：

```bash
bounded() {
  python3 tools/launch_bounded.py \
    --deadline "$DEADLINE_UTC" \
    --workdir "$RUN_DIR" \
    --shared-lock "$COMMON_GPU_LOCK" \
    -- "$@"
}
```

Launcher 会创建隔离的缓存与临时目录、限制可见设备，并在截止时结束自身启动的进程组。
它只接受未来一小时内的截止时间。多个阶段不能各自重新计算截止时间来延长同一轮实验。

- 检查已有控制器、锁和 GPU 进程，确认不会并发争用同一张卡。
- 多个既有共同锁可重复传入 `--shared-lock`。
- 已由独立的有截止控制器持有共同锁时，不要嵌套获取同一个锁；由该控制器统一串行调度。
- Launcher 的 `controller.lock` 与 worker 的 `gpu-work.lock` 是不同的锁。

## 3. 先运行微内核检查

```bash
bounded python3 benchmarks/micro.py \
  --strategy segmented \
  --workdir "$RUN_DIR"
```

该阶段记录 30 组形状、布局与归一化组合的 oracle，包括输出、状态及实际 dispatch。
原生 fallback 必须显式记录；微内核正确或更快不等于整模型已通过准入。

## 4. 运行整模型配对评测

```bash
bounded python3 benchmarks/e2e.py \
  --strategy segmented \
  --model "$MODEL_DIR" \
  --workdir "$RUN_DIR" \
  --batches 1,8,48
```

也可用 `--batches 48` 只复现该测试点。当前入口仅接受 batch 1、8、48 的非重复子集。
启动整模型阶段前，统一截止至少应剩余 500 秒；这不保证所有计划点都能在预算内完成。

脚本会在计时前执行短序列全词表 logits、四缓存及 A/B/A 请求检查，
然后执行真实 128 入 / 128 出生成检查与两侧各三次端到端复测。
实际门槛及完整 128 步独立诊断的范围见[正确性协议](CORRECTNESS.md)与[最新实验记录](OPTIMIZATION_ROUND2.md)。

计时包括 CPU 分词、H2D、fresh-cache prefill 与全部生成、D2H、文本解码；
加载、编译、capture、首轮 warmup 和独立正确性检查不计入请求延迟，另行记录。
不共享请求前缀，不量化，不修改 EOS 规则、输入输出长度或 token 计数。

遇到失败时：

- 保留源代码、日志、失败字段及已经完成的记录；不要删除失败样本后重算为通过。
- 原生基线自身不一致同样是失败，不能给该轮候选生成合格速度排名。
- 部分完成、OOM、未测试或截止终止不等于 0 token/s，也不能当作完整的三次重复。
- 使用新工作目录和新进程修复、重试；不同轮次不能无标注拼接样本。

## 5. 在 CPU 上汇总结果

只有完整通过准入的数据才能产生配对统计：

```bash
python3 tools/qualify.py \
  --results "$RUN_DIR/results/segmented" \
  --candidate segmented \
  --output "$RUN_DIR/paired-summary.json"
```

汇总会检查重复次数、完整输入输出哈希、有限值、token 计数、fallback 和实际融合 dispatch。
Dispatch 计数器是 Python warmup/capture 的累计计数，不是 CUDA graph 每次回放的 host 调用次数。
当前汇总 CLI 支持 `segmented` 和 `direct-state`，不能把其他实验名称直接传入该参数。

## 6. 退出与后续验证

实验结束后检查 launcher 的 `launches.jsonl`、自身 worker 退出情况，并使用部署提供的只读工具
（例如 `mx-smi`）核验已分配 GPU 的利用率和进程列表。Launcher 退出并不代替这一步。
不要结束与本轮无关的进程。

`direct-state` 是未准入的研究路径，已有完整请求失败，不能因短 oracle 通过就视为可替换默认方案。
新的硬件分配、依赖版本、模型、上下文、padding、多模态或并发执行均应建立独立验证记录。
公开提交仅收源码与去标识摘要，不提交权重、SDK、凭据、私有端点或私有原始日志。
