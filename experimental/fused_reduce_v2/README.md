# 精确归约融合实验（B48限定）

这不是默认或通用实现。只支持逐 token、FP32 row-major recurrent state、K=V=128；
其他形状/布局回退原生。保持原生归一化、BF16边界及独立状态回写，不直接修改输入状态。

安装版 MetaX 原生 sum 的实际 dispatch 为8行 lane，各按步长8顺序累加16项，
再按4/2/1合并。显式 RN 算术复现此顺序，移除 dec/prod 大临时张量；
不能用另一版 PyTorch/SDK 的归约顺序直接套用。首个4-lane假设已被微测试拒绝。

本轮B48结果、完整正确性范围、原生基线自身不一致与未测试尺寸见
`../../docs/OPTIMIZATION_ROUND2.md`。B1/B8没有新候选合格端到端成绩。
微内核速度不得作为模型吞吐。配置搜索是 kernel-only，不代表整模型提升。

复现先设置所有隔离缓存目录、共同GPU锁及未来硬截止，校验原始checkpoint，
保持冻结环境和单卡；不要直接运行过期控制器。用通用的有界 launcher 串行执行：

共同锁必须指向部署中现有的授权GPU锁；不要用随机新锁绕过其他控制器。
先定义同一未来硬截止 `DEADLINE` 和 `COMMON_GPU_LOCK`，例如：

```bash
mkdir -p "$RUN_DIR/results"
bounded() {
  python3 tools/launch_bounded.py --deadline "$DEADLINE" --workdir "$RUN_DIR" \
    --shared-lock "$COMMON_GPU_LOCK" -- "$@"
}
bounded python3 benchmarks/tune_reduce_v2.py "$RUN_DIR"
bounded python3 benchmarks/micro_reduce_v2.py --strategy fused-reduce-v2 --workdir "$RUN_DIR"
bounded python3 benchmarks/e2e_reduce_v2.py --strategy fused-reduce-v2 --model "$MODEL_DIR" --workdir "$RUN_DIR" --batches 48
```

搜索结果必须为冻结的BV128/W4后再比较本轮数据。benchmark内部仍强制三组短全词表/
四缓存/A-B-A、真实128生成、两侧n=3完整输出一致；本轮此外独立执行了B1的完整128步
全词表与四缓存诊断。任何缺失/失败都不能以放宽误差或只比较部分token替代。
