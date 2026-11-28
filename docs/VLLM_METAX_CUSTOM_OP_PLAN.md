# vLLM-MetaX 接入自有 GDN 算子的提案

**状态：设计，尚未启用算子，也没有集成后的 GPU 成绩。**

目标是在 vLLM 的调度、缓存和服务路径不变的情况下，测出自有算子能否缩短 Qwen3.8-27B BF16 的单请求 decode。先冻结已在 C500 跑通的 vLLM 0.28.0 + vLLM-MetaX 0.28.0、BF16 权重和 FP32 recurrent state。不叠加 DFlash2、W12、量化或前缀缓存；prefill、采样及页缓存保留原实现。优先优化单 token，并检验高并发没有回退，不用自有固定 B64 和在线 c=16 作 A/B。

## 入口与回退

1. **先证实 worker 实际派发。** 固定期望版本，记录实际加载的模块、签名、dtype 和布局。以 [vLLM v0.28.0 的 Qwen GDN 实现](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1632-L1684) 为线索，针对纯 AR 检查 `_forward_core_decode_non_spec` 后的 `fused_recurrent_gated_delta_rule_packed_decode` 是否实际命中。并列测 MLP、LM head、GDN 和调度耗时；如果 GDN 占比很小，不把它当作解码提速的主要来源。
2. **适配算子，不照搬旧的 Transformers 钩子。** 从仓库的 [精确归约融合实验](../experimental/fused_reduce_v2/README.md) 借用归约思路；旧实现历史上只在指定栈的 B48 获准，在这轮新栈与参考已经出现位差。新适配要尊重 [vLLM 的 packed decode ABI](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py#L287-L340)：BF16 输入、FP32 状态、按 slot 索引读取/更新、V/K stride、空 slot、原地写回、输出缓冲及相同舍入位置。第一版只接 K=V=128 的单 token、纯非推测 decode，尽量直接处理原有 slot，不用通用 transpose/clone 否则可能抵消收益。
3. **接入单独的可选 worker 插件。** 拟用 [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system/) 在独立 venv 的 worker 初始化时安装版本钉死、幂等的窄适配，不修改 SDK，也不覆盖 MetaX 平台插件。默认 off；shadow 仅用于诊断，在同一初始状态的独立 scratch 上对照，不触碰真实缓存、不跑吞吐；on 只在门槛通过后开。未知布局、混合 prefill/decode、推测解码以及无法证明的索引/别名回原厂路径。标志在 graph capture 前确定，换策略重新起进程，不在 graph replay 中动态切换。

## 准入与测速

- **先算子，后整模。** 至少 30 个小型 oracle，覆盖随机状态、BF16 舍入边界、空 slot、重排、回收重用、异常布局。然后按[仓库正确性协议](CORRECTNESS.md)做三个 holdout、全词表 raw logits、KV/conv/recurrent 四类逻辑缓存、完整 token IDs、A→B→A fresh-request/graph 重放和真实 128 步；同一输入和 checkpoint 下要求逐值一致。仅有 near-tie、输出字符串相同、或所有请求实际上回退均不算通过。
- **再对比同一个 vLLM。** 单 C500、共同 GPU 锁、独立目录和硬截止，原厂 vLLM BF16 AR 对完全相同的 vLLM + 仅替换 GDN。同提示词、输入/输出长度、EOS、seed、调度、cache budget 和 graph 设置，至少三组配对交错复测；报告 prefill tok/s、纯 decode/TPOT、端到端 tok/s、TTFT、峰值显存、失败及回退。先 B1/B8/B16 固定形状确认算子层收益，再跑 loopback c=1/4/8/16，不混用两类结果。采样实际 graph replay；host dispatch 计数不能当成 replay 计数。
- **取舍。** 先不改 prefill：自有 B1 128 入 / 128 出的 decode 约占全程 99%，优先获知 vLLM 中真正的 decode 瓶颈。先不引入 W12/DFlash2：多项一起改会混淆算子收益，而且复测过的 DFlash2 尚不满足严格一致性。如果 GDN 无法精确对齐或整模无收益，保留原厂默认，记录首个分歧/回退，转查数据搬运、投影和模型权重读取，而不是放宽门槛。

相关版本安装见 [MetaX 官方说明](https://vllm-metax.readthedocs.io/en/latest/getting_started/installation/en.html)；历史基线见 [Qwen3.8-27B/C500 归档](../results/2026-11-27-qwen38-c500/README.md)。
