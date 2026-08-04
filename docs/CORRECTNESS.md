# 准入协议

1. 同一模型权重、dtype、FP32状态、真实输入ID、greedy规则及输入/输出计数。
2. 三个随机 holdout：(B,L,N)=(1,13,4),(3,127,4),(17,32,4)。
3. prefill和decode完整词表 raw logits 有限且 `torch.equal`；完整输出ID相等。
4. key/value/conv/recurrent四类逻辑 cache 相等。不同后端物理布局不能直接当作逻辑一致。
5. graph引擎 A→B→A 重放必须逐次匹配新请求，不能复用前一请求状态。
6. 两路径的真实128步 warmup须完整词表有限；所有计时输出ID、输入SHA、文本一致。
7. 正向 fused dispatch、fallback=0；native fallback正确不代表优化生效。
8. 以上全部通过后才汇总n≥3配对测速，保留负提升、OOM、未测、超时、兼容失败。

所有短holdout通过不等于任意长度全数值证明。固定128→128之外尚无泛化保证。
本仓库不提供放宽误差阈值的“显示优势”选项。

## 异步 serving 的采集教训

vLLM外部 request_id 与内部 nonce ID不同。以经过源码核验的一对一映射关联。
全局 model call数不等于每请求输出长度；某次调用还可能改变行顺序。
必须按request逐行整理时间顺序，证明raw-logit argmax对应实际返回token。
不启用不安全 pickle-function RPC，不关闭request随机化。
历史正式比较用命名worker extension和字符串collective_rpc；未向仓库引入SDK源码。
