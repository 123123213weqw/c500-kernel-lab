# Qwen3.8-27B 单卡 C500 实验归档

2026-11-27，同一张 MetaX C500 64 GiB，校验过的官方 Qwen3.8-27B BF16 目标权重。此前仓库的 Qwen3.5-27B 实验不作跨模型速度比较。这一轮分三类，**不是同一个排行榜**：

| 记录 | 工作负载 | 作用域 |
|---|---|---|
| [自有 BF16](own-bf16.json) | 本地固定批，greedy、固定 128 输出 | MetaX PyTorch + Transformers，不用 vLLM；不计分词、HTTP、排队、加载、编译。正式峰值为三次中位数；短扫只用于选点。 |
| [Issue #86 独立引擎](issue86.json) | 单并发 5 题 × 256 上限、两轮；批处理 20 题 × 128 上限、每档一轮，另做选中档 256-token 对照 | DFlash2 / DSpark、Tree / TLV、fixed / shared / dynamic；热身后聚合输出，不代表严格无损或正式准入。 |
| [Issue #86 vLLM 对照](issue86.json) | 单并发同一组 5 题、两轮；并发档同一组 20 题、每档一轮；256 输出上限 | vLLM 0.28 + vLLM-MetaX，loopback 服务；BF16 AR、stock DFlash2、W12 v3+DFlash2；和自有引擎分开记。 |

## 已测观测

- 自有固定 B8、2048 入 / 128 出：prefill-only 图 **3,242.26 输入 token/s**。自有固定 B64、128 入 / 128 出：eager **480.82 decode tok/s**、**418.50 全程输出 tok/s**。这是两种形状，不外推到线上并发。
- 自有 BF16 无 vLLM；B1、128 入 / 128 出的分段图为 **20.70 全程输出 tok/s**。单请求约 0.081 秒到首 token，后续生成耗时约 6.10 秒。
- 独立 DFlash2 Tree-16 在自己的短题负载上观测到 **84.04 tok/s**，但与同引擎 AR 的逐 token 一致性检查未通过；不宣称无损。公开归档不含原始输出，只记录比较计数与诊断汇总。
- 20 题、并发 16、每题实际 256 输出的单轮对照中，vLLM BF16 AR **207.87**，独立 DFlash2 **203.26**，独立 DSpark **193.34**，vLLM W12 v3+DFlash2 **111.87 tok/s**。AR/独立 DFlash2 平均首 token 分别为 **0.56 s / 1.30 s**；并发样本不足三组，不宣称排名稳定，也不能和自有固定 B64 求加速比。
- 归档的 W12 v2/v3、旧 GDN oracle、MLP 子算子均标明微测范围，不是端到端的通过结果。保留 [OOM、超时、数值失败与未测项](negative-and-unmeasured.json)。

[环境与公开来源](provenance.json) · [vLLM 自有算子接入提案](../../docs/VLLM_METAX_CUSTOM_OP_PLAN.md) · [GitLink Issue #86](https://www.gitlink.org.cn/ccf-ai-infra/opengrow/issues/86) · [SpecPilot 论文](https://arxiv.org/abs/2611.02501)

公开 JSON 保留各配置与各轮计时、样本数、能公开的显存/接受率及负结果，不含运行日志、请求文本/ID、私有端点、权重、SDK 或运行目录。
