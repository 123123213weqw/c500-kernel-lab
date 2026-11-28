# C500：原生 vLLM-MetaX / SpecPilot 同输入评测（实验）

这不是默认内核或服务配置。目标是把原生 vLLM 0.28 + vLLM-MetaX 的 BF16 AR、模型自带 MTP K1，以及作者独立 SpecPilot 的 DFlash2 树批处理分开测量。不会把切换引擎、改变输出长度或固定 Graph replay 峰值写成算子加速。

## 固定的比较方法

- 一张 C500；同一已核验的 Qwen3.8-27B BF16 目标。作者路线按照其原实现加载 W12 目标和 W8A8 草稿；这是系统比较，不是假装三方算子相同。
- 同一份预分词请求，在 native scheduler / SpecPilot 中各自最多 16 个在途请求。固定 20 个作者 dev 题（每套四题），原顺序，一半原长、一半为 512/1024/2048 token；只给长条目附无关背景。20 题末段并不恒定满载，不能写成 C16 长时间饱和。
- **性能档**：固定 160 输出/题，禁止提前 EOS；总计 3,200 个输出 token。闭环提交到交付末 token，包含 prefill、decode 和队列，不含加载、显式预热、最后的文本解码。报告输出 tok/s、墙钟、TTFT 和 TPOT；不伪称纯 decode。不允许用累计可见 token 或猜中草稿数当输出数。
- **质量档**：允许 EOS、上限 2048；统一报总输出量、完成时间、最终数学答案、HumanEval/MBPP。长文本需独立盲评。允许数字路径和措辞不同，这是实验筛选，不替代 Lab 的全词表 logits、四缓存、真实 128 步、跨请求状态和三次交错复测准入。没有过准入不能叫稳定收益或默认切换。
- 固定 B64 decode-only、B64 整段请求、实际 C64 在线混合负载分开记录。B64 或共享树预算 B64 都不是“64 并发服务”；捕获 OOM 不能填 0 或借用其他模型的成绩。

## 怎么运行

使用已在 MetaX C500 上核验的 vLLM 0.28、MACA 和只读模型；由 [隔离目录 / 公共 GPU 锁 / 硬截止控制器](../../docs/REPRODUCING.md) 启动。不要切换系统驱动、显卡频率或现有服务默认。须先核验模型版本、权重摘要和官方草稿。作者源码通过 [C500 SpecPilot](https://www.gitlink.org.cn/JunkaiShi/c500-specpilot) 单独取得，固定快照 `693c478b3b436a9676a210b38477e466067fb744`。工具入口：

| 入口 | 参数和结果 |
|---|---|
| `build_workload.py` | `--model`、`--items`（作者原 5 个 dev split）、`--out`（记录 token IDs 的 JSON） |
| `bench.py` | `--model --workload --out --mode ar` 或 `--mode mtp1`，可加 `--natural`；两种配置用独立进程 |
| `specpilot_fixed.py` | `--target --draft --pkg --workload --out`；`pkg` 指作者的 c500_specpilot 源码 |
| `summarize.py` | 输入同一归档目录（含 workload.json 和 ar/、mtp1/、specpilot/）；拒绝缺失/重复的题 ID、错的输入/输出数和非有限计时，不用上限替代实际输出数 |

真实 GPU 就绪检查、时限和授权由操作人提供；不上载远端路径、模型权重、私有配置或逐题原始输出。发布前运行 CPU 契约和源码审计。跑到三个固定样本不等于通过了上述发布门槛。

## 搜索顺序

1. **MTP K1 原厂路径**：额外一个草稿 token，无独立完整草稿模型。和 AR 用相同输入及输出，Graph 分桶按同一请求数搜索，单列 MTP 的缓存代价。
2. **动态 K**：本机配置里能找到上游动态档位，但参数存在不代表 MetaX 的 MTP/DFlash2 每档都保持图回放。先固定 K 校准；再验证实际提案长度、接受数、图复用和状态，不能拿配置反推。
3. **C500 矩阵和访存特性**：优先测 MetaX MCTLASS W8A8 的大投影在实际 token 行数 M=16/32/64 的收益，同时计激活量化与误差；不要按并发数直接挑 GEMM 的 M。平行探索更少的目标验证状态复制、稳定形状 Graph 分桶。MetaX [mcTriton 指南](https://developer.metax-tech.com/api/client/document/preview/367/C500_mcTritonUserGuide_CN.html)、[MXMACA Runtime 流与图](https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html) 和 [vLLM 动态 K](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/) 只是路线依据，不是本机已验证的性能。
