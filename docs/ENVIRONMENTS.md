# 环境边界

## 已通过的原生/自研路径

- Python3.10，Torch2.8.0+metax3.3.0.2
- Triton3.0.0+metax3.3.0.2，Transformers5.3.0
- 具体其他包版本保存在冻结配置内。
- Qwen3.5-27B BF16，全VL模型驻留但只走文本；原生FP32 recurrent state。
- 不量化、不CPU offload、不共享请求prefix，不改变原生attention/RoPE/MLP算法。
- FP32逐步算术使用显式mul_rn/sub_rn/add_rn；不能把仅设置disable-fusion当作已验证的逐值相等证明。

## 官方重建对照路径（尚未通过严格数值准入）

- vLLM0.19.0，MetaX插件0.19.0+g933e92.d20260429.maca3.5.3.20.torch2.8
- 用户态SDK3.5.3.20仅解包到隔离目录，不安装系统驱动/包。
- Torch2.8.0+metax3.5.3.9，Triton3.0.0+metax3.5.3.9
- mcoplib0.4.4、FLA0.4.0、FA2 2.6.3，均为对应MetaX构建。
- cu-bridge官方3.5.3源：commit1608d1dc09325a7327f80d7f41da004ce4c4b8d6，远端C++构建。
- vLLM使用empty target的上游source wheel +官方MetaX插件，不安装NVIDIA kernel wheel。
- 不是未修改的官方Docker镜像，也不是最新0.28的成绩。
- 官方文本加载省略视觉tower；不可把其驻留显存差异全部归因于内核优化。

引擎 `enforce_eager=True` 只说明引擎级graph/编译策略；安装版
GemmaRMSNorm.forward_cuda还有独立的torch.compile路径。不能把eager配置
自动解释为“所有算子都未编译”。本轮独立算子探针结果另记，不外推为全模型根因。

上述环境包含不同数值实现：没有逐值数值准入时，不给官方与自研端到端TPS排名。
