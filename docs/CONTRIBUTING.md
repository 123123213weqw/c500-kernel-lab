# 贡献方向

- 最小化 MetaX Triton 的算术收缩复现：显式 mul_rn/add_rn 与仅禁用融合的差异，附版本和输入种子。
- GDN dense-layout/aliasing oracle、实际 dispatch 计数、graph cache重置回归。
- 官方 eager vs compiled 数值分歧定位：相同输入、权重映射、首个分歧层/算子。
- 真实请求 E2E 的可复现脚本和失败矩阵；不要只提交合成kernel排行榜。

新增kernel必须提供：形状/stride/dtype前提、native fallback、至少30小型oracle、
原生整模型full-vocab/caches/IDs门槛、n≥3配对耗时、显存和失败样本。
启用比对前不得改基线的精度、长度、EOS规则或token计数。

不提交模型权重、SDK/二进制wheel、远端地址、凭据、个人路径、原始私有数据。
不修改旧归档；新结果放入独立run，经过去标识后只收摘要。
当前未经验证的优化只进 experimental，不把微内核正确误写成整模型提速。
