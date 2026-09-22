# 诊断脚本（不是吞吐 benchmark）

所有命令都在授权远端、同一隔离工作目录、共同截止/锁控制下串行运行。
使用 tools/launch_bounded.py 包装，或置于已经统一管理的控制器下；不得另起竞争GPU任务。

1. 原生HF环境：`hf_reference.py --model MODEL_DIR --workdir RUN_DIR`
   保存三组原生raw logits和四类cache摘要，原生结果须在当前设备重建。
2. 原生HF环境：`norm_fixture.py --model MODEL_DIR --workdir RUN_DIR`
   从真实holdout的token embedding构造首层norm输入和原生输出。
3. 对应的官方0.19/SDK3.5.3环境：`norm_probe_v2.py --workdir RUN_DIR`
   使用官方pytest fixture同样的set_current_vllm_config上下文。
   比较forward_native、实际dispatch和重复调用；记录内部compile标志。
4. 原生HF环境：`long_graph_probe.py --model MODEL_DIR --workdir RUN_DIR`
   在direct-state E2E失败后运行；先复核无graph的原地状态写入，再分别运行
   keep_logits=False/True的128步capture，逐segment查完整token前缀和逻辑cache。

`*.private.pt` 包含少量checkpoint参数/输入激活，只留在隔离run中，
**不得加入Git、公开归档或上传**。本仓库gitignore已拒收pt/weights/wheel/SDK。
诊断holdout不是语义质量评测，也不能用其耗时作为整模型吞吐。
