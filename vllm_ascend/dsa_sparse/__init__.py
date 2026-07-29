"""DSA 模型家族稀疏卸载在 vLLM-Ascend 侧的核心实现包。

本包承载算法状态机、InputBatch 行语义投影、forward 数据契约、row-mode
eager/graph 共享运行时、HBM resident pool、worker-local DRAM hot store 以及
LIDU/KSC/SFA-Offload 与满块复制算子适配。对 vLLM 核心对象的 monkey patch 安装位于
``vllm_ascend.patch.dsa_sparse``，不属于本包的状态所有权。

依赖方向固定为：请求生命周期真源 -> ``DSAInputBatchState`` -> forward
数据契约 -> 物理镜像/动态 eager adapter -> layer hook 与设备算子。
``dsa_graph_gate`` 只读语义投影，不能创建资源或推进请求状态；所有
Ascend/DSA 专有实现都应留在 vLLM-Ascend，避免反向扩散到 vLLM 主仓。
"""
