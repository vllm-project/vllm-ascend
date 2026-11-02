# MoE / MegaMoE 代码目录导读

MoE 是模型内部的混合专家计算模块；MegaMoE 是执行 MoE 的方案之一，涉及专家计算和通信，通常不单独占进程。

```text
CPU：Model → MoE 模块 → 选择并调用 MegaMoE → 提交计算与通信任务
                                                    ↓
NPU：                                 token 分发 → 专家计算 → 汇总
```

本仓代码目录及阅读顺序：

| 顺序 | 目录 | 看什么 |
| --- | --- | --- |
| 1 | [vllm_ascend/models/](../../vllm_ascend/models/) | 模型哪一层调用 MoE |
| 2 | [vllm_ascend/ops/fused_moe/](../../vllm_ascend/ops/fused_moe/) | MoE 调用组织、MegaMoE 参数及能力适配 |
| 3 | [vllm_ascend/distributed/](../../vllm_ascend/distributed/) | 对应方案使用的通信路径 |
| 4 | [csrc/](../../csrc/) | 本仓自定义算子，是否走这里需沿调用核对 |

具体入口：[vllm_ascend/ops/fused_moe/fused_moe.py](../../vllm_ascend/ops/fused_moe/fused_moe.py)、[vllm_ascend/ops/fused_moe/mega_moe.py](../../vllm_ascend/ops/fused_moe/mega_moe.py)、[vllm_ascend/ops/fused_moe/moe_utils.py](../../vllm_ascend/ops/fused_moe/moe_utils.py)。MegaMoE 路径还会加载外部 CANN 算子，不能认为完整 kernel 实现都在本仓 `csrc/` 中。

返回 [vLLM Ascend 框架图](vllm-ascend-framework.md)。
