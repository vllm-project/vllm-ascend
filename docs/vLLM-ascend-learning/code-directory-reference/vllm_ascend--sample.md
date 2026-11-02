# vllm_ascend/sample：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/sample/`

Logits 处理、TopK/TopP、采样和投机验证恢复。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/sample/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [penalties.py](../../../vllm_ascend/sample/penalties.py) | 采样惩罚 的接口/实现或辅助；所属职责：Logits 处理、TopK/TopP、采样和投机验证恢复。 | 定义：`apply_all_penalties` |
| [rejection_sampler.py](../../../vllm_ascend/sample/rejection_sampler.py) | 投机候选验证、接受前缀、拒绝恢复和对应采样约束。 | 定义：`AscendRejectionSampler`, `apply_sampling_constraints`, `rejection_sample`, `expand_batch_to_tokens` |
| [sampler.py](../../../vllm_ascend/sample/sampler.py) | NPU token 采样、惩罚、TopK/TopP 和 logprobs 处理。 | 定义：`AscendSampler`, `AscendTopKTopPSampler`, `random_sample` |
| [topk_map.py](../../../vllm_ascend/sample/topk_map.py) | TopK 选择 的接口/实现或辅助；所属职责：Logits 处理、TopK/TopP、采样和投机验证恢复。 | 定义：`CompactDist` |
