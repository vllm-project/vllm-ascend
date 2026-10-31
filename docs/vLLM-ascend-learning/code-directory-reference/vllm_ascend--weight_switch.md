# vllm_ascend/weight_switch：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/weight_switch/`

Linear/输出投影完整权重切换和并行域相关生命周期。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/weight_switch/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Generic full-weight switching primitives. |
| [linear.py](../../../vllm_ascend/weight_switch/linear.py) | 线性投影 的接口/实现或辅助；所属职责：Linear/输出投影完整权重切换和并行域相关生命周期。 | 注释：Parallel-domain-agnostic full-weight switching for linear methods.；定义：`WeightSwitchGatherSpec`, `WeightSwitchRepeatSpec` |
| [o_proj.py](../../../vllm_ascend/weight_switch/o_proj.py) | 权重 的接口/实现或辅助；所属职责：Linear/输出投影完整权重切换和并行域相关生命周期。 | 注释：Single-target O-proj weight-switch lifecycle.；定义：`OProjWeightSwitchMixin` |
