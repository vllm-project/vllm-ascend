# vllm_ascend/eplb：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **15 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/eplb/`

专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。

| 子目录 | 职责 |
| --- | --- |
| [adaptor/](../../../vllm_ascend/eplb/adaptor) | adaptor 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 |
| [core/](../../../vllm_ascend/eplb/core) | core 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/eplb/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [eplb_updator.py](../../../vllm_ascend/eplb/eplb_updator.py) | 专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`EplbUpdator` |

## `vllm_ascend/eplb/adaptor/`

adaptor 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/eplb/adaptor/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [vllm_adaptor.py](../../../vllm_ascend/eplb/adaptor/vllm_adaptor.py) | 专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`VllmEplbAdaptor` |

## `vllm_ascend/eplb/core/`

core 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。

| 子目录 | 职责 |
| --- | --- |
| [policy/](../../../vllm_ascend/eplb/core/policy) | policy 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/eplb/core/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [eplb_device_transfer_loader.py](../../../vllm_ascend/eplb/core/eplb_device_transfer_loader.py) | 传输、加载器、专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`ExpertWeightUpdateState`, `D2DExpertWeightLoader` |
| [eplb_utils.py](../../../vllm_ascend/eplb/core/eplb_utils.py) | 辅助函数、专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`expert_file_to_tensor`, `generate_global_placement`, `init_eplb_config` |
| [eplb_worker.py](../../../vllm_ascend/eplb/core/eplb_worker.py) | 设备执行、专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`EplbWorker`, `EplbProcess` |

## `vllm_ascend/eplb/core/policy/`

policy 子模块；所属范围：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/eplb/core/policy/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [policy_abstract.py](../../../vllm_ascend/eplb/core/policy/policy_abstract.py) | 专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`EplbPolicy` |
| [policy_default_eplb.py](../../../vllm_ascend/eplb/core/policy/policy_default_eplb.py) | 专家负载均衡 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`DynamicTable`, `DefaultEplb` |
| [policy_factory.py](../../../vllm_ascend/eplb/core/policy/policy_factory.py) | 按配置创建对应的 EPLB 策略对象。 | 定义：`PolicyFactory` |
| [policy_flashlb.py](../../../vllm_ascend/eplb/core/policy/policy_flashlb.py) | FlashLB 负载均衡策略，计算专家副本/放置相关方案。 | 定义：`FlashTree`, `FlashLBConfig`, `min_max_replica`, `max_delta_replica`, `percentage_replica` |
| [policy_random.py](../../../vllm_ascend/eplb/core/policy/policy_random.py) | 加载 的接口/实现或辅助；所属职责：专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 | 定义：`RandomLoadBalance` |
| [policy_swift_balancer.py](../../../vllm_ascend/eplb/core/policy/policy_swift_balancer.py) | SwiftBalance 专家负载均衡策略与动态表组织。 | 定义：`DynamicTable`, `SwiftBalanceEplb` |
