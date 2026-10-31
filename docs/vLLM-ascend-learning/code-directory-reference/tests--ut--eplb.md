# tests/ut/eplb：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **10 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/eplb/`

验证 eplb 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [adaptor/](../../../tests/ut/eplb/adaptor) | 验证 eplb/adaptor 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [core/](../../../tests/ut/eplb/core) | 验证 eplb/core 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/eplb/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_eplb_updator.py](../../../tests/ut/eplb/test_eplb_updator.py) | 验证 专家负载均衡 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_compute_and_set_moe_load_normal`, `test_compute_and_set_moe_load_multi_stage` |

## `tests/ut/eplb/adaptor/`

验证 eplb/adaptor 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/eplb/adaptor/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_vllm_adaptor.py](../../../tests/ut/eplb/adaptor/test_vllm_adaptor.py) | 验证 vllm adaptor 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init_fp16`, `test_init_fp16_with_parameter_accessor`, `test_init_w8a8`, `test_language_model_w8a8` |

## `tests/ut/eplb/core/`

验证 eplb/core 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [policy/](../../../tests/ut/eplb/core/policy) | 验证 eplb/core/policy 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/eplb/core/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [expert_map.json](../../../tests/ut/eplb/core/expert_map.json) | 专家 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`moe_layer_count`, `layer_list` |
| [test_eplb_device_transfer_loader.py](../../../tests/ut/eplb/core/test_eplb_device_transfer_loader.py) | 验证 传输、加载器、专家负载均衡 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_generate_task_and_state_flow`, `test_generate_task_uses_layer_weight_key_buffer`, `test_asyn_transfer_and_update`, `test_set_log2phy_map` |
| [test_eplb_utils.py](../../../tests/ut/eplb/core/test_eplb_utils.py) | 验证 辅助函数、专家负载均衡 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_init_eplb_config_with_eplb`, `test_generate_global_placement_matches_vllm_physical_layout`, `test_init_eplb_config_with_eplb_withmap`, `test_generate_log2phy_map_rotates_tail_tp_rank_with_tp_size` |

## `tests/ut/eplb/core/policy/`

验证 eplb/core/policy 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/eplb/core/policy/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_policy_factory.py](../../../tests/ut/eplb/core/policy/test_policy_factory.py) | 验证 policy factory 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_swift_balance_rebalance_experts`, `test_flashlb_rebalance_experts` |
