# tests/e2e/common：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **23 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/common/`

端到端场景共用的部署与断言辅助。

| 子目录 | 职责 |
| --- | --- |
| [kv_pool/](../../../tests/e2e/common/kv_pool) | 端到端 池管理、KV 缓存 场景、运行脚本和配置。 |
| [multi_node/](../../../tests/e2e/common/multi_node) | 端到端 multi_node 场景、运行脚本和配置。 |
| [single_node/](../../../tests/e2e/common/single_node) | 端到端 single_node 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [kvpp.py](../../../tests/e2e/common/kvpp.py) | 分层 KV 放置 的测试辅助、测试入口或共享数据结构。 | 注释：Inputs and serving arguments shared by KVPP integration tests.；定义：`server_args`, `complete`, `output_texts` |

## `tests/e2e/common/kv_pool/`

端到端 池管理、KV 缓存 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/common/kv_pool/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：Shared KV pool support for E2E test frameworks. |
| [config.py](../../../tests/e2e/common/kv_pool/config.py) | 配置 的测试辅助、测试入口或共享数据结构。 | 定义：`KVPoolConfig`, `MooncakeKVPoolConfig`, `parse_kv_pool_config`, `validate_kv_pool_config` |

## `tests/e2e/common/multi_node/`

端到端 multi_node 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [external_dp/](../../../tests/e2e/common/multi_node/external_dp) | 端到端 external_dp 场景、运行脚本和配置。 |
| [internal_dp/](../../../tests/e2e/common/multi_node/internal_dp) | 端到端 internal_dp 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/common/multi_node/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [benchmark_results.py](../../../tests/e2e/common/multi_node/benchmark_results.py) | 性能测量 的测试辅助、测试入口或共享数据结构。 | 定义：`extract_hardware`, `get_vllm_version`, `task_passed` |
| [dp_mode.py](../../../tests/e2e/common/multi_node/dp_mode.py) | dp mode 的测试辅助、测试入口或共享数据结构。 | 注释：Resolve the DP launch mode declared by a multi-node test case.；定义：`resolve_config_path`, `resolve_dp_load_balancing` |
| [lws.yaml.jinja2](../../../tests/e2e/common/multi_node/lws.yaml.jinja2) | lws.yaml.jinja2 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [run.sh](../../../tests/e2e/common/multi_node/run.sh) | run 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 函数：`print_section`, `print_failure`, `print_success` |
| [test_multi_node.py](../../../tests/e2e/common/multi_node/test_multi_node.py) | 验证 multi node 的测试；具体断言场景由右栏测试符号定位。 | 注释：Common pytest entry for internal and external DP multi-node cases.；定义：`test_multi_node` |
| [utils.py](../../../tests/e2e/common/multi_node/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`ProxyServer`, `temp_env`, `setup_logger`, `load_yaml_mapping` |

## `tests/e2e/common/multi_node/external_dp/`

端到端 external_dp 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/common/multi_node/external_dp/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：External DP E2E test helpers. |
| [external_dp_config.py](../../../tests/e2e/common/multi_node/external_dp/external_dp_config.py) | 配置 的测试辅助、测试入口或共享数据结构。 | 定义：`RoutingConfig`, `NodeInfo`, `replace_cluster_placeholders`, `resolve_current_node_index` |
| [runtime.py](../../../tests/e2e/common/multi_node/external_dp/runtime.py) | runtime 的测试辅助、测试入口或共享数据结构。 | 定义：`ServerCommand`, `ExternalDPKVPoolManager`, `create_kv_pool_manager`, `build_all_server_commands`, `build_dist_envs` |
| [test_external_dp.py](../../../tests/e2e/common/multi_node/external_dp/test_external_dp.py) | 验证 external dp 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_external_dp` |
| [utils.py](../../../tests/e2e/common/multi_node/external_dp/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`format_server_cmd`, `start_logged_process`, `terminate_process_tree` |

## `tests/e2e/common/multi_node/internal_dp/`

端到端 internal_dp 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/common/multi_node/internal_dp/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [multi_node_config.py](../../../tests/e2e/common/multi_node/internal_dp/multi_node_config.py) | 配置 的测试辅助、测试入口或共享数据结构。 | 定义：`NodeInfo`, `DisaggregatedPrefillCfg` |
| [test_multi_node.py](../../../tests/e2e/common/multi_node/internal_dp/test_multi_node.py) | 验证 multi node 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_multi_node` |
| [utils.py](../../../tests/e2e/common/multi_node/internal_dp/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 导入：`os`, `tests.e2e.common.multi_node.utils` |

## `tests/e2e/common/single_node/`

端到端 single_node 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/common/single_node/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [kv_pool_runtime.py](../../../tests/e2e/common/single_node/kv_pool_runtime.py) | 池管理、KV 缓存 的测试辅助、测试入口或共享数据结构。 | 定义：`SingleNodeKVPoolManager`, `SingleNodeMooncakeManager`, `create_single_node_kv_pool_manager` |
| [single_node_config.py](../../../tests/e2e/common/single_node/single_node_config.py) | 配置 的测试辅助、测试入口或共享数据结构。 | 定义：`SingleNodeConfig`, `SingleNodeConfigLoader` |
| [test_single_node.py](../../../tests/e2e/common/single_node/test_single_node.py) | 验证 single node 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_single_node` |
