# tests/e2e/cases：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **214 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/cases/`

按模型或功能组织的测试场景。

| 子目录 | 职责 |
| --- | --- |
| [features/](../../../tests/e2e/cases/features) | 端到端 features 场景、运行脚本和配置。 |
| [models/](../../../tests/e2e/cases/models) | 端到端 模型 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [E2E_TEST_CASE_GUIDE.md](../../../tests/e2e/cases/E2E_TEST_CASE_GUIDE.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：E2E_TEST_CASE_GUIDE。 | 标题：Periodic E2E Test Case Guide |

## `tests/e2e/cases/features/`

端到端 features 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [kv_pool/](../../../tests/e2e/cases/features/kv_pool) | 端到端 池管理、KV 缓存 场景、运行脚本和配置。 |
| [openai_api_compatibility/](../../../tests/e2e/cases/features/openai_api_compatibility) | 端到端 openai_api_compatibility 场景、运行脚本和配置。 |
| [structured_output/](../../../tests/e2e/cases/features/structured_output) | 端到端 输出 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_qwen3_grid_exceed_65536.py](../../../tests/e2e/cases/features/test_qwen3_grid_exceed_65536.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_qwen3_large_mixed_batch_exceeding_triton_grid_limit` |

## `tests/e2e/cases/features/kv_pool/`

端到端 池管理、KV 缓存 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [DSv4-Flash-Memcache-PD-layerwise.yaml](../../../tests/e2e/cases/features/kv_pool/DSv4-Flash-Memcache-PD-layerwise.yaml) | 缓存 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-Flash-Memcache-PD-acc.yaml](../../../tests/e2e/cases/features/kv_pool/DeepSeek-V4-Flash-Memcache-PD-acc.yaml) | 缓存 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml](../../../tests/e2e/cases/features/kv_pool/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml) | Mooncake 传输/存储、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3-Memcache-PD.yaml](../../../tests/e2e/cases/features/kv_pool/GLM-5.2-W8A8C8-A3-Memcache-PD.yaml) | 缓存、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-397B-Memcache-perf.yaml](../../../tests/e2e/cases/features/kv_pool/Qwen3.5-397B-Memcache-perf.yaml) | 缓存、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [test_kvpp_dspark.py](../../../tests/e2e/cases/features/kv_pool/test_kvpp_dspark.py) | 验证 DSpark 草稿、分层 KV 放置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_glm_dspark_kvpp_outputs` |
| [test_kvpp_pool.py](../../../tests/e2e/cases/features/kv_pool/test_kvpp_pool.py) | 验证 分层 KV 放置、池管理 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_kvpp_memcache_reload` |
| [test_qwen3_30b_acc.py](../../../tests/e2e/cases/features/kv_pool/test_qwen3_30b_acc.py) | 验证 Qwen 模型 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_models` |

## `tests/e2e/cases/features/openai_api_compatibility/`

端到端 openai_api_compatibility 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [tests/](../../../tests/e2e/cases/features/openai_api_compatibility/tests) | 端到端 tests 场景、运行脚本和配置。 |
| [utility/](../../../tests/e2e/cases/features/openai_api_compatibility/utility) | 端到端 utility 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/cases/features/openai_api_compatibility/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/e2e/cases/features/openai_api_compatibility/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`api_client`, `pytest_addoption` |

## `tests/e2e/cases/features/openai_api_compatibility/tests/`

端到端 tests 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_chat_template_kwargs.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_chat_template_kwargs.py) | 验证 chat template kwargs 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_chat_template_kwargs_accepts_supported_values`, `test_chat_template_kwargs_accepts_streaming_request`, `test_chat_template_kwargs_rejects_invalid_top_level_types`, `test_chat_template_kwargs_accepts_or_rejects_boundary_objects` |
| [test_content.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_content.py) | 验证 content 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_content_accepts_supported_shapes`, `test_content_accepts_streaming_request`, `test_content_accepts_or_rejects_implementation_dependent_boundaries`, `test_content_rejects_invalid_shapes` |
| [test_context_length.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_context_length.py) | 验证 context length 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_context_length_within_limit`, `test_context_length_exceed_limit_returns_validation_error` |
| [test_frequency_penalty.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_frequency_penalty.py) | 验证 采样惩罚 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_frequency_penalty_accepts_supported_range`, `test_frequency_penalty_accepts_streaming_request`, `test_frequency_penalty_rejects_invalid_values`, `test_frequency_penalty_accepts_coerced_or_default_values` |
| [test_logprobs.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_logprobs.py) | 验证 对数概率 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_logprobs_accepts_supported_top_logprobs`, `test_logprobs_accepts_streaming_request`, `test_logprobs_rejects_top_logprobs_above_limit`, `test_logprobs_combines_with_logit_bias` |
| [test_max_completion_tokens.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_max_completion_tokens.py) | 验证 max completion tokens 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_max_completion_tokens_accepts_positive_integers`, `test_max_completion_tokens_accepts_streaming_request`, `test_max_completion_tokens_rejects_invalid_values`, `test_max_completion_tokens_accepts_coerced_or_default_values` |
| [test_max_tokens.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_max_tokens.py) | 验证 max tokens 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_max_tokens_accepts_positive_integers`, `test_max_tokens_accepts_streaming_request`, `test_max_tokens_rejects_invalid_values`, `test_max_tokens_accepts_coerced_values` |
| [test_n.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_n.py) | 验证 n 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_n_accepts_supported_values`, `test_n_omitted_uses_default_single_choice`, `test_n_rejects_invalid_values`, `test_n_accepts_coerced_string_integer` |
| [test_presence_penalty.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_presence_penalty.py) | 验证 采样惩罚 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_presence_penalty_accepts_supported_range`, `test_presence_penalty_accepts_streaming_request`, `test_presence_penalty_rejects_invalid_values`, `test_presence_penalty_accepts_coerced_or_default_values` |
| [test_repetition_penalty.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_repetition_penalty.py) | 验证 采样惩罚 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_repetition_penalty_accepts_supported_values`, `test_repetition_penalty_accepts_streaming_request`, `test_repetition_penalty_rejects_invalid_values`, `test_repetition_penalty_accepts_vllm_compatible_values` |
| [test_request_id.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_request_id.py) | 验证 请求 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_request_id_is_propagated_under_concurrency` |
| [test_role.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_role.py) | 验证 role 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_role_accepts_supported_message_sequences`, `test_role_accepts_streaming_multi_turn_request`, `test_role_rejects_invalid_role_values`, `test_role_accepts_or_rejects_implementation_specific_values` |
| [test_stop.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_stop.py) | 验证 stop 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_stop_accepts_supported_shapes`, `test_stop_accepts_streaming_request`, `test_stop_rejects_invalid_values`, `test_stop_accepts_default_or_large_values` |
| [test_temperature.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_temperature.py) | 验证 temperature 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_temperature_accepts_supported_range`, `test_temperature_accepts_streaming_request`, `test_temperature_rejects_invalid_values`, `test_temperature_accepts_vllm_compatible_values` |
| [test_think_tag.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_think_tag.py) | 验证 think tag 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_chat_think_tag_output_when_enabled`, `test_completions_do_not_emit_think_tag` |
| [test_tool_choice.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_tool_choice.py) | 验证 tool choice 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_tool_choice_accepts_supported_values`, `test_tool_choice_accepts_streaming_request`, `test_tool_choice_rejects_invalid_values`, `test_tool_choice_none_uses_default_behavior` |
| [test_top_k.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_top_k.py) | 验证 TopK 选择 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_top_k_accepts_representative_values`, `test_top_k_accepts_streaming_request`, `test_top_k_rejects_invalid_values`, `test_top_k_accepts_vllm_compatible_values` |
| [test_top_p.py](../../../tests/e2e/cases/features/openai_api_compatibility/tests/test_top_p.py) | 验证 TopP 选择 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_top_p_accepts_representative_values`, `test_top_p_accepts_streaming_request`, `test_top_p_rejects_invalid_values`, `test_top_p_accepts_coerced_or_default_values` |

## `tests/e2e/cases/features/openai_api_compatibility/utility/`

端到端 utility 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/cases/features/openai_api_compatibility/utility/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [assertion.py](../../../tests/e2e/cases/features/openai_api_compatibility/utility/assertion.py) | assertion 的测试辅助、测试入口或共享数据结构。 | 定义：`Check`, `assert_status_code_200`, `assert_finish_reason_valid`, `assert_chat_completion_success` |
| [completion_request.py](../../../tests/e2e/cases/features/openai_api_compatibility/utility/completion_request.py) | 请求 的测试辅助、测试入口或共享数据结构。 | 定义：`chat_request_body`, `send_chat_request`, `send_completion_request` |
| [http_client.py](../../../tests/e2e/cases/features/openai_api_compatibility/utility/http_client.py) | http client 的测试辅助、测试入口或共享数据结构。 | 定义：`HTTPClient` |

## `tests/e2e/cases/features/structured_output/`

端到端 输出 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../tests/e2e/cases/features/structured_output/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：Structured Output E2E Tests |
| [__init__.py](../../../tests/e2e/cases/features/structured_output/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [api.py](../../../tests/e2e/cases/features/structured_output/api.py) | api 的测试辅助、测试入口或共享数据结构。 | 定义：`structured_output_request_kwargs`, `create_structured_chat_completion`, `create_plain_chat_completion` |
| [assertions.py](../../../tests/e2e/cases/features/structured_output/assertions.py) | assertions 的测试辅助、测试入口或共享数据结构。 | 定义：`assert_structured_output`, `assert_plain_output` |
| [cases.py](../../../tests/e2e/cases/features/structured_output/cases.py) | cases 的测试辅助、测试入口或共享数据结构。 | 定义：`StructuredOutputCase` |
| [conftest.py](../../../tests/e2e/cases/features/structured_output/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`offline_runner`, `openai_server`, `openai_client` |
| [test_offline_api.py](../../../tests/e2e/cases/features/structured_output/test_offline_api.py) | 验证 offline api 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_offline_structured_output`, `test_offline_switches_constraints_without_state_leak` |
| [test_openai_api.py](../../../tests/e2e/cases/features/structured_output/test_openai_api.py) | 验证 openai api 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_openai_structured_output`, `test_openai_streaming_structured_output`, `test_openai_mixed_concurrent_structured_outputs`, `test_invalid_request_does_not_break_following_requests` |

## `tests/e2e/cases/models/`

端到端 模型 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [configs/](../../../tests/e2e/cases/models/configs) | 端到端 配置 场景、运行脚本和配置。 |

## `tests/e2e/cases/models/configs/`

端到端 配置 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [DeepSeek/](../../../tests/e2e/cases/models/configs/DeepSeek) | 端到端 DeepSeek 场景、运行脚本和配置。 |
| [GLM/](../../../tests/e2e/cases/models/configs/GLM) | 端到端 GLM 模型 场景、运行脚本和配置。 |
| [Gemma/](../../../tests/e2e/cases/models/configs/Gemma) | 端到端 Gemma 模型 场景、运行脚本和配置。 |
| [Hunyuan/](../../../tests/e2e/cases/models/configs/Hunyuan) | 端到端 Hunyuan 场景、运行脚本和配置。 |
| [Kimi/](../../../tests/e2e/cases/models/configs/Kimi) | 端到端 Kimi 模型 场景、运行脚本和配置。 |
| [MiniMax/](../../../tests/e2e/cases/models/configs/MiniMax) | 端到端 MiniMax 场景、运行脚本和配置。 |
| [Qwen/](../../../tests/e2e/cases/models/configs/Qwen) | 端到端 Qwen 模型 场景、运行脚本和配置。 |

## `tests/e2e/cases/models/configs/DeepSeek/`

端到端 DeepSeek 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [DS-V4-flash-w8a8-PD-prefix-weekly.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DS-V4-flash-w8a8-PD-prefix-weekly.yaml) | DS-V4-flash-w8a8-PD-prefix-weekly 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-R1-0528-W8A8.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-R1-0528-W8A8.yaml) | DeepSeek-R1-0528-W8A8 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `_benchmarks_perf`, `test_cases` |
| [DeepSeek-R1-W8A8-EPLB.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-R1-W8A8-EPLB.yaml) | 专家负载均衡 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V3.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V3.yaml) | DeepSeek-V3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-Flash-W8A8-A3-Breakable.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Flash-W8A8-A3-Breakable.yaml) | DeepSeek-V4-Flash-W8A8-A3-Breakable 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [DeepSeek-V4-Flash-W8A8-A3-MRV2.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Flash-W8A8-A3-MRV2.yaml) | DeepSeek-V4-Flash-W8A8-A3-MRV2 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [DeepSeek-V4-Flash-W8A8-A3.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Flash-W8A8-A3.yaml) | DeepSeek-V4-Flash-W8A8-A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [DeepSeek-V4-Pro-w4a8-1M-PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Pro-w4a8-1M-PD.yaml) | DeepSeek-V4-Pro-w4a8-1M-PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-Pro-w4a8-PD-no-prefix.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Pro-w4a8-PD-no-prefix.yaml) | DeepSeek-V4-Pro-w4a8-PD-no-prefix 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-Pro-w4a8-prefix-cache-PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Pro-w4a8-prefix-cache-PD.yaml) | 缓存 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-Pro-w4a8-prefix-cache-PD_weekly.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Pro-w4a8-prefix-cache-PD_weekly.yaml) | 缓存 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-flash-w8a8-PD-no-prefix.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-flash-w8a8-PD-no-prefix.yaml) | DeepSeek-V4-flash-w8a8-PD-no-prefix 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4-flash-w8a8-PD-prefix.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-flash-w8a8-PD-prefix.yaml) | DeepSeek-V4-flash-w8a8-PD-prefix 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek-V4.1-Flash-w8a8-PD-prefix.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4.1-Flash-w8a8-PD-prefix.yaml) | DeepSeek-V4.1-Flash-w8a8-PD-prefix 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V3.1T_MTP1_128K_1K_PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V3.1T_MTP1_128K_1K_PD.yaml) | DeepSeek V3.1T MTP1 128K 1K PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V3.1T_MTP1_3_5K_1_5K_PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V3.1T_MTP1_3_5K_1_5K_PD.yaml) | DeepSeek V3.1T MTP1 3 5K 1 5K PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V3.1T_MTP1_PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V3.1T_MTP1_PD.yaml) | DeepSeek V3.1T MTP1 PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V3.1T_MTP3_PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V3.1T_MTP3_PD.yaml) | DeepSeek V3.1T MTP3 PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V3.1T_layerwise_PD.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V3.1T_layerwise_PD.yaml) | DeepSeek V3.1T layerwise PD 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [DeepSeek_V31_Terminus_w4a8_w8a8_A5.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V31_Terminus_w4a8_w8a8_A5.yaml) | DeepSeek V31 Terminus w4a8 w8a8 A5 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_server_cmd`, `_benchmarks_per2`, `test_cases` |
| [DeepSeek_V31_Terminus_w4a8_w8a8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V31_Terminus_w4a8_w8a8_A5_weekly.yaml) | DeepSeek V31 Terminus w4a8 w8a8 A5 weekly 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_server_cmd`, `_benchmarks_per2`, `test_cases` |
| [DeepSeek_V31_W4A4C8_A5.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V31_W4A4C8_A5.yaml) | DeepSeek V31 W4A4C8 A5 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [DeepSeek_V31_W4A4C8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V31_W4A4C8_A5_weekly.yaml) | DeepSeek V31 W4A4C8 A5 weekly 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [DeepSeek_V4_Flash_A5.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V4_Flash_A5.yaml) | DeepSeek V4 Flash A5 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_per1`, `_benchmarks_per2`, `test_cases` |
| [DeepSeek_V4_Flash_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/DeepSeek_V4_Flash_A5_weekly.yaml) | DeepSeek V4 Flash A5 weekly 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_per1`, `_benchmarks_per2`, `test_cases` |
| [Deepseek_R1_W8A8_A3.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/Deepseek_R1_W8A8_A3.yaml) | DeepSeek 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Deepseek_R1_W8A8_Reasoning_output_A3.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/Deepseek_R1_W8A8_Reasoning_output_A3.yaml) | DeepSeek 模型、输出 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [MTPX-DeepSeek-R1-0528-W8A8.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/MTPX-DeepSeek-R1-0528-W8A8.yaml) | MTPX-DeepSeek-R1-0528-W8A8 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_gsm8k`, `_benchmarks_aime`, `test_cases` |
| [Prefix-Cache-DeepSeek-R1-0528-W8A8.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/Prefix-Cache-DeepSeek-R1-0528-W8A8.yaml) | 缓存 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml](../../../tests/e2e/cases/models/configs/DeepSeek/acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml) | DeepSeek 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf`, `test_cases` |

## `tests/e2e/cases/models/configs/GLM/`

端到端 GLM 模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [GLM-4.7.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-4.7.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [GLM-5.1-W8A8-PrefillMC2.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8-PrefillMC2.yaml) | 上下文预填充、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.1-W8A8C8-128k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-128k-1k-90-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-3.5k-1.5k-0-20-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-3.5k-1.5k-0-20-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-3.5k-1.5k-0-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-3.5k-1.5k-0-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-64k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-64k-1k-90-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-A3_128k_90_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-A3_128k_90_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-A3_198k_function.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-A3_198k_function.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-A3_3.5k_1.5k_0_20.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-A3_3.5k_1.5k_0_20.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-A3_3.5k_1.5k_0_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-A3_3.5k_1.5k_0_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.1-W8A8C8-A3_64k_90_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.1-W8A8C8-A3_64k_90_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W4A8-DSpark-A3-V2.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8-DSpark-A3-V2.yaml) | DSpark 草稿、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.2-W4A8-DSpark-A3.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8-DSpark-A3.yaml) | DSpark 草稿、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.2-W4A8-MTP-A3-V2.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8-MTP-A3-V2.yaml) | 多 token 预测、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.2-W4A8-MTP-A3.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8-MTP-A3.yaml) | 多 token 预测、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.2-W4A8C8-1M-PD-DCP.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8C8-1M-PD-DCP.yaml) | Decode 上下文并行、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W4A8C8-SFA-DCP.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W4A8C8-SFA-DCP.yaml) | 稀疏注意力、Decode 上下文并行、GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [GLM-5.2-W8A8C8-128k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-128k-1k-90-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-3.5k-1.5k-0-20-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-3.5k-1.5k-0-20-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-3.5k-1.5k-0-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-3.5k-1.5k-0-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-64k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-64k-1k-90-50-PD.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3_128k_90_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_128k_90_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3_198k_function.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_198k_function.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3_3.5k_1.5k_0_20.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_3.5k_1.5k_0_20.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3_3.5k_1.5k_0_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_3.5k_1.5k_0_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.2-W8A8C8-A3_64k_90_50.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_64k_90_50.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM-5.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [GLM-5_1-W8A8_A3_weekly.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM-5_1-W8A8_A3_weekly.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `_benchmarks_bs8`, `_benchmarks_bs32` |
| [GLM5_1-W8A8-A2-dual-nodes.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1-W8A8-A2-dual-nodes.yaml) | GLM5 1-W8A8-A2-dual-nodes 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM5_1-W8A8-A3-dual-nodes.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1-W8A8-A3-dual-nodes.yaml) | GLM5 1-W8A8-A3-dual-nodes 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM5_1-W8A8-EP-external.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1-W8A8-EP-external.yaml) | GLM5 1-W8A8-EP-external 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM5_1-W8A8-EP.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1-W8A8-EP.yaml) | GLM5 1-W8A8-EP 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM5_1_W4A4_A5.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1_W4A4_A5.yaml) | GLM5 1 W4A4 A5 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [GLM5_1_W4A4_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_1_W4A4_A5_weekly.yaml) | GLM5 1 W4A4 A5 weekly 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [GLM5_2-W8A8-A3-dual-nodes.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_2-W8A8-A3-dual-nodes.yaml) | GLM5 2-W8A8-A3-dual-nodes 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM5_3-W8A8-A3-dual-nodes.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM5_3-W8A8-A3-dual-nodes.yaml) | GLM5 3-W8A8-A3-dual-nodes 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM_5_1_PD_in32k_bs16-0.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM_5_1_PD_in32k_bs16-0.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM_5_1_PD_in32k_bs20-90.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM_5_1_PD_in32k_bs20-90.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [GLM_5_1_PD_in64k_bs16-90.yaml](../../../tests/e2e/cases/models/configs/GLM/GLM_5_1_PD_in64k_bs16-90.yaml) | GLM 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |

## `tests/e2e/cases/models/configs/Gemma/`

端到端 Gemma 模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Gemma4-31B-Dense.yaml](../../../tests/e2e/cases/models/configs/Gemma/Gemma4-31B-Dense.yaml) | 稠密计算、Gemma 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_model`, `_envs`, `_server_cmd`, `_api_keyword_args`, `_benchmarks` |
| [Gemma4.yaml](../../../tests/e2e/cases/models/configs/Gemma/Gemma4.yaml) | Gemma 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_model`, `_envs`, `_server_cmd`, `_api_keyword_args`, `_benchmarks` |

## `tests/e2e/cases/models/configs/Hunyuan/`

端到端 Hunyuan 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Hy3-preview.yaml](../../../tests/e2e/cases/models/configs/Hunyuan/Hy3-preview.yaml) | Hy3-preview 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |

## `tests/e2e/cases/models/configs/Kimi/`

端到端 Kimi 模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Kimi-K2-Thinking.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2-Thinking.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Kimi-K2.5-W4A8-128k-1k-TPOT50.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.5-W4A8-128k-1k-TPOT50.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.5-W4A8-16k-1k-TPOT50.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.5-W4A8-16k-1k-TPOT50.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.5.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.5.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Kimi-K2.6-W4A8-128k-1k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-W4A8-128k-1k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.6-W4A8-254k-1k-TPOT20-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-W4A8-254k-1k-TPOT20-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.6-W4A8-3.5k-1.5k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-W4A8-3.5k-1.5k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.6-W4A8-64k-1k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-W4A8-64k-1k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.6-w4a8-A3.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-w4a8-A3.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Kimi-K2.6-w4a8-A3_weekly.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.6-w4a8-A3_weekly.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Kimi-K2.7-W4A8-128k-1k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.7-W4A8-128k-1k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.7-W4A8-254k-1k-TPOT20-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.7-W4A8-254k-1k-TPOT20-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.7-W4A8-3.5k-1.5k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.7-W4A8-3.5k-1.5k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.7-W4A8-64k-1k-TPOT50-PD.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.7-W4A8-64k-1k-TPOT50-PD.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Kimi-K2.7-w4a8-A3.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2.7-w4a8-A3.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Kimi-K2_5-W4A8-A2-dual-nodes.yaml](../../../tests/e2e/cases/models/configs/Kimi/Kimi-K2_5-W4A8-A2-dual-nodes.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [KimiK26_w4a4c8_A5.yaml](../../../tests/e2e/cases/models/configs/Kimi/KimiK26_w4a4c8_A5.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [KimiK26_w4a4c8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Kimi/KimiK26_w4a4c8_A5_weekly.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [kimi-k2.6_max_model_len.yaml](../../../tests/e2e/cases/models/configs/Kimi/kimi-k2.6_max_model_len.yaml) | 模型、Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf`, `test_cases` |
| [kimi25_w4a8_step_3_7.yaml](../../../tests/e2e/cases/models/configs/Kimi/kimi25_w4a8_step_3_7.yaml) | Kimi 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf`, `test_cases` |

## `tests/e2e/cases/models/configs/MiniMax/`

端到端 MiniMax 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [MiniMax-M2.5-W8A8-A3.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M2.5-W8A8-A3.yaml) | MiniMax-M2.5-W8A8-A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [MiniMax-M2.5-w8a8-QuaRot-A2.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M2.5-w8a8-QuaRot-A2.yaml) | MiniMax-M2.5-w8a8-QuaRot-A2 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [MiniMax-M2.5-w8a8-QuaRot-A3.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M2.5-w8a8-QuaRot-A3.yaml) | MiniMax-M2.5-w8a8-QuaRot-A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [MiniMax-M2.5-w8a8-QuaRot-step.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M2.5-w8a8-QuaRot-step.yaml) | MiniMax-M2.5-w8a8-QuaRot-step 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [MiniMax-M3-BF16-A3.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M3-BF16-A3.yaml) | MiniMax-M3-BF16-A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [MiniMax-M3-W8A8-A3.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-M3-W8A8-A3.yaml) | MiniMax-M3-W8A8-A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [MiniMax-PD-in32k-bs4-1.yaml](../../../tests/e2e/cases/models/configs/MiniMax/MiniMax-PD-in32k-bs4-1.yaml) | MiniMax-PD-in32k-bs4-1 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Minimax_M2.7_w8a8_A3.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_M2.7_w8a8_A3.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_3500`, `_benchmarks_prefix_64k_50`, `_benchmarks_prefix_128k_50` |
| [Minimax_m2.7_in128k_1k_prefix90_tpot50.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_m2.7_in128k_1k_prefix90_tpot50.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Minimax_m2.7_in198_prefix99.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_m2.7_in198_prefix99.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Minimax_m2.7_in3_5_tpot20.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_m2.7_in3_5_tpot20.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Minimax_m2.7_in3_5_tpot50.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_m2.7_in3_5_tpot50.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Minimax_m2.7_in64k_1k_prefix90_tpot50.yaml](../../../tests/e2e/cases/models/configs/MiniMax/Minimax_m2.7_in64k_1k_prefix90_tpot50.yaml) | MiniMax 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |

## `tests/e2e/cases/models/configs/Qwen/`

端到端 Qwen 模型 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Prefix-Cache-Qwen3-32B-Int8.yaml](../../../tests/e2e/cases/models/configs/Qwen/Prefix-Cache-Qwen3-32B-Int8.yaml) | 缓存、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [QWEN3_235B_PD_3_5K_1_5k.yaml](../../../tests/e2e/cases/models/configs/Qwen/QWEN3_235B_PD_3_5K_1_5k.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen-3.5-397B-A17B-W8A8-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen-3.5-397B-A17B-W8A8-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen2.5-VL-7B-Instruct-EPD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen2.5-VL-7B-Instruct-EPD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-14B-w8a8sc-310p.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-14B-w8a8sc-310p.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-235B-A22B-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-235B-A22B-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3-235B-A22B-W8A8_piecewise_fullgraph_A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-235B-A22B-W8A8_piecewise_fullgraph_A3.yaml) | 设备图、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3-30B-A3B-BF16-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-30B-A3B-BF16-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-30B-A3B-W4A8-llm-compressor.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-30B-A3B-W4A8-llm-compressor.yaml) | 长程压缩、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-30B-A3B-W8A8-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-30B-A3B-W8A8-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-30B-A3B-W8A8.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-30B-A3B-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-30B-QuaRot-eagle3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-30B-QuaRot-eagle3.yaml) | Eagle 投机、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-32B-Int8-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B-Int8-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3-32B-Int8.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B-Int8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3-32B-QuaRot-eagle3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B-QuaRot-eagle3.yaml) | Eagle 投机、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-32B-W8A8C8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B-W8A8C8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3-32B-w8a8sc-310p.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B-w8a8sc-310p.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-32B.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-32B.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-8B-w8a8sc-310p.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-8B-w8a8sc-310p.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3-VL-235B-A22B-Instruct-W8A8.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-VL-235B-A22B-Instruct-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3-VL-235B-disagg-pd.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-VL-235B-disagg-pd.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3-VL-32B-Instruct-W8A8.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3-VL-32B-Instruct-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3.5-122B-A10B-W8A8-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-W8A8-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `test_cases` |
| [Qwen3.5-122B-A10B-W8A8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-W8A8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [Qwen3.5-122B-A10B-W8A8-A3_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-W8A8-A3_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmark_acc`, `_benchmark_TPOT50_3_5k_1_5K`, `_benchmark_TPOT50_16K_1K` |
| [Qwen3.5-122B-A10B-w4a8-128k-1k-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-w4a8-128k-1k-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-122B-A10B-w4a8-16k-1k-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-w4a8-16k-1k-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-122B-A10B-w4a8-984k-1k-0-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-w4a8-984k-1k-0-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-122B-A10B-w4a8-A3-accuracy.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-w4a8-A3-accuracy.yaml) | 精度、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-122B-A10B-w4a8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-122B-A10B-w4a8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-27B-w8a8-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-27B-w8a8-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-27B-w8a8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-27B-w8a8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-27B-w8a8-A3_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-27B-w8a8-A3_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `test_cases` |
| [Qwen3.5-2b-310p.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-2b-310p.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `_benchmarks_perf`, `test_cases` |
| [Qwen3.5-397B-A17B-W8A8-mtp-A3-perf_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-W8A8-mtp-A3-perf_weekly.yaml) | Qwen 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml) | Qwen 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-397B-A17B-W8A8-mtp-A3_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-W8A8-mtp-A3_weekly.yaml) | Qwen 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks`, `_benchmarks_bs144`, `_benchmarks_bs36` |
| [Qwen3.5-397B-A17B-w4a8-mtp-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-w4a8-mtp-A2.yaml) | Qwen 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-397B-A17B-w8a8-3.5k-1.5k-0-50-single.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-w8a8-3.5k-1.5k-0-50-single.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-397B-A17B-w8a8-64k-1k-90-50-single.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-w8a8-64k-1k-90-50-single.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.5-397B-w4a8_128k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-w4a8_128k-1k-90-50-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-397B-w4a8_3.5k-1.5k-0-50-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-w4a8_3.5k-1.5k-0-50-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-397B-w4a8_64k-1k-90-50-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-w4a8_64k-1k-90-50-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.5-397B-w4a8_986k-function-PD.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-w4a8_986k-function-PD.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_name`, `model`, `num_nodes`, `npu_per_node`, `dp_load_balancing` |
| [Qwen3.6-27B-W8A8-310P-300I-DUO.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6-27B-W8A8-310P-300I-DUO.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.6-35B-A3B-w4a8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6-35B-A3B-w4a8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.6-35B-A3B-w8a8-310p-MTP.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6-35B-A3B-w8a8-310p-MTP.yaml) | Qwen 模型、多 token 预测 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.6-35B-A3B-w8a8-A3-accuracy.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6-35B-A3B-w8a8-A3-accuracy.yaml) | 精度、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.6-35B-W8A8-310P-300I-DUO.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6-35B-W8A8-310P-300I-DUO.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`test_cases` |
| [Qwen3.6_35B_27B_A3_acceptance_rate.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.6_35B_27B_A3_acceptance_rate.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf`, `test_cases` |
| [Qwen3.8-27B-w8a8-A2.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.8-27B-w8a8-A2.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `test_cases` |
| [Qwen3.8-27B-w8a8-A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3.8-27B-w8a8-A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `test_cases` |
| [Qwen3VL-8B-310p.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3VL-8B-310p.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `_benchmarks_perf`, `test_cases` |
| [Qwen3_235b_w8a8_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_235b_w8a8_A5.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_gpqa`, `test_cases` |
| [Qwen3_235b_w8a8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_235b_w8a8_A5_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_gpqa`, `test_cases` |
| [Qwen3_32b_w8a8_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_32b_w8a8_A5.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `test_cases` |
| [Qwen3_32b_w8a8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_32b_w8a8_A5_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `test_cases` |
| [Qwen3_5_35B_A3B_FP8_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_5_35B_A3B_FP8_A5.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf1`, `test_cases` |
| [Qwen3_5_35B_A3B_FP8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_5_35B_A3B_FP8_A5_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf1`, `test_cases` |
| [Qwen3_5_397B_A17B_w8a8_mxfp8_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_5_397B_A17B_w8a8_mxfp8_A5.yaml) | MXFP 数值格式、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_per1`, `_benchmarks_per3`, `test_cases` |
| [Qwen3_5_397B_A17B_w8a8_mxfp8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_5_397B_A17B_w8a8_mxfp8_A5_weekly.yaml) | MXFP 数值格式、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_per1`, `_benchmarks_per3`, `test_cases` |
| [Qwen3_6_27B_W8A8_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_6_27B_W8A8_A5.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [Qwen3_6_27B_W8A8_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_6_27B_W8A8_A5_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `test_cases` |
| [Qwen3_6_27b_A5.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_6_27b_A5.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf1`, `test_cases` |
| [Qwen3_6_27b_A5_weekly.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwen3_6_27b_A5_weekly.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_perf1`, `test_cases` |
| [Qwq_32b_A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/Qwq_32b_A3.yaml) | Qwq 32b A3 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_gsm8k_lite`, `_benchmarks_perf1`, `test_cases` |
| [qwen_3_6_27b_w8a8_A3.yaml](../../../tests/e2e/cases/models/configs/Qwen/qwen_3_6_27b_w8a8_A3.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`_envs`, `_server_cmd`, `_benchmarks_acc`, `_benchmarks_3500_50`, `_benchmarks_3500_20` |
