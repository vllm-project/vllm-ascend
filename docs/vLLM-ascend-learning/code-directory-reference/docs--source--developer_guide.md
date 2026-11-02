# docs/source/developer_guide：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **38 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `docs/source/developer_guide/`

开发架构、贡献规范、测试评估和调优排错说明。

| 子目录 | 职责 |
| --- | --- |
| [Design_Documents/](../../source/developer_guide/Design_Documents) | 执行、算子、缓存和并行机制的专题设计说明。 |
| [contribution/](../../source/developer_guide/contribution) | 父主题的 contribution 文档、模板或站点资源。 |
| [evaluation/](../../source/developer_guide/evaluation) | 父主题的 evaluation 文档、模板或站点资源。 |
| [performance_and_debug/](../../source/developer_guide/performance_and_debug) | 父主题的 performance_and_debug 文档、模板或站点资源。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [deepseek_v41_qli_candidate.md](../../source/developer_guide/deepseek_v41_qli_candidate.md) | 说明 DeepSeek V4.1 QLI V2 and candidate integration。 | 标题：DeepSeek V4.1 QLI V2 and candidate integration |
| [index.md](../../source/developer_guide/index.md) | 说明 Developer Guide。 | 标题：Developer Guide |

## `docs/source/developer_guide/Design_Documents/`

执行、算子、缓存和并行机制的专题设计说明。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ACL_Graph.md](../../source/developer_guide/Design_Documents/ACL_Graph.md) | 说明 ACL Graph。 | 标题：ACL Graph |
| [KV_Cache_Pool_Guide.md](../../source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md) | 说明 KV Cache Pool。 | 标题：KV Cache Pool |
| [ModelRunner_prepare_inputs.md](../../source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md) | 说明 Prepare inputs for model forwarding。 | 标题：Prepare inputs for model forwarding |
| [add_custom_aclnn_op.md](../../source/developer_guide/Design_Documents/add_custom_aclnn_op.md) | 说明 Adding a custom aclnn operation。 | 标题：Adding a custom aclnn operation |
| [balance_schedule_refactor.md](../../source/developer_guide/Design_Documents/balance_schedule_refactor.md) | 说明 Balance Schedule Refactor。 | 标题：Balance Schedule Refactor |
| [context_parallel.md](../../source/developer_guide/Design_Documents/context_parallel.md) | 说明 Context Parallel。 | 标题：Context Parallel |
| [cpu_binding.md](../../source/developer_guide/Design_Documents/cpu_binding.md) | 说明 CPU Binding。 | 标题：CPU Binding |
| [disaggregated_prefill.md](../../source/developer_guide/Design_Documents/disaggregated_prefill.md) | 说明 Disaggregated-prefill。 | 标题：Disaggregated-prefill |
| [dynamic_chunked_pipeline_parallel.md](../../source/developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md) | 说明 Dynamic Chunked Pipeline Parallel (CPP)。 | 标题：Dynamic Chunked Pipeline Parallel (CPP) |
| [index.md](../../source/developer_guide/Design_Documents/index.md) | 说明 Design Documents。 | 标题：Design Documents |
| [kvpp.md](../../source/developer_guide/Design_Documents/kvpp.md) | 说明 KVPP Design。 | 标题：KVPP Design |
| [layerwise_and_sparse_kv_cache_offloading.md](../../source/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.md) | 说明 Layerwise and Sparse KV Cache Offloading Design。 | 标题：Layerwise and Sparse KV Cache Offloading Design |
| [model_runner_v1_eplb.md](../../source/developer_guide/Design_Documents/model_runner_v1_eplb.md) | 说明 Model Runner V1 EPLB Architecture。 | 标题：Model Runner V1 EPLB Architecture |
| [model_runner_v2_eplb.md](../../source/developer_guide/Design_Documents/model_runner_v2_eplb.md) | 说明 Model Runner V2 EPLB Architecture。 | 标题：Model Runner V2 EPLB Architecture |
| [non_contiguous_kv_cache.md](../../source/developer_guide/Design_Documents/non_contiguous_kv_cache.md) | 说明 Non-Contiguous KV Cache。 | 标题：Non-Contiguous KV Cache |
| [npugraph_ex.md](../../source/developer_guide/Design_Documents/npugraph_ex.md) | 说明 Npugraph_ex。 | 标题：Npugraph_ex |
| [patch.md](../../source/developer_guide/Design_Documents/patch.md) | 说明 Patch in vLLM Ascend。 | 标题：Patch in vLLM Ascend |
| [persistent_csrc_build_cache.md](../../source/developer_guide/Design_Documents/persistent_csrc_build_cache.md) | 说明 Persistent Incremental Csrc Build Cache。 | 标题：Persistent Incremental Csrc Build Cache |
| [quantization.md](../../source/developer_guide/Design_Documents/quantization.md) | 说明 Quantization Adaptation Guide。 | 标题：Quantization Adaptation Guide |
| [speculative_kv_sliding_window.md](../../source/developer_guide/Design_Documents/speculative_kv_sliding_window.md) | 说明 Speculative Decoding KV Sliding Window。 | 标题：Speculative Decoding KV Sliding Window |

## `docs/source/developer_guide/contribution/`

父主题的 contribution 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [doc_writing.md](../../source/developer_guide/contribution/doc_writing.md) | 说明 Doc writing guide。 | 标题：Doc writing guide |
| [e2e_ci_test.md](../../source/developer_guide/contribution/e2e_ci_test.md) | 说明 E2E CI Test。 | 标题：E2E CI Test |
| [index.md](../../source/developer_guide/contribution/index.md) | 说明 Contributing。 | 标题：Contributing |
| [multi_node_test.md](../../source/developer_guide/contribution/multi_node_test.md) | 说明 Multi Node Test。 | 标题：Multi Node Test |
| [nightly_ci_test.md](../../source/developer_guide/contribution/nightly_ci_test.md) | 说明 Nightly CI Test。 | 标题：Nightly CI Test |
| [testing.md](../../source/developer_guide/contribution/testing.md) | 说明 Testing。 | 标题：Testing |

## `docs/source/developer_guide/evaluation/`

父主题的 evaluation 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [index.md](../../source/developer_guide/evaluation/index.md) | 说明 Accuracy。 | 标题：Accuracy |
| [using_ais_bench.md](../../source/developer_guide/evaluation/using_ais_bench.md) | 说明 Using AISBench。 | 标题：Using AISBench |
| [using_evalscope.md](../../source/developer_guide/evaluation/using_evalscope.md) | 说明 Using EvalScope。 | 标题：Using EvalScope |
| [using_lm_eval.md](../../source/developer_guide/evaluation/using_lm_eval.md) | 说明 Using lm-eval。 | 标题：Using lm-eval |
| [using_opencompass.md](../../source/developer_guide/evaluation/using_opencompass.md) | 说明 Using OpenCompass。 | 标题：Using OpenCompass |

## `docs/source/developer_guide/performance_and_debug/`

父主题的 performance_and_debug 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [index.md](../../source/developer_guide/performance_and_debug/index.md) | 说明 Performance and Debug。 | 标题：Performance and Debug |
| [msprobe_guide.md](../../source/developer_guide/performance_and_debug/msprobe_guide.md) | 说明 MSProbe Debugging Guide。 | 标题：MSProbe Debugging Guide |
| [optimization_and_tuning.md](../../source/developer_guide/performance_and_debug/optimization_and_tuning.md) | 说明 Optimization and Tuning。 | 标题：Optimization and Tuning |
| [performance_benchmark.md](../../source/developer_guide/performance_and_debug/performance_benchmark.md) | 说明 Performance Benchmark。 | 标题：Performance Benchmark |
| [service_profiling_guide.md](../../source/developer_guide/performance_and_debug/service_profiling_guide.md) | 说明 Service Profiling Guide。 | 标题：Service Profiling Guide |
