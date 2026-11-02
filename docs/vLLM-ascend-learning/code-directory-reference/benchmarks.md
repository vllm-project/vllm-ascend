# benchmarks：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **10 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `benchmarks/`

性能测量脚本、输入配置和报告模板。

| 子目录 | 职责 |
| --- | --- |
| [scripts/](../../../benchmarks/scripts) | scripts 子模块；所属范围：性能测量脚本、输入配置和报告模板。 |
| [tests/](../../../benchmarks/tests) | tests 子模块；所属范围：性能测量脚本、输入配置和报告模板。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../benchmarks/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：vLLM Ascend Benchmarks |
| [prepare_indexer_indices.py](../../../benchmarks/prepare_indexer_indices.py) | 历史位置索引、准备 性能测量、运行配置或报告模板。 | 注释：Compare fused indexer postprocessing in an NPU graph.；定义：`reference_indices`, `graph_latency_us`, `benchmark` |
| [quantize_indexer_query.py](../../../benchmarks/quantize_indexer_query.py) | 量化、历史位置索引 性能测量、运行配置或报告模板。 | 注释：Compare indexer query quantization latency inside an NPU graph.；定义：`reference`, `graph_latency_us`, `main` |
| [requirements-bench.txt](../../../benchmarks/requirements-bench.txt) | 该目录运行、测试或构建所需的依赖清单。 | 依据文件格式及所在目录归类。 |

## `benchmarks/scripts/`

scripts 子模块；所属范围：性能测量脚本、输入配置和报告模板。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [convert_json_to_markdown.py](../../../benchmarks/scripts/convert_json_to_markdown.py) | 卷积 性能测量、运行配置或报告模板。 | 定义：`read_markdown`, `results_to_json` |
| [perf_result_template.md](../../../benchmarks/scripts/perf_result_template.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：perf_result_template。 | 标题：Online serving tests |
| [run-performance-benchmarks.sh](../../../benchmarks/scripts/run-performance-benchmarks.sh) | 性能测量 性能测量、运行配置或报告模板。 | 函数：`check_npus`, `ensure_sharegpt_downloaded`, `json2args` |

## `benchmarks/tests/`

tests 子模块；所属范围：性能测量脚本、输入配置和报告模板。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [latency-tests.json](../../../benchmarks/tests/latency-tests.json) | 延迟 性能测量、运行配置或报告模板。 | JSON list 数据。 |
| [serving-tests.json](../../../benchmarks/tests/serving-tests.json) | serving-tests 性能测量、运行配置或报告模板。 | JSON list 数据。 |
| [throughput-tests.json](../../../benchmarks/tests/throughput-tests.json) | 吞吐 性能测量、运行配置或报告模板。 | JSON list 数据。 |
