# vllm_ascend/observability：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/observability/`

运行观测的 provider、handler 与指标接入。

| 子目录 | 职责 |
| --- | --- |
| [config/](../../../vllm_ascend/observability/config) | 配置 子模块；所属范围：运行观测的 provider、handler 与指标接入。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/observability/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Observability integrations owned by vLLM Ascend. |
| [handlers.py](../../../vllm_ascend/observability/handlers.py) | 注册、指标、专家负载均衡 的接口/实现或辅助；所属职责：运行观测的 provider、handler 与指标接入。 | 注释：vLLM Ascend-owned metric handlers.；定义：`eplb_do_update_hotness_handler` |
| [provider.py](../../../vllm_ascend/observability/provider.py) | 向 MS Service Metric 提供指标配置。 | 注释：Metric configuration contributed to MS Service Metric.；定义：`get_metric_provider` |

## `vllm_ascend/observability/config/`

配置 子模块；所属范围：运行观测的 provider、handler 与指标接入。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [base_metrics.yaml](../../../vllm_ascend/observability/config/base_metrics.yaml) | 所在模块的结构化配置或参数模板。 | 结构化配置；语义依调用方。 |
| [eplb_metrics.yaml](../../../vllm_ascend/observability/config/eplb_metrics.yaml) | 所在模块的结构化配置或参数模板。 | 结构化配置；语义依调用方。 |
