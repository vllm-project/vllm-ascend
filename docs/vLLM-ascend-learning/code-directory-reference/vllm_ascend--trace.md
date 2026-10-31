# vllm_ascend/trace：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **6 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/trace/`

请求时间线标记、日志和 tracing 包装。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../vllm_ascend/trace/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：vLLM Ascend Trace |
| [__init__.py](../../../vllm_ascend/trace/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 注释：Lightweight request timeline tracing for vLLM Ascend. |
| [patcher.py](../../../vllm_ascend/trace/patcher.py) | 时间线跟踪 的接口/实现或辅助；所属职责：请求时间线标记、日志和 tracing 包装。 | 定义：`trace_enabled`, `iter_namelist_paths`, `apply_trace_patches` |
| [prof_wrapper.py](../../../vllm_ascend/trace/prof_wrapper.py) | 用 marker/timer 包装操作，记录对应时间线与耗时信息。 | 定义：`execute_operation`, `marker_prof_wrapper`, `timer_prof_wrapper` |
| [tracelogger_namelist.yml](../../../vllm_ascend/trace/tracelogger_namelist.yml) | 所在模块的结构化配置或参数模板。 | 顶层项：`type`, `targets` |
| [utils.py](../../../vllm_ascend/trace/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：请求时间线标记、日志和 tracing 包装。 | 定义：`safe_print`, `get_ip`, `get_role` |
