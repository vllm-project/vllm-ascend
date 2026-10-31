# vllm_ascend：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **12 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## 阅读关系

先读 `setup.py` 的 entry points，再读本包注册函数、`platform.py`、`utils.py`，最后进入 Worker。配置文件说明意图，`device/` 的能力与平台验证决定组合是否成立。完整路线见总导读 2.2。

[关键文件与调用关系](../code-directory-guide.md) · [原理与时序](../detailed-design.md)

## `vllm_ascend/`

Ascend 插件 Python 主包：将上游 vLLM 的引擎和模型接口接到 NPU 实现。

| 子目录 | 职责 |
| --- | --- |
| [_310p/](vllm_ascend--_310p.md) | Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [_cann_ops_custom/](vllm_ascend--_cann_ops_custom.md) | 自定义 CANN 算子安装/打包位置的占位目录。 |
| [attention/](vllm_ascend--attention.md) | Attention 后端、元数据构建、KV 访问和稀疏注意力执行。 |
| [compilation/](vllm_ascend--compilation.md) | NPU 图捕获、重放、运行参数更新与编译融合接入。 |
| [core/](vllm_ascend--core.md) | 调度器扩展、KV 缓存规格、缓存管理和 KVPP 物理放置。 |
| [device/](vllm_ascend--device.md) | 设备识别、硬件能力、策略及设备算子接口。 |
| [device_allocator/](vllm_ascend--device_allocator.md) | 设备内存分配及 sleep/wakeup 相关资源管理。 |
| [distributed/](vllm_ascend--distributed.md) | 并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [ec_manager/](vllm_ascend--ec_manager.md) | 多模态编码器结果缓存策略，包括重算成本与访问得分。 |
| [eplb/](vllm_ascend--eplb.md) | 专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 |
| [lora/](vllm_ascend--lora.md) | LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 |
| [model_executor/](vllm_ascend--model_executor.md) | 模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。 |
| [model_loader/](vllm_ascend--model_loader.md) | 可选加载器与权重载入扩展。 |
| [models/](vllm_ascend--models.md) | 模型注册、模型专有层结构、权重映射与草稿模型实现。 |
| [observability/](vllm_ascend--observability.md) | 运行观测的 provider、handler 与指标接入。 |
| [ops/](vllm_ascend--ops.md) | 模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 |
| [patch/](vllm_ascend--patch.md) | 在既有上游接口缺少所需扩展点时安装兼容补丁。 |
| [profiler/](vllm_ascend--profiler.md) | 设备性能分析器与运行记录接入。 |
| [quantization/](vllm_ascend--quantization.md) | 量化配置解析、层适配、权重后处理与计算方案。 |
| [sample/](vllm_ascend--sample.md) | Logits 处理、TopK/TopP、采样和投机验证恢复。 |
| [spec_decode/](vllm_ascend--spec_decode.md) | 草稿生成器、候选组织及投机解码辅助。 |
| [tokenization/](vllm_ascend--tokenization.md) | 可选的长文本并行 tokenization。 |
| [trace/](vllm_ascend--trace.md) | 请求时间线标记、日志和 tracing 包装。 |
| [weight_switch/](vllm_ascend--weight_switch.md) | Linear/输出投影完整权重切换和并行域相关生命周期。 |
| [worker/](vllm_ascend--worker.md) | Worker 生命周期、V1/V2 Runner、输入组批、缓存分配与设备元数据。 |
| [xlite/](vllm_ascend--xlite.md) | Xlite 专用 Worker、模型包装及执行适配。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/__init__.py) | 注册 NPU 平台、连接器、加载器和模型，安排进程级补丁。 | 定义：`register`, `register_connector`, `register_model_loader` |
| [ascend_config.py](../../../vllm_ascend/ascend_config.py) | 解析和验证 Ascend 扩展配置，描述图、通信、KV、调度等策略。 | 定义：`KVPPConfig`, `AscendCompilationConfig`, `compute_mega_moe_buffer_tokens_per_rank`, `is_mega_moe_supported`, `validate_additional_config_bool` |
| [ascend_forward_context.py](../../../vllm_ascend/ascend_forward_context.py) | 为单次前向选择 MoE 通信等运行条件，管理 Ascend 扩展上下文。 | 定义：`MoECommType`, `_ExtraForwardContextProxy`, `override_mrv2_in_profile_run`, `get_mrv2_in_profile_run`, `use_cann_megamoe` |
| [batch_invariant.py](../../../vllm_ascend/batch_invariant.py) | 批次不变性相关环境与计算适配，减少批次变化造成的数值差异。 | 定义：`add_rms_norm`, `reduce_sum`, `override_envs_for_invariance` |
| [config_utils.py](../../../vllm_ascend/config_utils.py) | 提供较少依赖的上游兼容 config/dataclass 辅助。 | 注释：Dependency-light helpers for vLLM-compatible config dataclasses.；定义：`config`, `config`, `config` |
| [cpu_binding.py](../../../vllm_ascend/cpu_binding.py) | 识别 CPU/设备关系并安排 CPU 亲和性绑定。 | 定义：`DeviceInfo`, `CpuAlloc`, `is_arm_cpu`, `execute_command`, `bind_cpus` |
| [envs.py](../../../vllm_ascend/envs.py) | 集中定义与读取 Ascend 环境变量及其默认值。 | 定义：`_strict_binary_env`, `__getattr__`, `__dir__` |
| [logger.py](../../../vllm_ascend/logger.py) | 日志格式、模块标记和文件输出设置。 | 注释：Logging configuration for vLLM-Ascend.；定义：`AscendFormatter`, `AscendColoredFormatter`, `configure_ascend_file_logging`, `configure_ascend_logging` |
| [meta_registration.py](../../../vllm_ascend/meta_registration.py) | 为自定义 LoRA 等算子注册 shape/meta 推断实现。 | 定义：`register_meta_if_necessary`, `bgmv_expand_meta`, `sgmv_expand_meta` |
| [platform.py](../../../vllm_ascend/platform.py) | NPU 平台接入；选择 Attention、编译、Worker，并验证模型/并行/缓存配置。 | 定义：`NPUPlatform` |
| [profiling_config.py](../../../vllm_ascend/profiling_config.py) | 生成服务性能分析的配置文件。 | 注释：Service profiling configuration generator module.；定义：`get_config_dir`, `generate_service_profiling_config` |
| [utils.py](../../../vllm_ascend/utils.py) | 模型/设备能力判定、补丁安装、配置兼容与公共工具。 | 定义：`is_gqa_pcp_dcp_config`, `extract_dsv4_layer_index`, `get_dsv4_spec_layer_idx_from_weight_name` |
