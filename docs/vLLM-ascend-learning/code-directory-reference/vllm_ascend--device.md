# vllm_ascend/device：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **7 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/device/`

设备识别、硬件能力、策略及设备算子接口。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/device/__init__.py) | Python 包入口，导入或重新导出本模块接口；需检查导入副作用。 | 导入：`vllm_ascend.device.device_config`, `vllm_ascend.device.hardware_profile` |
| [device_config.py](../../../vllm_ascend/device/device_config.py) | 当前设备配置读取与初始化。 | 注释：Runtime entry point for Ascend hardware configuration.；定义：`DeviceConfig`, `get_device_config`, `get_ascend_device_type`, `is_310p` |
| [device_op.py](../../../vllm_ascend/device/device_op.py) | 集中封装硬件差异和设备算子调用。 | 定义：`BaseDeviceAdaptor`, `A5DeviceAdaptor`, `get_device_adaptor` |
| [hardware.py](../../../vllm_ascend/device/hardware.py) | 设备类型识别与相关硬件定义。 | 注释：Pure hardware identification helpers.；定义：`AscendDeviceType`, `device_type_from_soc_version`, `device_type_from_runtime_soc` |
| [hardware_profile.py](../../../vllm_ascend/device/hardware_profile.py) | 明确设备 capability、policy、family、量化/权重布局与 Worker 选择契约。 | 注释：Semantic profiles for supported Ascend hardware families.；定义：`HardwareCapability`, `AttentionBackendFamily`, `get_hardware_profile`, `get_current_hardware_profile` |
| [mxfp_compat.py](../../../vllm_ascend/device/mxfp_compat.py) | MXFP 格式/算子版本之间的兼容辅助。 | 定义：`is_add_rms_norm_dynamic_mx_quant_fusion_available`, `is_rms_norm_dynamic_mx_quant_fusion_available`, `ensure_mxfp8_scale_dtype_available` |
| [utils.py](../../../vllm_ascend/device/utils.py) | 辅助函数 的接口/实现或辅助；所属职责：设备识别、硬件能力、策略及设备算子接口。 | 定义：`npu_large_head_prefill_attention` |
