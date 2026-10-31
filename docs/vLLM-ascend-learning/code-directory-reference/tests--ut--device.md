# tests/ut/device：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **4 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/device/`

验证 device 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/device/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_device_config.py](../../../tests/ut/device/test_device_config.py) | 验证 配置 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_device_type_from_soc_version`, `test_device_type_from_runtime_soc`, `test_device_config_uses_build_info`, `test_device_config_supports_legacy_soc_build_info` |
| [test_device_op.py](../../../tests/ut/device/test_device_op.py) | 验证 device op 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_partial_rotary_preserves_storage_and_adapts_native_signature`, `test_rms_norm_cast_only_uses_supported_fused_operator`, `test_host_registration_flags_follow_runtime_abi`, `test_deepseek_v41_backend_is_device_routed` |
| [test_hardware_profile.py](../../../tests/ut/device/test_hardware_profile.py) | 验证 性能分析 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_hardware_profile_implementation_matrix`, `test_hardware_profile_capability_matrix`, `test_current_hardware_profile_uses_device_config`, `test_current_hardware_profile_is_dynamo_safe` |
