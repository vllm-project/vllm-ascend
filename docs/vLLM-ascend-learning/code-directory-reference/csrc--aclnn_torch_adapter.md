# csrc/aclnn_torch_adapter：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/aclnn_torch_adapter/`

PyTorch 张量与 ACLNN 算子调用之间的适配。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [NPUBridge.cpp](../../../csrc/aclnn_torch_adapter/NPUBridge.cpp) | NPUBridge 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [NPUBridge.h](../../../csrc/aclnn_torch_adapter/NPUBridge.h) | NPUBridge 的公共接口/实现，供所在算子工程复用。 | 类型：`NPUBridge` |
| [NPUStorageImpl.cpp](../../../csrc/aclnn_torch_adapter/NPUStorageImpl.cpp) | NPUStorageImpl 的公共接口/实现，供所在算子工程复用。 | 按代码所在工程与编译角色归类；实现细节需结合调用方。 |
| [NPUStorageImpl.h](../../../csrc/aclnn_torch_adapter/NPUStorageImpl.h) | NPUStorageImpl 的公共接口/实现，供所在算子工程复用。 | 类型：`NPUStorageDesc`, `use_byte_size_t`, `NPUStorageImpl` |
| [op_api_common.h](../../../csrc/aclnn_torch_adapter/op_api_common.h) | 公共逻辑 的公共接口/实现，供所在算子工程复用。 | 类型：`aclOpExecutor`, `aclTensor`, `aclScalar` |
