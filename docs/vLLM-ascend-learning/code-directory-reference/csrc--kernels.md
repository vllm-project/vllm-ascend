# csrc/kernels：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **7 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `csrc/kernels/`

独立设备 kernel 和辅助实现。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [bgmv_expand.cpp](../../../csrc/kernels/bgmv_expand.cpp) | bgmv_expand 的公共接口/实现，供所在算子工程复用。 | 类型：`BGMVExpand` |
| [bgmv_shrink.cpp](../../../csrc/kernels/bgmv_shrink.cpp) | bgmv_shrink 的公共接口/实现，供所在算子工程复用。 | 类型：`BGMVShrink` |
| [math_utils.h](../../../csrc/kernels/math_utils.h) | 辅助函数 的公共接口/实现，供所在算子工程复用。 | 函数：`RoundUp` |
| [sgmv_expand.cpp](../../../csrc/kernels/sgmv_expand.cpp) | sgmv_expand 的公共接口/实现，供所在算子工程复用。 | 类型：`SGMVExpand` |
| [sgmv_shrink.cpp](../../../csrc/kernels/sgmv_shrink.cpp) | sgmv_shrink 的公共接口/实现，供所在算子工程复用。 | 类型：`SGMVShrink` |
| [types.h](../../../csrc/kernels/types.h) | types 的公共接口/实现，供所在算子工程复用。 | 类型：`AscendType` |
| [utils.h](../../../csrc/kernels/utils.h) | 辅助函数 的公共接口/实现，供所在算子工程复用。 | 类型：`AccType` |
