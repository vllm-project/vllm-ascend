# vllm_ascend/device_allocator：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/device_allocator/`

设备内存分配及 sleep/wakeup 相关资源管理。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/device_allocator/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [camem.py](../../../vllm_ascend/device_allocator/camem.py) | CaMem 设备分配管理：建立映射、记录分配信息以及解除映射/释放。 | 定义：`AllocationData`, `CaMemAllocator`, `find_loaded_library`, `create_and_map`, `unmap_and_release` |
| [sleep_mem_optimized.py](../../../vllm_ascend/device_allocator/sleep_mem_optimized.py) | 休眠资源 的接口/实现或辅助；所属职责：设备内存分配及 sleep/wakeup 相关资源管理。 | 定义：`SleepWakeupManager`, `AclGraphSleepWakeupManager` |
