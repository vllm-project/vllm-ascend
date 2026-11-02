# vllm_ascend/distributed/ec_transfer：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **2 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/distributed/ec_transfer/`

传输 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/distributed/ec_transfer/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [ec_memcache_connector.py](../../../vllm_ascend/distributed/ec_transfer/ec_memcache_connector.py) | 缓存 的接口/实现或辅助；所属职责：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 | 注释：ECMemcacheConnector backed by memcache.；定义：`ECMemcacheConnectorMetadata`, `_GetRequestParam` |
