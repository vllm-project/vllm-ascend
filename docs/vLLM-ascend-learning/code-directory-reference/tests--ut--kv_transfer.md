# tests/ut/kv_transfer：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **4 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/kv_transfer/`

验证 kv_transfer 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_ascend_store_layerwise_lifecycle.py](../../../tests/ut/kv_transfer/test_ascend_store_layerwise_lifecycle.py) | 验证 存储 的测试；具体断言场景由右栏测试符号定位。 | 注释：Raw-token lifecycle with real scheduling, connectors, threads and CPU bytes.；定义：`test_raw_tokens_hybrid_roundtrip` |
| [test_layerwise_dcp_sharding.py](../../../tests/ut/kv_transfer/test_layerwise_dcp_sharding.py) | 验证 Decode 上下文并行 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for DCP shard handling in the layerwise KV pool.；定义：`test_dcp1_keeps_put_step_semantics`, `test_mla_tp4_dcp2_leaders_are_shard_minima`, `test_gqa_put_step1_all_ranks_lead`, `test_gqa_tp4_dcp2_put_step2` |
| [test_layerwise_multi_cache_spec.py](../../../tests/ut/kv_transfer/test_layerwise_multi_cache_spec.py) | 验证 缓存 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for ``build_layerwise_reuse_layout`` multi-cache-spec layers.；定义：`test_dsa_layer_with_5_specs_no_longer_raises`, `test_attn_selected_as_main`, `test_extra_main_specs_preserved`, `test_indexer_spec_optional` |
| [test_layerwise_pp_global_layers.py](../../../tests/ut/kv_transfer/test_layerwise_pp_global_layers.py) | 验证 layerwise pp global layers 的测试；具体断言场景由右栏测试符号定位。 | 注释：Unit tests for global-layer addressing of layerwise pool keys under PP.；定义：`test_default_offset_is_backward_compatible`, `test_offset_shifts_layer_ids`, `test_producer_stages_cover_decode_keyspace_exactly`, `test_no_overlap_between_stages` |
