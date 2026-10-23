# Qwen4Exp LBNHC 四行 KV cache

## 调度与层对应关系

`build_layer_tuples` 根据原始模型层号和 `layer_types` 构造周期：Attention、Indexer、
三个有序 GDN、RingBuffer。它校验成员完整性、重复层名和 token 覆盖关系，单独识别一份 PLE。
配置生成和 worker 使用同一个确定性函数；字典顺序不影响结果。

主模型使用普通 `KVCacheGroupSpec`，内部保留 `UniformTypeKVCacheSpecs`：

| Group | 成员 | 生命周期 |
| --- | --- | --- |
| G0 | Attention K/V、Indexer compressed K | 完整 token 历史 |
| G1 | GDN Conv/SSM、单份 PLE | 相同 checkpoint、恢复和回收规则 |
| G2 | RingBuffer state | 请求内循环，不参与 prefix cache 或 KV transfer |

GDN/PLE 合组前检查 block size、prefix/replay、cache mode、投机 block 数、checkpoint 数与对齐。
各层原始 dtype、`mamba_type`、`tp_replicated` 保持不变，不修改全局 uniform 判定。

PP 仅投影 `layer_names`，保留全局 group ID、空组和各组原始成员 specs；本地 specs 覆盖对应成员。
这些普通字段足以重新构造全局周期，没有额外挂载拓扑。布局只为本地有效组件计算容量，槽位
仍依据原始模型层号，不把 PP 本地层序号当成全局层号。

## 行几何与槽位

布局固定为 `LBNHC`。每个槽位内部依次保存所有 block，随后才是下一个槽位。

| 行 | 有效内容 | 主模型槽位映射 |
| --- | --- | --- |
| 1 | V、PLE、RingBuffer state | V/Ring 使用周期号；PLE 只绑定槽位 2 |
| 2 | K、GDN SSM | K 使用周期号；SSM 使用 `3 × 周期号 + 周期内 GDN 序号` |
| 3 | GDN Conv | 使用 `3 × 周期号 + 周期内 GDN 序号` |
| 4 | Indexer compressed K | 使用周期号 |

各组从槽位 0 开始复用地址。同组不同层独立；共享 BlockPool 保证同一 block ID 同时只属于一个组。
PLE 位于第一个状态周期的第三列，即示意图的第 4 列，只保存一份。

设 `B = num_blocks`，`Pr` 为该行所有有效组件的最大字节数向上对齐 16 字节，
`Lr` 为最大有效槽位号加一，则：

```text
block_stride = Pr
layer_stride = B × Pr
Sr = Lr × B × Pr
address = backing.data_ptr + row.offset + slot × layer_stride + block_id × block_stride
```

跨组组件共用该行的 block stride。小组件只暴露有效 payload，padding 由真实字节容量决定，
不再将 RingBuffer 剩余容量填入所有行。Indexer 使用真实压缩后的 shape。

参考 BF16、Attention block size 128、12 个周期：

| 行 | 公共 page 字节数 Pr | 槽位数 Lr | 每 pool block 字节数 |
| --- | ---: | ---: | ---: |
| 1 | 65536 | 12 | 786432 |
| 2 | 65536 | 36 | 2359296 |
| 3 | 2560 | 36 | 92160 |
| 4 | 8192 | 12 | 98304 |
| 合计 | | | 3336192 |

这是单元测试规格的示例，实际 TP 分片、block size 和混合 dtype 以真机配置为准。

## 四个标准描述符、两份 backing

| 描述符 | size | offset | layer_stride | block_stride |
| --- | --- | --- | --- | --- |
| tensor1 | S1 + S2 + S3 | 0 | B × P1 | P1 |
| tensor2 | S1 + S2 + S3 | S1 | B × P2 | P2 |
| tensor3 | S1 + S2 + S3 | S1 + S2 | B × P3 | P3 |
| tensor4 | S4 | 0 | B × P4 | P4 |

全部是标准 `KVCacheTensor`。前三个 `size` 表示同一 backing 的大小，预算为
`S1 + S2 + S3 + S4`，不能累加四个描述符的 `size`。
`layers` 表示参与该行的层，混合组列表下标不表示物理槽位；槽位由 tuples 推导。

主模型严格调用 allocator 两次。backing1 经 `split_tensor` 无复制地切成前三行；backing2
保存第四行。allocator 保留设备、初始化和对齐处理。带步长视图的 `storage_offset`
包含 allocator 返回切片自身的偏移。启用 KV transfer 时，预算另计每份分配的 2 MiB 对齐空间。

worker 保留薄入口，分配阶段按层输出 `(K, V)`、`(Conv, SSM)` 或单组件 tuple。
reshape 根据原 spec 的 shape、dtype、状态顺序绑定，使用 dtype view 和 inner-dimension
unflatten 保留外层 stride；不通过 `cat`、`clone` 或 `contiguous` 重建缓存。

MTP 使用独立 Full/RingBuffer 调度组，独立 block 所有权允许它复用主模型行槽位。
hidden-state cache 使用额外描述符和 backing。只有部分组件的 PP rank 不分配空 backing。
这些情形单独测试，不要求它们的描述符或分配数等于主模型。

## 布局选择、预算与传输

Ascend QSA attention/state 后端均声明 `LBNHC`。混合 page 校验仅对已识别的 Qwen4Exp
使用上述行布局规则；其他模型仍调用通用校验。预算、worker block 数统一重算和 PP 投影
由现有 hooks 路由到模型实现。

传输元数据从 groups 遍历真实层一次，组件地址、block stride、有效长度和 shape 均来自
已绑定 tensor。同一层可以出现在多个行描述符中。注册范围按真实 storage 去重，主模型
注册两份 backing；传输只复制所属层的有效数据，不复制其他组或 padding。

## 验证

主要测试入口：

```bash
pytest -q tests/ut/models/qwen4_exp/test_cache_config.py
pytest -q tests/ut/worker/test_model_runner_v1.py \
  tests/ut/kv_offload/mooncake_v2/test_base_worker.py -k qwen_packed_cache
pytest -q tests/ut/patch/worker/test_patch_mamba_utils.py
pytest -q tests/e2e/pull_request/one_card/test_qwen4_exp_cache_layout.py
```

覆盖四描述符与两 storage、地址/stride、一份 PLE、混合 dtype 和页内 padding、
Indexer 压缩 shape、block 隔离、跨组复用、预算和对齐开销、序列化、PP、MTP/hidden extras。
NPU 测试覆盖缓存写入、checkpoint 状态复制与图重放。

真机服务验证另外按 TP rank 记录 `CacheGroups`、`kv_cache_config`、allocate 和 reshape
输出元数据，以及实际 allocator 调用。真实权重的 prefill/decode、重复前缀和并发请求
结果与启动命令、服务日志一并保存，不能用单元测试或仅启动成功代替这些证据。
