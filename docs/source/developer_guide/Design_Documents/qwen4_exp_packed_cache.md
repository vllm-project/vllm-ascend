# Qwen4Exp 标准 LBNHC 缓存

## 模型分组策略

Qwen 专用逻辑只负责识别缓存、物理页归一和固定组宽 12。
容量规划、`KVCacheTensor` 生成、跨 worker block 数重算和 block 管理使用标准路径。

先区分 target/draft，再按缓存类型及完整 Spec 分桶。
相同桶的层按原始层号排序，使用 `members = layer_names[index::num_groups]`
轮转拆组，其中 `num_groups = cdiv(len(layer_names), 12)`。
完整 Spec 相等才会同组，因此 checkpoint、投机块数、TP 复制属性等差异不会被吞掉。

无 MTP 的 48 层模型产生 7 个普通 `KVCacheGroupSpec`，包含 73 个真实缓存层：

| 组 | 成员数 | 管理语义 |
| --- | ---: | --- |
| Attention | 12 | 完整 token 历史 |
| Indexer | 12 | 压缩历史，保持原始压缩比例 |
| GDN | 12 + 12 + 12 | 各组独立 checkpoint 和回收 |
| PLE | 1 | 单独组，保留 TP replicated |
| RingBuffer | 12 | 请求内循环，不参与 prefix cache 和 KV transfer |

PLE 不添加 11 个虚构层。标准规划器根据最大真实组宽预留物理空间。
MTP 的 target/draft 分桶独立，保留 `is_eagle_group`；7 组只描述无 MTP 配置。

## 物理页归一

从无 padding 的实际 Spec 推导最大有效页，并满足 dtype、组件边界和 GBSA 子页对齐。
只调整 `page_size_padded`，不改变 dtype、有效 shape、token block size、压缩比例、
RingBuffer 长度、checkpoint 或投机状态。当前真实 TP8 规格推导出 768 KiB；不是硬编码常量。

归一后标准 LBNHC 校验直接通过，无混合页大小豁免。设单层物理页为 P，block 数为 N，
当前 worker 最大组宽为 W：

```text
block_stride = P
layer_stride = N * P
backing_size = W * N * P
layer_address = backing_base + descriptor.offset + layer_index * layer_stride
```

完整 TP8 rank 的 W=12。无 MTP 时标准规划器生成 7 个描述符，全部共享一份 backing。
各描述符的 size 重复描述这份 backing，不能累加。各组使用共享 BlockPool 的不同 block ID；
同组层占用不同 layer slot，不同组可复用 layer slot，但不能同时拥有同一个 block ID。

## Runner 与后端

Runner 不再调用 Qwen 专用 allocate/reshape。标准共享路径校验描述符的大小、offset、
layer stride 和 block stride，只分配一次 backing；QSA 的 sparse 属性不会导致每层单独分配。
组件视图保留 allocator 的非零 storage offset，不通过复制整个 cache 改变连续性。

- GDN：每页内依次保存 Conv、SSM，各自使用实际 dtype 和有效 shape。
- PLE：每页一个状态视图，使用独立组的 block table。
- Attention：QSA 后端把标准页解释为连续 K 和连续 V 两段，支持 GBSA 128-token 子页寻址。
- Indexer、RingBuffer：保持有效 shape，block 维使用归一后的物理页步长。

PP 保留全局 group ID，仅投影本地真实层，不为远端层预留 slot。
对上游投影保留一个最小元数据修复：用 `replace` 保留 `enable_kv_transfer`、draft 和 role 等字段。
空组保留元数据，标准描述符生成跳过空层集合。

KV transfer 按实际 storage 注册，按组件有效 payload 和实际 stride 传输。
开启传输时，一份 backing 的固定 2 MiB 分配对齐预留在 worker 可用预算中扣除一次；
标准 planner 和跨 rank 重算无需专用公式。

## 验证与容量比较

测试覆盖 7 组/73 层、页归一、标准描述符、一次分配、不同组的 block 隔离、
非零 storage offset、BF16/FP32、PP、MTP、跨 rank 重算、checkpoint 和有效传输范围。
NPU 测试覆盖 QSA cache 写入、Graph 重放和 checkpoint 复制。
真实验证还需执行 Eager prefill/decode、并发及 prefix cache 命中，并导出运行时配置。

新分组会改变一个请求从 BlockPool 占用的总 block 数，不能只比较每 pool block 字节数。
应在相同工作负载下比较所有组的实际 block 占用乘以 W×P，并分别报告容量和推理性能。
旧周期方案的 22.97% 节省不适用于本方案。

## 移除路径

通用分组器支持合适的组宽策略和物理页归一后，删除 Qwen 分组分支。
上游 PP 投影完整保留 group 元数据后，删除投影包装。
标准容量规划、描述符和 block 管理已无需 Qwen 专用实现。
