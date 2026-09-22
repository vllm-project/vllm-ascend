> Imported from private `xwLearnsLLM/nanovllm-DSA-offload`, commit `012962af05f06bf1bdd089ca7e7e4357d021682f`. In this repository the entry point is `torch.ops._C_ascend.npu_fused_li_manage_mtp`. CANN names and build integration are adapted to vLLM-Ascend.

# `fused_li_manage_mtp` 标准接口与行为

`fused_li_manage_mtp` 将官方 Lightning Indexer（LI）的 TopK 检索、HBM
sparse-cache 的首次填充/稳态淘汰，以及 MTP 多路 query 的 union 管理融合为一次
NPU 调用。一个 batch 可混合 MTP0–MTP13（每请求 1–14 路 query），也可混合
非卸载、首次卸载和卸载稳态请求。

## 接口

```python
fused_li_manage_mtp(
    index_weights,                 # bf16/fp16 [T, N]，只读；每条 query、每个 index head 的聚合权重，N 必须为 32 或 64。
    query_dequant_scale,           # fp32 [T, N]，只读；C8 ABI 参数，当前 kernel 仅校验 shape/dtype，不读取数值。
    query,                         # bf16/fp16 [T, N, 128]，只读；TND 格式 index query，同一请求的多路 query 在 T 维连续。
    index_key_dequant_scale,       # fp32 [INDEX_BLOCKS, 128, 1]，只读；C8 ABI 参数，当前 kernel 仅校验 shape/dtype，不读取数值。
    index_key_cache,               # bf16/fp16 [INDEX_BLOCKS, 128, 1, 128]，只读；index key cache，dtype 必须与 query/index_weights 相同。
    index_block_table,             # int32 [B, INDEX_MAX_BLOCKS]，只读；每个请求的 index-key block 映射，INDEX_MAX_BLOCKS <= 16384。
    actual_seq_lengths_query,      # int32 [B]，只读；各请求 query 在 T 中的累计结束位置，严格递增，最后一项必须等于 T。
    actual_seq_lengths_key,        # int32 [B]，只读；每个请求最后一路 query 对应的实际 KV 长度，其他路的因果可见长度由该值推导。
    offload_seq_lengths_key,       # int32 [B]，只读；每个请求可检索、可卸载的稳定 source prefix L，仅 -2/-1 使用，必须为 128 的倍数。
    num_cache_tokens,              # int32 [B]，只读；每个请求的 HBM sparse-cache 预算 C，仅 -2/-1 使用，必须为 128 的倍数。
    request_state,                 # int32 [B]，只读；请求状态，仅允许 -3（非卸载）、-2（首次卸载）或 -1（卸载稳态）。
    req_pool_entries,              # int32 [B]，只读；每个请求所拥有的 cache_slots_pool 行号，同一次调用的 active request 必须互不重复。
    cache_slots_pool,              # int32 [POOL_SIZE, SOURCE_CAPACITY]，读写、持久化；source token ID 到 HBM logical slot 的映射状态。
    topk_src_ids,                  # int32 [T, 1, 2048]，只写；每路 query 的 TopK source token ID。
    topk_dst_slots,                # int32 [T, 1, 2048]，只写；与 topk_src_ids 同位置对应的 HBM logical slot，或 -1 padding。
    topk_miss_counts,              # int32 [T]，只写；每路 query 的 TopK miss 数。
    miss_src_ids,                  # int32 [B, 32768]，只写；每请求本轮需要搬入 HBM 的 source token ID 列表。
    miss_dst_slots,                # int32 [B, 32768]，只写；与 miss_src_ids 同位置对应的目标 HBM logical slot。
    miss_counts,                   # int32 [B]，只写；每请求有效搬运列表长度。
) -> None
```

`query`、`index_key_cache` 与
`index_weights` 必须使用同一种 BF16 或 FP16。除两个 dequant scale 外，metadata
和所有输出均为 int32。`cache_slots_pool` 与所有输出均为原地写入；接口没有返回
tensor。

`SOURCE_CAPACITY = INDEX_MAX_BLOCKS * 128`，并且
`INDEX_MAX_BLOCKS <= 16384`，所以当前最大物理 source capacity 为
`2^21 = 2,097,152`。

## MTP 路数、query 布局与因果长度

令请求编号为 `i`：

```text
query_start = i == 0 ? 0 : actual_seq_lengths_query[i - 1]
query_end   = actual_seq_lengths_query[i]
Q           = query_end - query_start                    # 1 <= Q <= 14
```

请求 `i` 的 query 行为 `[query_start, query_end)`。例如请求依次为 MTP2、MTP3、
MTP0 时，`actual_seq_lengths_query = [3, 7, 8]`。

`actual_seq_lengths_key[i]` 是最后一路 query 的 KV 长度。对该请求中的某个
`query_row`：

```text
later_queries = query_end - 1 - query_row
visible_length = actual_seq_lengths_key[i] - later_queries
```

因此越靠前的 MTP 路，因果可见 key 越少。`-3` 直接使用各自的
`visible_length`；`-2/-1` 使用共同的稳定 prefix `[0, L)`，框架必须保证该 prefix
对该请求的每一路 query 都因果可见。

## `cache_slots_pool` 映射语义

对请求 `i`，其 pool 行为 `row = req_pool_entries[i]`。该行的物理宽度始终是
`SOURCE_CAPACITY`，而不是 `actual_seq_lengths_key` 或 `L`。

- `-3` 非卸载：整行被写为 identity mapping
  `pool[row, source] = source`，范围为 `[0, SOURCE_CAPACITY)`。
- `-2/-1` 卸载：对稳定 prefix `[0, L)`，值在 `[0, C)` 表示该 source 已常驻
  HBM 并映射到相应 logical slot；`INT32_MIN` 表示该 source 不常驻。
- `-1` 稳态中，`[C, SOURCE_CAPACITY)` 的 identity 值不是合法 sparse mapping。
  它只会在 `-3 -> -1` 转换的 identity row 中暂时出现，并由 kernel 清理。

因此，“任意非负值”不是卸载 row 的通用合法状态：在卸载模式中，只有 `[0, C)`
才是有效 resident slot；无效值统一为 `INT32_MIN`。

## 请求状态与输出

### `request_state = -3`：非卸载

- 每一路在 `[0, visible_length)` 上执行普通 LI；若可见长度不足 2048，剩余
  TopK 位置填 `-1`。
- 不使用 `offload_seq_lengths_key` 或 `num_cache_tokens` 做检索/缓存管理；不执行
  hit/miss 分类、淘汰和搬运列表生成。
- `miss_counts[i] = 0`，该请求的全部 `topk_miss_counts = 0`。
- `topk_src_ids` 保持普通 LI 的 score 排序；`topk_dst_slots` 与其逐项相同，
  padding 也为 `-1`。此状态不存在“miss prefix / hit suffix”语义。
- 每次调用都会把该请求的完整 pool 行恢复成 identity mapping。
- `miss_src_ids`、`miss_dst_slots` 的内容无效。

### `request_state = -2`：首次卸载 decode

- 每路在 `[0, L)` 上检索；该请求已有的 pool 内容不作为输入状态使用。kernel
  会清空整行并重新建立 sparse mapping，因此可从空 row、旧 `-1` row 或 `-3`
  identity row 进入。
- 先取多路 Top-2048 的 unique union；union 内按 source ID 递增。再从不在 union
  中的 source 按 source ID 递增补齐，直至 resident token 总数为 `C`。
- 最终 resident source 写入 `miss_src_ids[i, :C]`，并分配
  `miss_dst_slots[i, :C] = [0, 1, ..., C - 1]`；`miss_counts[i] = C`。
- 每路 TopK 均视为 miss，因此 `topk_miss_counts[route] = 2048`。TopK source
  输出的 miss 区按 source ID 递增，destination 是新分配的 resident slot。

### `request_state = -1`：卸载稳态

- 正常 row 在 `[0, L)` 中恰有 `C` 个 resident source，slot 构成 `[0, C)` 的
  双射；其他 source 的映射为 `INT32_MIN`。
- kernel 通过探测 `cache_slots_pool[row, C]` 识别从 `-3` 遗留的 identity row。
  仅 probe 显示 identity 时，才将后缀 `[C, SOURCE_CAPACITY)` 批量填为
  `INT32_MIN`；已经是正常 sparse row 的稳态调用不会扫描整行。
- 随后在 `[0, L)` 完成 hit/miss 判定、各路 miss union、受 TopK 保护的 victim
  选择、slot 回收、pool 更新与搬运列表生成。
- 对卸载 TopK，miss 位于前缀、hit 位于后缀；两个区间均按 source ID 递增。
  `topk_miss_counts[route]` 是该路 miss 前缀长度。
- `miss_src_ids[i, :miss_counts[i]]` 是所有路 TopK unique miss source 的有序 union；
  `miss_dst_slots` 是这些 incoming source 获得的回收 slot。

除 `-3/-2/-1` 外的所有状态（包括 `request_state >= 0`）当前均非法；没有已实现
的“部分空 slot”预留状态。

## 动态约束

令 `L = offload_seq_lengths_key[i]`、`C = num_cache_tokens[i]`：

- `1 <= Q <= 14`；`actual_seq_lengths_query` 严格递增，最后一项等于 `T`。
- `Q <= actual_seq_lengths_key[i] <= SOURCE_CAPACITY`。
- 对 `-3`：`index_key_cache` 与 `index_block_table` 必须覆盖
  `[0, actual_seq_lengths_key[i])`。
- 对 `-2/-1`：`index_key_cache` 与 `index_block_table` 必须覆盖 `[0, L)`，且
  `C <= L <= actual_seq_lengths_key[i]`、`L >= 2048`、`L/C` 均为 128 的倍数。
- 为保证共同 prefix 对所有 MTP 路因果可见：

  ```text
  L <= floor((actual_seq_lengths_key[i] - Q) / 128) * 128
  ```

- 当 `2048 <= L <= Q * 2048` 时，必须 `C = L`；当 `L > Q * 2048` 时，必须
  `Q * 2048 <= C <= 32640`。
- `req_pool_entries[i]` 必须落在 `[0, POOL_SIZE)`；同一次调用的 active request
  不得并发写同一 pool row。

Python/C++ binding 负责静态 shape、dtype、连续性、设备和容量校验。生产框架在
构造 metadata 时负责满足以上逐请求动态约束；kernel 对动态非法值执行安全保护：
不破坏 pool，TopK 填 `-1`，并将相关计数写 0。该保护不能替代框架侧校验。

## 生命周期

生命周期由框架通过 `request_state` 驱动；`cache_slots_pool[row]` 是跨调用持久化的
缓存状态，且同一请求生命周期内 `row = req_pool_entries[i]` 必须保持不变。

| 调度场景 | 状态序列 | 框架切换要求 | 算子处理 | 切换后 row |
|---|---|---|---|---|
| 请求从一开始就卸载 | `-2 -> -1 -> ...` | 首次调用前 pool 可为空 row、旧 sparse row 或 identity row；必须覆盖 `[0,L)` | `-2` 忽略旧映射，清空整行，按当前多路 TopK union/filler 建立 C 个 resident token，并分配 slot `0..C-1` | 合法 sparse row；后续由 `-1` 维护 |
| 非卸载后直接转稳态卸载 | `-3 -> ... -> -1 -> ...` | 前期 `-3`；达到卸载阈值后，框架可释放 dense tail block，但必须保留 `[0,L)` 所需 block | `-1` probe `pool[row,C]`；若为 identity row，则仅清理 `[C,SOURCE_CAPACITY)` 为 `INT32_MIN`，保留 `[0,C)`，然后执行稳态判 miss、淘汰和 slot 更新 | 由 identity row 归一化为 sparse row |
| 非卸载后重新选择 resident token | `-3 -> ... -> -2 -> -1 -> ...` | 前期 `-3`；达到卸载阈值后传入 `-2`，无需依赖旧 pool 映射，但仍必须覆盖 `[0,L)` | `-2` 覆盖清空整行，重新按当前 TopK union/filler 选择 C 个 resident token | 新建 sparse row；后续由 `-1` 维护 |

状态职责可概括为：

- `-3`：普通 LI，完整写 identity row；不生成搬运列表。
- `-2`：首次初始化或强制重建 cache；不是“带空 slot 的稳态”，而是覆盖写整行、
  重新分配 `0..C-1` 的初始化操作。
- `-1`：维护已建立的 sparse row；正常稳态不扫描整行，仅检测到 identity row 时
  完成一次 `-3 -> -1` 归一化。

三种场景中，`L` 都必须对该请求的全部 MTP 路因果可见。相同 query、相同合法
sparse row 连续执行 `-1` 时，第二次应无 miss；query、可见 source 或 TopK 结果
变化时，允许正常产生新的 miss、淘汰和 slot 更新。

## 长序列编码

source capacity 最大为 21 bit。长度不超过 `2^17` 时，TopK/淘汰 payload 可直接
保存 source ID。超过 `2^17` 时，payload 保存 source ID 的低 17 bit，排序 key
附带高 4 bit tag；排序和候选选择后再恢复完整 21-bit source ID。

该编码只服务内部排序/候选传递：`topk_src_ids`、`miss_src_ids` 和
`cache_slots_pool` 对外始终使用完整 int32 source ID；`INT32_MIN` 无效 slot 不会
作为有效 payload 或输出 slot 打包。
