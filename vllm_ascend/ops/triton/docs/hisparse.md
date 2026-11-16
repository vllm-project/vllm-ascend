# HiSparse transfer, layout and invalidation operations

## Description

- **Function**: Move packed KV rows between mapped Host memory, Resident/Hot
  device caches and Attention staging; invalidate stale Hot-cache tags. Residency
  selection and LRU planning belong to the separate AscendC resolver.
- **Formula**: A paged row `r` starts at
  `base + (r // block_size) * block_stride + (r % block_size) * row_bytes`.
  Refill copies a planned Host row to its Hot destination. Invalidation replaces
  a matching cached Host-row tag with `-1`, without clearing the KV payload.
- **Algorithm flow**: The common HiSparse runtime supplies the page tables and
  plans. `HiSparseNPUOps` validates them and launches the kernels below on the
  current stream. Completion dependencies remain owned by the runtime/connector.
- **Supported modes**: Ascend 950PR/950DT with the qualified CANN 9.2 stack.
  Atlas A2/A3 are not qualified by this adapter. Refill, layout and written-slot
  invalidation have eager and ACLGraph test cases. Recycled-block invalidation
  runs outside capture and uses an initialization-time sentinel warmup.

Source: `vllm_ascend/ops/triton/hisparse.py`. Launch wrappers are in
`vllm_ascend/hisparse/cache_ops.py` and
`vllm_ascend/hisparse/recycled_invalidate.py`.

## Parameters

All pointer parameters are device tensors unless explicitly marked as a mapped
Host address. Dimensions and strides are nonnegative integer attributes, with
positive block sizes and row widths. Byte strides and element strides are
distinguished below; the Python wrappers supply the launch geometry.

### `gather_rows_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `host_address` | Input | Device-accessible base address of registered Host KV | Mapped pointer to UINT8 | Contiguous Host rows |
| `hot` | Output | Destination device KV storage | UINT8 | Paged, dense rows |
| `source`, `destination` | Input | Source Host and destination device row IDs | INT32 | Contiguous `[queries, top_k]` |
| `counts` | Input | Compact plan length per query; used when `PARALLEL_COLUMNS > 0` | INT32 | `[queries]` |
| `mask_ptr` | Input | Select plan entries in the noncompact path | BOOL, UINT8 or INT32 | Same shape as `source` |
| `attention` | Output | Optional mapped Attention row IDs | INT32 | Same shape as `source` |
| `state` | Input | Optional request-state IDs; negative means padded query | INT32 | `[queries]` |
| `HOST_ROWS`, `HOT_ROWS`, `TOP_K` | Attribute | Source capacity, destination capacity and plan width | Integer | Scalar |
| `ROW_BYTES`, `HOT_BLOCK`, `HOT_STRIDE_BYTES` | Attribute | Bytes per row, rows per page and physical page stride in bytes | Integer | Scalar |
| `ATTENTION_BLOCK`, `HAS_ATTENTION`, `HAS_STATE` | Attribute | Attention page stride in rows and optional-input flags | Integer / boolean | Scalar |
| `BLOCK`, `PARALLEL_COLUMNS` | Attribute | Byte tile and compact-plan parallelism; zero selects mask mode | Integer | Scalar |

### `invalidate_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tables` | Input | Pointers to each group's Hot tag table | INT64 | `[groups]` |
| `states` | Input | Request-to-state mapping | INT32 | `[requests]` |
| `requests`, `written` | Input | Request ID and updated Host row ID per token | INT32; INT32 or INT64 | `[tokens]` |
| Tables addressed by `tables` | Input/Output | Invalidate entries matching an updated row in its state | INT32 | Contiguous `[states, hot_capacity]` |
| `REQUESTS`, `STATE_ROWS`, `REGION`, `BLOCK` | Attribute | Request count, state count, Hot capacity and scan tile | Integer | Scalar |

### `pack_written_slot_keys` and `invalidate_written_slots_sorted`

The large-write route packs `(state << 31) | host_row` keys, sorts them and
binary-searches those keys while scanning each Hot table. Invalid writes use
the maximum INT64 sentinel. Sorting is performed by the Python wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `states`, `requests`, `written` | Input | Request mapping and updated rows, as above | INT32; `written` INT32 or INT64 | Contiguous 1D |
| `keys` in `pack_written_slot_keys` | Output | Packed keys before sorting | INT64 | `[ROWS]` |
| `keys` in `invalidate_written_slots_sorted` | Input | Sorted packed keys | INT64 | `[ROWS]` |
| `tables` | Input | Device pointers to Hot tag tables | INT64 | `[groups]` |
| Tables addressed by `tables` | Input/Output | Matching entries are replaced with `-1` | INT32 | Contiguous `[states, HOT]` |
| `ROWS` | Attribute | Number of write keys | Integer | Scalar |
| `REQUESTS`, `STATES` | Attribute | Packing-path request and state bounds | Integer | Scalar |
| `TILE`, `HOT`, `STEPS` | Attribute | Scan tile, Hot capacity and binary-search iteration count | Integer | Scalar |

### `pack_attention_rows_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `cache` | Input | Packed `[latent, RoPE]` KV rows | BF16 in the qualified SFA path | Paged, dense rows |
| `indices` | Input | Selected physical KV row IDs | INT32 | `[queries, K]`, dense columns |
| `latent`, `rope` | Output | Selected components, with invalid selections zero-filled | Same as `cache` | Dense `[queries, K, component_width]` |
| `K`, `CACHE_ROWS`, `CACHE_BLOCK` | Attribute | Selection width, cache capacity and page rows | Integer | Scalar |
| `CACHE_STRIDE`, `WIDTH`, `LATENT` | Attribute | Physical page stride in elements, packed width and latent width | Integer | Scalar |
| `INDEX_STRIDE`, `BLOCK` | Attribute | Query stride in indices and row tile | Integer | Scalar |

### `scatter_packed_kv_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `latent`, `rope` | Input | Components to concatenate into the destination row | Matching payload dtype; BF16 or UINT8 views for C8 | Dense trailing dimensions |
| `cache` | Output | Current layer's packed KV cache | Same as inputs | Paged, dense rows |
| `slots` | Input | Destination physical row IDs; negative/out-of-range IDs skip writes | INT32 or INT64 | `[tokens]` |
| `valid_rows` | Input | Optional live-token prefix length | INT32 or INT64 | One device element |
| `CACHE_ROWS`, `CACHE_BLOCK`, `CACHE_STRIDE` | Attribute | Capacity, rows per page and page stride in elements | Integer | Scalar |
| `WIDTH`, `LATENT` | Attribute | Packed and first-component widths | Integer | Scalar |
| `LATENT_STRIDE`, `ROPE_STRIDE` | Attribute | Input row strides in elements | Integer | Scalar |
| `HAS_VALID`, `BLOCK` | Attribute | Whether to apply the valid prefix and row tile | Boolean / integer | Scalar |

### `gather_resident_rows_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `source`, `destination` | Input / Output | Resident cache and independent staging, copied without conversion | UINT8 aliases | Paged, dense byte rows |
| `row_ids` | Input | Source row for each staging row | INT32 or INT64 | `[ROWS]` |
| `ROWS`, `SOURCE_ROWS` | Attribute | Output row count and source capacity | Integer | Scalar |
| `SOURCE_BLOCK`, `DEST_BLOCK` | Attribute | Source and destination rows per page | Integer | Scalar |
| `SOURCE_STRIDE`, `DEST_STRIDE` | Attribute | Physical page strides in bytes | Integer | Scalar |
| `ROW_BYTES`, `BLOCK_ROWS`, `BLOCK_BYTES` | Attribute | Payload bytes and launch tiles | Integer | Scalar |

### `_invalidate_recycled_groups`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tables` | Input | Pointers to nonoverlapping live group tag tables | INT64 | `[groups]` |
| `blocks` | Input | Sorted recycled Host block IDs, padded by repeating the last ID | INT32 | `[BLOCK_COUNT]` |
| `state_indices` | Input | Live scheduler state indices; negative sentinel masks warmup | INT64 | Contiguous 1D |
| Tables addressed by `tables` | Input/Output | Tags whose `tag // PAGE_ROWS` is recycled become `-1` | INT32 | Contiguous `[STATES, HOT]` |
| `STATES`, `HOT`, `PAGE_ROWS` | Attribute | State count, Hot capacity and Host block rows | Integer | Scalar |
| `BLOCK_COUNT`, `ITERATIONS`, `TILE` | Attribute | Eight-ID bucket, search iterations and tag scan tile | Integer | Scalar |

## Constraints

- Inputs must satisfy the Python wrappers' device, shape, dtype, stride and
  lifetime checks. Mapped Host storage must remain registered through all
  dependent work. Device row IDs and stored Hot tags must fit INT32.
- Refill preserves payload bytes. Compact counts are clamped to `[0, top_k]`;
  entries beyond the current count are not read. Negative source/destination
  IDs skip a copy. A nonnegative Host ID beyond capacity writes a zero row.
  Padded request states skip copies and produce `-1` Attention indices.
- Written-slot invalidation changes tags only. The sorted route treats writes
  outside INT32 range as nonmatching sentinels. The wrapper selects direct or
  sorted invalidation from the actual geometry and device capability.
- Scatter requires unique valid destinations within one launch. Padding and
  rows beyond the optional valid prefix must leave other storage unchanged.
- Resident gather follows Core's negative-ID-to-row-zero convention; it is not
  a skip convention. IDs above the source capacity leave staging unchanged.
  Source and destination must have matching row byte widths and separate storage.
- Recycled invalidation supports the bound A5 tables and stream only. Its
  initialization launch uses state `-1`, so no payload/tag memory is modified.
  Unsupported geometry, changed owners/stream, capture or disabled warmup retain
  the upstream worker path. Recycled block lists are deduplicated and split into
  eight-ID buckets before launch.
- These operations do not change quantization or Attention arithmetic. Pure
  copies and tag updates are tested exactly (`rtol=0`, `atol=0`), including
  neighboring bytes, page gaps and invalid rows.

## Origin and Differences

- **Origin**: Ascend implementations of the merged Core HiSparse cache-operation
  interfaces and its Resident staging / written-slot / recycled-block semantics.
- **Differences**: Host refill uses ACL-registered mapped addresses rather than
  CUDA pointers. Explicit source/destination page strides preserve shared HMA
  layouts. Resident gathering and packed writes use byte-compatible views.
  Ascend batches recycled-tag invalidation across groups and retains the original
  worker method when its qualified path is unavailable. No Triton residency
  resolver is provided; that operation is implemented in AscendC.

## Test Cases

The operator suite covers compact and masked refill, page gaps, padding, changing
Graph inputs, direct/sorted written-slot invalidation, packed KV write/read,
Resident BF16-576 and C8-656 rows, and recycled-block bucket boundaries.
Native resolver/DMA and direct-mirror integration cases remain separately in
`tests/e2e/pull_request/one_card/test_hisparse_integration.py`; they are not Triton-only
operator tests.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_hisparse.py
pytest -sv tests/ut/model_executor/warmup/test_hisparse_recycled_invalidate.py
```

Warmup unit tests use CPU contract doubles for eligibility, the no-write sentinel,
bucket padding and fallback checks. They do not establish NPU correctness or
hardware compilation coverage. Execution status belongs in the PR description;
adding a case is not a claim that it has passed on hardware.

## Example

After Core has bound the runtime, its operator object consumes the resolver's
compact miss plan. The caller retains the mapping, buffers and stream ordering:

```python
runtime.cache_ops.hisparse_gather_compact(
    runtime.host_cache,
    runtime.hot.cache,
    miss_host_rows,
    miss_device_rows,
    miss_counts,
)
```
