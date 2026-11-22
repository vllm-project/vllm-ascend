# Step5 _append_current_region (GBSA id pack)

## Description

- **Function**: Packs the indexer's top-k region ids into the layout the
  aclnn `GenericBlockSparseAttention` operator consumes: copies each row's
  `[topk]` selected ids into a `topk + 1` wide tensor, counts the live
  (non-`-1`) ids, appends the row's current region
  `floor(position / region_size)` at slot `count`, and emits the per-row
  block counts (`live + 1`) for the operator's scheduler. Count + copy +
  append run as one fused Triton kernel. Called on every DSA layer forward
  inside `gbsa_sparse_attention`
  (`vllm_ascend/models/step5/ops/generic_block_sparse_attention.py`), which
  is the exact-attention replacement for the previous pack + Triton chain.
- **Location**: `vllm_ascend/ops/triton/step5/summary_update.py`
  — kernel `_append_current_region_kernel`, wrapper `append_current_region`.
- **Formula** (per row `(group, token)`, rows group-major
  `r = group * tokens + token`):
    - `live = |{ c : ids[r, c] != -1 }|`
    - `wide[r, 0:topk] = ids[r, 0:topk]`
    - `wide[r, topk] = -1`
    - `wide[r, min(live, topk)] = positions[token] // region_size`
    - `counts[r] = live + 1`
    Because the selection pool is strictly below the current region, the
    appended id is larger than every selected id and the row stays ascending;
    a full top-k selection plus the current region fits without evicting a
    selected id (the `topk`-th column is the append slot when full).
- **Algorithm flow**:
    1. Grid `(groups, cdiv(tokens, block_t))`; `block_k =
       next_power_of_2(min(topk, 512))`, `block_t = max(1, min(64, 8192 /
       block_k))` — a `(block_t, block_k)` tile per program.
    2. Stream the id row in `block_k` chunks: copy the chunk and accumulate
       the live count.
    3. Seed the append column with `-1`, then write the current region at
       `min(live, topk)` and the count.
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend; execution
  evidence on Ascend950DT), decode and prefill forwards (it is called inside the DSA
  layer's attention path on every step).

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `ids` | Input | `indexer_score_topk` output, `[groups * tokens, topk]` logical region ids, ascending, `-1` tail padding, current region not included | int32 | ND 2-D contiguous |
| `positions` | Input | Absolute position per token, shared across groups (`[tokens]`) | int32/int64 | ND (1-D) |
| `groups` | Attribute | Number of KV head groups G (`rows = groups * tokens`) | int | - |
| `region_size` | Attribute | Tokens per region (Step5: 8) | int | - |
| return `wide` | Output | `[groups, tokens, topk + 1]` packed ids, ascending per row with the current region appended | int32 | ND 3-D, freshly allocated |
| return `counts` | Output | `[groups, tokens]` live+current block counts | int32 | ND 2-D, freshly allocated |

## Constraints

- `ids` rows must be ascending with `-1` only as tail padding (the
  `IndexerRegionScore` `sort_mode=1` contract); `live` would be
  miss-computed otherwise (any `-1` counts as dead).
- The selection pool must be strictly below `floor(position / region_size)`
  (guaranteed by the fused indexer's causal top-k) — otherwise the appended
  current region could break the row's ascending order.
- `ids.shape[0]` must equal `groups * tokens`; `topk <= 1024` (Step5 DSA
  limit); the kernel streams `topk` in power-of-two chunks
  (`block_k <= 512`), so any supported `topk` is one variant shape.
- `positions[token] // region_size` must fit int32 (logical region ids).
- Empty inputs (`rows == 0`) return empty tensors; a zero grid dimension
  launches nothing.
- Specialization: `topk`, `block_k`, `block_t` are constexpr (fixed per
  model config); `tokens`, `region_size` are plain runtime integers. The
  integer specialization classes of `tokens` (the `== 1` decode case and the
  `% 16` classes) and `region_size` (Step5: 8, one class) bound the serving
  variants to a small constant set per model config.
- Determinism: integer copy/append only — output is exact and is compared
  bit-exact in tests. No in-place behavior; both outputs are freshly
  allocated.

## Origin and Differences

- **Origin**: Ascend adaptation of the current-region append that this
  repo's earlier Triton pack kernels (`decode_pack_from_ids` /
  `prefill_pack_from_ids`, since removed in favor of the aclnn operator)
  performed for the Triton sparse-attention chain, repurposed as the id
  front-end for the aclnn `GenericBlockSparseAttention` operator (which
  takes `topk + 1` wide ids and per-row counts directly).
- **Differences**: the earlier Ascend Triton path packed per-region `valid`
  counts for its own Triton attention kernels; here the count is the
  operator's
  `sparse_block_count` (`live + 1`) and the causal truncation moved into the
  operator (`mask_mode=CAUSAL` with right-aligned varlen), so this kernel is
  a pure integer pack. The AICPU scheduling metadata is built separately
  once per step (`gbsa_sparse_attention_metadata`) from expected counts —
  this kernel's counts are what those expected counts must match.

## Test Cases

Tests compare `wide` and `counts` exactly (integer semantics) against a
pure-torch reference. Cases cover the real Step5 shape (`topk = 512`,
`region_size = 8`) and smaller topk values, full selections (`live ==
topk`), partial selections, empty selections (`live == 0`, current region at
slot 0), multiple groups, single-token (decode) batches, `tokens` around the
`block_t` tile boundary, and large random positions.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_append_current_region.py
```
