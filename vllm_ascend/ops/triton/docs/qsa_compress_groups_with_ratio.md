# QSA compressed-key pooling, normalization and cache store

## Description

- **Location**: `vllm_ascend/ops/triton/qwen4_exp/qsa.py`.
- **Function**: public wrapper `qsa_compress_groups_with_ratio` pools raw Indexer
  keys. Supplying `compressed_cache` enables normalization, MRoPE and direct
  cache publication in the same kernel.
- **Caller**: `AscendQSAIndexer._update_and_compress` in the Qwen4Exp model.
  This is model-specific preparation, not Indexer scoring or main attention.
- **Formula**: `pooled = sum(group_keys) / compress_ratio`;
  `normalized = pooled * rsqrt(mean(pooled**2) + eps) * (1 + weight)`;
  apply partial MRoPE to the first 64 dimensions, then store by compressed slot.
  Pooling, normalization, rotary products and final add/sub preserve the original
  BF16 materialization boundaries.
- **Algorithm flow**: one program per input row reads a completed group's keys
  from current inputs or the historical ring, obtains the first token's position,
  pools the keys, and either returns pooled values/positions or computes and
  publishes the fused result.
- **Supported modes**: NPU with Triton; new fused-path tests target A3. Final-head
  hardware execution and other applicable device validation remain pending.

## Parameters

| Parameter | Role | Type/layout and meaning |
| --- | --- | --- |
| `raw_keys` | Input | `[rows, 1, head_dim]`, contiguous last dimension; token stride may include padding |
| `raw_positions` | Input | int64 `[rows, 1, 3]` current position rows |
| `compressor_state_cache` | Input | Historical raw K ring `[blocks, ring_size, 1, head_dim]` |
| `compressor_state_block_table` | Input | Per-request ring block IDs, 2D |
| `token_to_req`, `query_start_loc` | Input | Row owners and request cumulative query boundaries |
| `logical_positions`, `compressed_slots` | Input | One logical position and destination slot per row; negative slots are inactive |
| `compress_ratio` | Attribute | Positive pooling ratio; fused mode requires 4 |
| `rope_cache` | Optional input | int64 historical positions `[blocks, ring_size, 1, 3]`; absent means use logical positions |
| `norm_weight`, `rotary_cache` | Optional inputs | Required together with output cache: BF16 `[128]` and `[cache_rows, 64]` |
| `compressed_cache` | Optional input/output | BF16 `[pages, page_size, 1, 128]`, updated in place |
| `norm_eps` | Attribute | RMS epsilon, default `1e-6` |
| Return | Output | Ordinary mode: `(pooled, first_positions)`; fused mode: `None` |

## Constraints

The combined ABI is BF16, head dimension 128, ratio 4, rotary dimension 64 and
interleaved MRoPE sections `[11, 11, 10]`. Norm weights must be contiguous. The
rotary table is nonempty with contiguous rotary width. Norm weights, rotary
table and output cache share the raw-key device and dtype. Cache strides are
passed explicitly; padded row layouts are supported. Metadata is device-local
and follows the model's request/slot mapping contract. This wrapper is not a
general validator of arbitrary metadata contents or overlapping storage.

The model uses a fixed compression ratio of 4 and selects the combined path
using `_fused_norm_rope_eligible`. When this predicate is false, it uses the
decomposed path; malformed inputs can still be rejected by either path.
The public wrapper rejects unsupported fused arguments rather than silently
falling back. Empty rows produce no launch. Rotary positions are clamped; output
slots outside cache capacity are masked. Valid active rows must not race on the
same output slot, and compressed outputs must not alias raw inputs or rings.

**Side effects and ordering**: only compressed output slots are written in the
fused launch. Raw K and position rings remain read-only. The caller must enqueue
their updates after all compression readers finish, on an ordered stream (or
with an equivalent explicit dependency). Moving ring writes into independently
scheduled row programs can overwrite history still required by another row.
NPUGraph replay reads current device metadata and preserves this ordering.

## Origin and Differences

Extends the existing QSA pooling kernel from PR #21957 and its source work.
Ordinary mode still materializes pooled values and group positions. Fused mode
avoids those allocations and the separate normalization/store stages. It does
not update the raw ring or change the downstream TopK algorithm.

## Test Cases

`test_qsa_compress_norm_store.py` compares with pooling followed by the existing
Norm/MRoPE operator and cache store, using exact BF16 comparison. It covers
three requests, chunks 1/4/7, position-cache presence, strided storage, inactive
and out-of-range slots, empty rows, graph replay and a 12-step ring transaction.
This checks decomposition equivalence, not independent model accuracy.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_compress_norm_store.py
```
