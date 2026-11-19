# QSA compressed-key preparation

## Description

- **Implementation**: `qsa_compress_groups_with_ratio` wraps
  `_compress_qsa_groups_kernel` in `vllm_ascend/ops/triton/qwen4_exp/qsa.py`.
- **Caller**: Qwen QSA Indexer `_update_and_compress`.
- **Algorithm**: each row pools a completed group from current keys and the
  historical ring: `pooled = sum(keys) / compress_ratio`.
  Ordinary mode returns pooled keys and the first token's positions. Supplying
  `compressed_cache` additionally computes
  `normalized = pooled * rsqrt(mean(pooled**2) + eps) * (1 + weight)`, applies
  partial MRoPE, and writes the result directly to compressed slots.
  BF16 intermediate rounding matches the decomposed path.
- **Devices**: requires NPU and Triton. The fused path has prior A3 test evidence;
  the latest revision requires fresh device validation.

## Parameters

| Parameter | Role | Shape / meaning |
| --- | --- | --- |
| `raw_keys` | Input | `[rows, 1, head_dim]` current keys |
| `raw_positions` | Input | int64 `[rows, 1, 3]` positions |
| `compressor_state_cache` | Input | `[blocks, ring_size, 1, head_dim]` historical keys |
| `compressor_state_block_table` | Input | 2D per-request ring block IDs |
| `token_to_req`, `query_start_loc` | Input | Row owners and cumulative request boundaries |
| `logical_positions`, `compressed_slots` | Input | One position and output slot per row |
| `compress_ratio` | Attribute | Positive pooling ratio; fused mode requires 4 |
| `rope_cache` | Optional input | int64 `[blocks, ring_size, 1, 3]`; absent means logical positions |
| `norm_weight`, `rotary_cache` | Optional inputs | BF16 `[128]` and `[cache_rows, 64]`; required in fused mode |
| `compressed_cache` | Optional input/output | BF16 `[pages, page_size, 1, 128]`, written in place |
| `norm_eps` | Attribute | RMS epsilon, default `1e-6` |
| Return | Output | Ordinary: `(pooled, first_positions)`; fused: `None` |

## Constraints

- Fused mode requires BF16, head dimension 128, ratio 4, rotary dimension 64,
  and interleaved MRoPE sections `[11, 11, 10]`.
- Norm weights are contiguous; the rotary table is nonempty with contiguous
  width. Weights, rotary table and output cache share the raw-key dtype/device.
  Cache strides are passed explicitly; metadata follows the model's device-local
  request/slot contract.
- Empty rows launch no kernel. Rotary positions are clamped to cache bounds;
  invalid output slots are masked. Active output slots must be unique and output
  storage must not alias the raw inputs or rings.
- The kernel only reads raw K/position rings. The caller must update them after
  compression completes on an ordered stream or equivalent dependency, including
  graph replay, so one row cannot overwrite history needed by another.

## Origin and Differences

Extends the existing QSA pooling kernel with fused Norm/MRoPE/cache stores,
avoiding pooled intermediates and separate normalization/store launches.
Raw-ring updates and downstream Indexer scoring/TopK remain outside the kernel.

## Test Cases

`test_qsa_compress_groups_with_ratio.py` compares against pooling followed by the existing
Norm/MRoPE and cache store with exact BF16 equality. Cases cover chunk sizes
1/4/7, strided rings, optional positions, invalid slots, empty rows, graph replay
and repeated ring updates. Foriv reported 9 passes on A3 at commit `5c0b6728`;
this is prior operator evidence, not a current-head or model-accuracy result.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_compress_groups_with_ratio.py
```
