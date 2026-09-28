# copy_pcp_kv_cache

## Description

- **Location**: `vllm_ascend/ops/triton/pcp_kv_cache.py` — wrapper `copy_pcp_kv_cache`, Triton kernel `_copy_pcp_kv_cache_kernel`.
- **Function**: Packs the KV cache rows selected by `slots` into a contiguous tensor. `AscendSFAPCPImpl._sfa_preprocess_prolog_v3` uses it to gather newly written prefill KV across PCP ranks before scattering the gathered rows back to their cache slots.
- **Formula** (for output row `t` and slot `s = slots[t]`):
  - If `s < 0`, `packed[t, :] = 0`.
  - Otherwise, `block = s // cache_block_size` and `offset = s % cache_block_size`.
  - With separate caches, `packed[t] = concat(key_cache[block, offset, 0, :], rope_cache[block, offset, 0, :])`.
  - With one C8 cache, `packed[t]` is the entire stored row, copied as raw `int8` bytes. This preserves the K, RoPE, and scale payload, including FP8 bit patterns.
- **Algorithm flow**:
  1. The wrapper accepts one packed C8 cache or separate K and RoPE caches, makes `slots` contiguous, and allocates `[slots.numel(), k_dim + rope_dim]` output. An empty `slots` tensor returns an empty output without launching the kernel.
  2. It sets `BLOCK_COLS` to the next power of two of `max(k_dim, rope_dim)` and selects `BLOCK_ROWS` from the vector-core count and a conservative UB budget. The launch grid is bounded by the vector-core count.
  3. Each program iterates over its assigned slot rows, converts each valid slot to a block and offset, and loads K plus optional RoPE values. Masked loads and stores zero-fill invalid slots and exclude padded feature columns.
- **Supported modes**: One byte-sized packed C8 cache (`int8` or FP8 storage) or two matching `fp16`/`bf16` caches, on Ascend NPU.

## Parameters

> [!NOTE]
> `cache` and `slots` are required. The output is returned by the wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `cache` | Input | Tuple containing one packed C8 cache or separate K and RoPE caches. Each tensor has shape `[num_blocks, cache_block_size, 1, feature_dim]`. | C8: byte-sized `int8` / FP8; separate: matching `fp16` / `bf16` | ND |
| `slots` | Input | Slot IDs to read; `-1` denotes a zero-filled output row. Length determines `num_tokens`. | integer tensor (the tests use `int64`) | 1D |
| `packed` | Output | Selected rows in input order, shape `[num_tokens, k_dim + rope_dim]`. For C8, `rope_dim=0` and `k_dim` is the complete packed row width in bytes. | C8: `int8`; separate: same dtype as `cache[0]` | ND |

## Constraints

- `len(cache)` must be 1 or 2. Every cache tensor must be 4D with a singleton third dimension. With two caches, their dtypes and first three dimensions must match.
- A single C8 cache must have one-byte elements. The wrapper views it as `torch.int8` before copying, so the returned tensor contains raw bytes rather than decoded FP8 values.
- Every nonnegative slot must identify an allocated cache row: `0 <= slot < num_blocks * cache_block_size`. The wrapper makes a non-contiguous `slots` tensor contiguous before launch.
- `rope_dim`, `BLOCK_COLS`, and `BLOCK_ROWS` are compile-time parameters because they select the RoPE path or tile shape. Token count, cache block size, block/offset strides, and `k_dim` do not specialize on their values. The innermost feature strides retain Triton's default specialization so Ascend can lower the multi-row loads within UB.
- The single-row kernel path uses scalar slot addressing because the Ascend compiler cannot lower the modulo expression in a singleton 2D tile.

## Example

This separate-cache example copies slots `2` and `1` into output rows `0` and `2`; slot `-1` produces a zero row.

```python
import torch

from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache

k = torch.tensor(list(range(16)), dtype=torch.bfloat16, device="npu").reshape(2, 2, 1, 4)
rope = torch.tensor([100, 101, 110, 111, 120, 121, 130, 131], dtype=torch.bfloat16, device="npu")
rope = rope.reshape(2, 2, 1, 2)
slots = torch.tensor([2, -1, 1], dtype=torch.int64, device="npu")

packed = copy_pcp_kv_cache((k, rope), slots)
assert packed.shape == (3, 6)
assert packed.cpu().tolist() == [
    [8, 9, 10, 11, 120, 121],
    [0, 0, 0, 0, 0, 0],
    [4, 5, 6, 7, 110, 111],
]
```

## Origin and Differences

- **Origin**: Added for PCP SFA Prolog V3 KV synchronization in `vllm_ascend/attention/context_parallel/sfa_cp.py`.
- **Differences**: Unlike a general cache copy, this helper reads only the requested slots into contiguous rows for an all-gather. The C8 path copies the entire stored byte payload, while the separate-cache path concatenates latent K and RoPE features. Row batching is limited by an Ascend UB estimate.

## Test Cases

The NPU tests compare copied rows against an independent PyTorch reference for C8 `int8`/FP8 and separate `fp16`/`bf16` caches. They cover empty input, padded slots, multi-row batches, and non-contiguous block layouts. The unit tests cover UB-based row selection.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_pcp_kv_cache.py
pytest -q tests/ut/ops/test_pcp_kv_cache.py
```
