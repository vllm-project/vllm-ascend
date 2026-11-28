# Strided MLA prefill gather

## Description

- **Function** gathers cached MLA history for chunked prefill and writes the latent and rotary components to separate contiguous TND buffers. The shared MLA prefill path in `vllm_ascend/attention/mla_v1.py` calls `try_gather_strided_prefill` before projecting the gathered history.
- **Formula** maps token `j` of request `r` to logical token `starts[r] + j`. The page table maps its page index to a physical page, and the token's offset within the page selects the source row. The destination row is `cumulative[r] + j` in both outputs.
- **Algorithm flow**
    1. Read each request's chunk boundaries and starting position from device metadata.
    2. Align the first tile to an 8-token boundary. Distribute the tiles across vector cores so each tile stays within a single 128-token page.
    3. Load both components through their respective page and token strides. Zero rows that reference an invalid physical page and mask writes outside the requested token range.
- **Supported modes** include Atlas A3 inference with component-major or otherwise strided MLA caches. This path is used for prefill. Full decode graph replay uses a separate operator.

## Parameters

| Parameter | Input/Output | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key_cache` | Input | Latent cache with shape `[num_pages, 128, 1, 512]` | BF16 or FP16 | Strided ND |
| `rope_cache` | Input | Rotary cache with shape `[num_pages, 128, 1, 64]` | Same as `key_cache` | Strided ND |
| `page_table` | Input | Physical page IDs per request, with at least `num_requests` rows | Int32 | 2D, contiguous within each row |
| `cumulative` | Input | Cumulative output token counts with shape `[num_requests + 1]` | Int32 | Contiguous |
| `starts` | Input | Starting token in the cached history of each request, with shape `[num_requests]` | Int32 | Contiguous |
| `key` | Output | Preallocated latent history with shape `[num_tokens, 1, 512]` | Same as `key_cache` | Contiguous TND |
| `rope` | Output | Preallocated rotary history with shape `[num_tokens, 1, 64]` | Same as `key_cache` | Contiguous TND |
| Return value | Output | `True` if the wrapper handled the gather, including an empty output. `False` selects the caller's existing cache reader. | Boolean | Scalar |

## Constraints

- All tensors must reside on the same NPU. The cache components must use NPU format 0 or 2 and have unit stride in their last dimension. At least one cache component must be noncontiguous.
- `cumulative` starts at zero, is nondecreasing, and ends at `num_tokens`. Each start is nonnegative. Nonempty input spans require at least one physical cache page and must fit within their logical page table. Physical page IDs outside `[0, num_pages)` produce zero rows.
- The wrapper writes the two output buffers in place. Give both output buffers independent storage. A supported call with zero output tokens returns without launching a kernel.
- Cache row strides, tile size, page size, and component widths are compile-time constants. Request count, page strides, and page-table row stride are runtime arguments and can specialize under Triton's default rules. Triton reuses the compiled variant when only page IDs, token starts, or cumulative lengths in the existing buffers change. The first eligible prefill call can compile the operator.

## Origin and Differences

This operator is implemented for the Ascend MLA prefill cache-loading path. It consumes the caller's existing device-side chunk boundaries and writes directly from strided pages into the two contiguous outputs. The caller uses `DeviceOperator.kv_cache_load` when the wrapper reports an unsupported input.

## Test Cases

The test uses Ling's 128-token pages and 512/64 component widths. It compares both outputs exactly with a PyTorch CPU reference for BF16 and FP16. The inputs include unaligned starts, shuffled physical pages, ragged chunks, and invalid page IDs. Hardware validation uses the A3 nightly operator suite.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_strided_mla_prefill_gather.py
```
