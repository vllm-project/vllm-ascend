# triton_q_rms

Normalize query rows over the last dimension. The wrapper takes a view-compatible tensor shaped `[batch, heads, dim]` and a variance epsilon. It returns a fresh tensor of the original shape and dtype. The last dimension must be at most 2048.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/rms_norm.py) · [Production caller](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/device_op.py)

## Caller-side changes (`triton_q_rms`)

The host wrapper `triton_q_rms` applies the same RMSNorm math to a `[bs, head_num, dim]` Q tensor. The tile size `BLOCK_M` is now derived adaptively in `_rms_block_m`:

```python
resv_buffer = 6144                      # 6 KB reserve to prevent UB overflow
available_ub_size = get_ub_size_bytes() - resv_buffer
element_size = torch.empty(1, dtype=dtype).element_size()   # bytes per element (2 for fp16/bf16, 4 for fp32)

# data_multiplier accounts for intermediate tensors in flight:
#   fp32:      input(4B) + offsets(4B) + mid_out(4B) + output(4B) + others(4B) = 5x
#   fp16/bf16: input(2B) + offsets(4B) + mid_out(4B) + output(2B) + others(2B) = 7x
data_multiplier = 5 if element_size == 4 else 7

row_block_size = int(available_ub_size / (dim * element_size * data_multiplier))
raw = min(row_block_size, int(batch_per_core))
BLOCK_M = 1 << (max(raw, 1).bit_length() - 1)   # largest power-of-two <= raw
```

This replaces the previous hardcoded `row_block_size = 16`, so small hidden dims get larger tiles (fewer loop iterations) and large hidden dims shrink the tile to avoid UB overflow. The UB size is auto-detected at runtime via `get_ub_size_bytes()`; `dim > 2048` raises `NotImplementedError`.

See [batch-invariant RMS normalization](rms_norm.md) for the shared kernel and its tests.
