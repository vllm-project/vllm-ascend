# FP8 RoPE

Source: `vllm_ascend/ops/triton/rope.py` (`rope_forward_triton`, `_triton_rope_fp8`).

## Purpose

This kernel rotates BF16 query and key tensors with NeoX-style RoPE and writes
separate fixed-scale `torch.float8_e4m3fn` outputs. MiniMax-M3 uses it when its
sparse index cache has the E4M3 dtype. The ordinary Q/K path uses
`torch_npu.npu_mrope`, and the SFA indexer path uses `torch_npu.npu_rotary_mul`.

The wrapper accepts `cos_sin_cache`, `positions`, and an explicit `rope_dim`.
Q and K must have the same token count and head width. `rope_dim` must be even
and no larger than the head width. Only NeoX style is supported. It makes
contiguous inputs and allocates contiguous E4M3 outputs unless `q_out` and
`k_out` are supplied.

For each head, the kernel rotates the first `rope_dim` elements in fp32,
passes through the remaining elements, clips all output values to `[-448, 448]`,
and converts them to E4M3 with scale 1.0. It processes each token row according
to `positions` and uses separate Q and K head tiles capped at 16.

## Validation

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py` compares
full and partial RoPE, non-power-of-two rotary dimensions, decode and prefill
token counts, and E4M3 clipping against a PyTorch reference on NPU.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py
```
