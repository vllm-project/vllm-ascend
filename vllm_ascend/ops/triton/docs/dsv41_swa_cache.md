# DeepSeek V4.1 fused SWA cache store

## Description

`dsv41_rmsnorm_rope_cache` fuses post-WKV RMSNorm, interleaved partial RoPE,
and the in-place paged SWA cache write on Ascend A2 and A3. Both devices use the
same kernel, input checks, and dispatch policy by default. The attention path
and operator tests use a single wrapper, which validates the tensors, extracts
strides, and returns `False` without writing if the input is unsupported. The
projection GEMM remains separate.

Each 512-element row is multiplied by
`rsqrt(mean(row²) + eps) * norm_weight` in FP32 and rounded to BF16 before
RoPE. Each rotary lane `i` then combines the normalized lane and its
interleaved partner `i ^ 1`: `row[i] * cos[i] + sign(i) * row[i ^ 1] * sin[i]`,
where the sign is negative for even lanes.

## Parameters

| Parameter | Role | Type and layout |
| --- | --- | --- |
| `projected_kv` | WKV projection output | BF16 `[T, 512]` |
| `norm_weight` | RMSNorm scale | Contiguous `[512]` |
| `cos`, `sin` | Per-token RoPE tables | `[T, ..., 64]` |
| `cache` | In-place output | BF16 `[blocks, block_size, 1, 512]` |
| `slot_mapping` | Physical `[block, offset]` per row | INT32 or INT64 `[T, 2]` |
| `eps`, `nope_dim` | RMSNorm epsilon and non-rotary width | Scalar parameters (`nope_dim=448`) |

## Constraints

Tensor inputs are on the same NPU. The innermost feature or coordinate
stride must be one; outer strides can include padding. Fusion supports 448
non-rotary and 64 rotary dimensions. A negative block or offset leaves the
entire cache row untouched; valid destinations must be in bounds and unique. Cache
addresses use 64-bit arithmetic, and negative coordinates use a safe row offset
before cache pointer construction. Rotary pointers only address the 64 valid
rotary lanes. Graph replay reads the updated inputs and
slot values from the captured buffers; it does not change their launch shapes.

## Origin and Differences

The original PR #21183 kernel used one program per token and reloaded the KV
and weight partners through masked 512-lane loads. Its replacement loads weights
once per program, extracts the rounded 64-lane suffix, gathers partners within
that suffix, and writes the non-rotary prefix and rotary suffix separately.

All eligible nonempty batches use the same grid-stride loop with
`min(num_tokens, get_vectorcore_num())` programs. On a 40-vector-core A2,
1024 and 4096 rows both launch 40 programs. The attention path retains the
8192-row fusion limit and uses the original three-operator path for unsupported
inputs, including replicated PCP writes with global metadata, packed caches,
and RMSNorm with a loaded bias. The kernel does not specialize on token count;
it still specializes on the compile-time dimensions, epsilon, and strides.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dsv41_swa_cache.py
python benchmarks/benchmark_dsv41_swa_triton.py --quick --json swa-quick.json
```

The operator tests compare the production wrapper with a BF16-rounded PyTorch
reference, including strided cache views, negative slots, and graph replay.
The standalone benchmark compares eager latency and cache contents against the
original three-operator path without importing vLLM. It needs the matching
Ascend custom operator library and does not measure serving latency.
