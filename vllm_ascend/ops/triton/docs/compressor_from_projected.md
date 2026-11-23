# compressor_from_projected

## Description

- **Function**: pool ratio-2 DeepSeek V4.1 FP32 projected values and scores into
  token-aligned BF16 rows. Optionally apply RMSNorm in the pooling kernel.
  Production caller: `DeepseekV41Compressor.pool_projected`.
- **Formula**: for each completed pair and feature j, `p_j = sum_i
  softmax(scores[:,j])_i * kv_ij`; materialize `b=bf16(p)`, then optionally
  `bf16(b * rsqrt(mean(b*b)+eps) * norm_w)`.
- **Algorithm flow**: clear output; pool complete pairs using current projections,
  gathered predecessors or ring history; optionally normalize; publish only
  completion-token rows. Without gathered predecessors, update FP32 ring tails
  after pooling on the same stream. With them, the PCP caller owns later writes.
- **Supported modes**: NPU token-aligned ratio-2 path. The fusion is not tied to
  one hardware-specific arithmetic instruction. Actual A2/A3/950PR&950DT
  compilation, accuracy and production-path validation must be reported per
  applicable device; no hardware results have been produced for this proposal.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| kv, scores | Input | [tokens,width] projections | FP32 | contiguous |
| state_cache | Input/Output | [blocks,32,2*width] ring values/scores | FP32 | contiguous |
| metadata | Input | starts, lengths, output bases, input bases, block IDs | INT32 | contiguous [5,requests] |
| out | Output | caller-owned token-aligned output | BF16 | contiguous [tokens,width] |
| max_query_len | Attribute | request bound, <= tokens | int | scalar |
| num_cores | Attribute | positive grid bound | int | scalar |
| prev_rows | Input | optional gathered projections | FP32/None | contiguous [N,2*width] |
| prev_row_indices | Input | per-request predecessor indices | INT64/None | contiguous [requests] |
| norm_w | Input | optional normalization weights | BF16/FP32/None | contiguous [width] |
| rms_eps | Attribute | finite positive epsilon when norm_w is supplied | float | scalar |

## Constraints

- All tensors share one device; nonempty kernel work requires NPU. Width is a
  positive power of two; state rows are 32. Bounds must fit the provided storage.
- prev_rows and prev_row_indices must be supplied together, or both be None.
  Block 0 is reserved; padding requests use zero length and zero block ID.
- Inputs/weights are finite, and controls/predecessor indices are valid.
  The wrapper validates tensor metadata without reading contents on host.
- out is deliberately zeroed; incomplete-pair and padded rows stay zero.
  Inputs do not alias out/state. State mutation follows the ordering above.
- Loaded normalization bias is handled by the model's original separate norm
  path. The BF16 pooling boundary before RMSNorm must not be removed.
- HEAD_DIM and algorithm/branch flags remain constexpr. Pooling epsilon and
  cache block bounds become runtime values; scheduler scalars suppress value
  specialization. FUSE_TOKEN_NORM/HAS_PREV_ROWS and weight pointer dtypes still
  distinguish required variants. This patch does not change the separate
  `_cache_update_kernel` specialization policy.
- Graph capture/replay and remaining variants must be checked on the target
  runtime. New warmup integration is not included; register actual required
  variants using the existing shared registry if measurement shows uncovered
  serving-time compilation.

## Origin and Differences

- **Origin**: existing token-aligned `_pool_kernel` and projected compressor.
- **Differences**: optional RMSNorm after the original BF16 pooling round-trip;
  no change to FP32 ring-state ordering or PCP ownership.
- **Accuracy reference**: existing `norm_w=None` pooling followed by native
  NPU RMSNorm. BF16 output uses rtol=atol=2^-6, >=99% passing combined
  tolerance and max absolute error<=1. State, inputs and padding require exact
  equality where semantics are unchanged. The original pooling implementation's
  independent reference tests must also be retained.

## Test Cases

```bash
VLLM_ASCEND_DSV41_TEST_CONFIG=/path/to/deployed/model/config.json \
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_k_fusion.py -k c2
python -m pytest tests/ut/ops/test_indexer_k_fusion.py tests/ut/models/test_deepseek_v41_cache.py tests/ut/attention/test_dsa_v41_cp.py -q
```

Use actual head_dim and epsilon from the deployed model. Device cases include
complete/incomplete pairs, odd start position, ring wrap and gathered predecessor
paths. Also retain full production PCP tests proving residual reads happen before
the caller's ring write, bias fallback and exactly-once normalization.

## Example

```python
pooled = compressor_from_projected(
    kv, scores, state, metadata, output,
    max_query_len=max_query_len, num_cores=num_cores,
    prev_rows=prev_rows, prev_row_indices=prev_row_indices,
    norm_w=norm.weight, rms_eps=norm.variance_epsilon,
)
```
