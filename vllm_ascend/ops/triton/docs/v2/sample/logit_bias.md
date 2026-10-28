# Logit bias and minimum-token masking (`_bias_kernel`)

## Description

- **Function**: applies an allowed-token restriction, additive logit biases,
  and minimum-length stop-token masking to each MRv2 logit row in place.
- **Entry point**: the upstream
  `vllm.v1.worker.gpu.sample.logit_bias.apply_logit_bias` wrapper remains in
  use. `vllm_ascend/patch/worker/patch_v2/patch_triton.py` rebinds its module's
  `_bias_kernel` to `vllm_ascend/ops/triton/v2/sample/logit_bias.py` during
  worker initialization. Existing imported wrapper aliases use the same
  module globals and therefore also resolve to the Ascend kernel.
- **Supported modes**: Ascend NPU MRv2 sampling, including structured-output
  all-masked-row recovery. The regression was validated on Atlas A3; this
  change does not establish additional hardware or graph-mode coverage.

For logit row `t`, request-state index `r = expanded_idx_mapping[t]`, and token
`i`, the operations execute in this order:

```text
if num_allowed_token_ids[r] > 0:
    preserve logits only at allowed_token_ids[r]; set other entries to -inf

logits[t, bias_token_ids[r, j]] += bias[r, j]
    for j < num_logit_bias[r]

if num_stop_token_ids[r] > 0 and pos[t] + 1 < min_lens[r]:
    save stop logits when recovery is enabled
    set stop logits to -inf
    if CHECK_ALL_MASKED_ROWS and restore_when_all_masked[r]:
        if max(logits[t, :]) == -inf:
            restore finite saved stop logits; leave other stop logits at -inf
```

**Algorithm flow**: one Triton program owns a logit row. It saves allowed
logits before clearing the row, applies sparse biases, and then processes
stop tokens. When recovery is enabled, it scans the vocabulary in
`LOGITS_BLOCK_SIZE` tiles to determine whether stop masking removed the last
usable logits. Existing `tl.debug_barrier()` calls preserve the ordering of
loads, overwrites, and scans within that program.

## Parameters

All tensor pointers refer to NPU tensors. `R` denotes request-state capacity;
`T` denotes the number of logit rows. Strides are measured in elements.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input/Output | Logits updated in place by the upstream wrapper. | fp32 | `[T, vocab_size]`, contiguous vocabulary dimension |
| `logits_stride` | Attribute | Row stride of logits. | integer | scalar |
| `vocab_size` | Attribute | Number of vocabulary entries. | integer | scalar |
| `expanded_idx_mapping_ptr` | Input | Request-state index for each logit row. | integer, int64 in regression | `[T]` |
| `num_allowed_token_ids_ptr` | Input | Allowed-token count; zero disables restriction. | int32 | `[R]` |
| `allowed_token_ids_ptr` | Input | Allowed vocabulary IDs. | int32 | `[R, allowed_capacity]` |
| `allowed_token_ids_stride` | Attribute | Allowed-ID row stride. | integer | scalar |
| `num_logit_bias_ptr` | Input | Number of additive biases. | int32 | `[R]` |
| `bias_token_ids_ptr` | Input | Vocabulary IDs receiving biases. | int32 | `[R, bias_capacity]` |
| `bias_token_ids_stride` | Attribute | Bias-ID row stride. | integer | scalar |
| `bias_ptr` | Input | Additive bias values. | fp32 | `[R, bias_capacity]` |
| `bias_stride` | Attribute | Bias-value row stride. | integer | scalar |
| `pos_ptr` | Input | Token position associated with each logit row. | integer, int64 in regression | `[T]` |
| `min_lens_ptr` | Input | Per-request minimum length. | int32 | `[R]` |
| `num_stop_token_ids_ptr` | Input | Number of stop-token IDs. | int32 | `[R]` |
| `restore_when_all_masked_ptr` | Input | Per-request flag enabling finite stop-logit recovery. | int32 flag | `[R]` |
| `stop_token_ids_ptr` | Input | Stop vocabulary IDs. | int32 | `[R, stop_capacity]` |
| `stop_token_ids_stride` | Attribute | Stop-ID row stride. | integer | scalar |
| `BLOCK_SIZE` | Attribute | Width of the sparse-ID processing tile. | constexpr integer | scalar |
| `LOGITS_BLOCK_SIZE` | Attribute | Width of each full-vocabulary scan/clear tile. | constexpr integer | scalar |
| `CHECK_ALL_MASKED_ROWS` | Attribute | Compile-time switch for recovery checks. | constexpr bool | scalar |

The public wrapper takes the corresponding tensors without the `_ptr` suffix,
passes their row strides, selects the tile sizes, and launches one program per
logit row. It returns no tensor; logits are modified in place.

## Constraints

- Logits and metadata must use the same device. All ID lists are contiguous
  within a row; request-state indices and vocabulary IDs must be in range.
- Counts must fit their list capacities and `BLOCK_SIZE`. Sparse ID lists
  must not contain duplicate destinations within a row; programs must own
  distinct logit rows.
- Minimum-length masking applies only while `pos[t] + 1 < min_lens[r]`.
- Recovery requires both the compile-time switch and the per-request flag.
  If either is disabled, stop logits stay masked until the minimum length.
- Recovery uses the upstream `row_max == -inf` condition, not a general
  check for all non-finite values. Saved NaN, `+inf`, and `-inf` stop logits
  are not restored. This change does not redefine exceptional-value behavior
  elsewhere in the row.
- No host synchronization or cross-stream barrier is added. The existing
  device-local barriers are retained. Graph-mode acceptance is separate from
  the eager stream-overlap regression below.
- The operator is inference-only; backward support is N/A.

## Origin and Differences

- **Origin**: upstream MRv2 `_bias_kernel` in
  `vllm.v1.worker.gpu.sample.logit_bias`.
- **Recovery-store adaptation**: upstream combines the stop-count mask and
  the data-dependent finite-value predicate in the scatter-store mask.
  Ascend selects the value with
  `tl.where(isfinite(saved_stop_logit), saved_stop_logit, -inf)` and keeps
  only the stop-count prefix in the store mask. Non-finite saved values
  were already overwritten with `-inf`, so writing `-inf` again preserves
  the result.
- Affected Triton-Ascend versions generate an unordered lock workspace for
  the former recovery pattern. The rewrite avoids that lowering in the
  tested specializations; it keeps the existing allowed-token, additive-bias,
  and minimum-length logic. See
  [Triton-Ascend #2817](https://github.com/triton-lang/triton-ascend/pull/2817).
- The fix is confined to vLLM Ascend. It requires no upstream vLLM source or
  Triton launcher changes and does not serialize asynchronous execution.
  The paired [grammar-bitmask rewrite](../apply_grammar_bitmask.md) avoids
  the corresponding workspace in grammar processing.
- The worker override can be removed once the supported Triton-Ascend
  launcher safely initializes unordered-lock workspaces and the original
  upstream path passes the same accuracy and overlap regressions.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_logit_bias.py
```

The tests import the upstream wrapper before worker patch registration and
verify that its kernel binding changes. They compare every output element
against a CPU expectation using `rtol=0`, `atol=0`, and `equal_nan=True` for:

- recovery enabled and disabled through `check_all_masked_rows`;
- finite, infinite, and NaN saved stop logits;
- a remaining finite non-stop token, which prevents recovery;
- a disabled per-request recovery flag;
- a request that has reached its minimum length.

The overlap test additionally runs 20 grammar-to-bias iterations with separate
mask-copy and main streams, using 18 rows and vocabulary size 129280. It
checks the complete grammar and bias outputs after one final synchronization.
These tests focus on the changed recovery and overlap paths; they do not
claim new coverage of nonzero allowed-token lists or additive biases.
