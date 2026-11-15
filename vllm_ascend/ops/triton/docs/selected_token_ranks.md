# selected_token_ranks

> Source: `vllm_ascend/worker/v2/sample/logprob.py` (`_ranks_kernel`, called by `compute_topk_logprobs`).

## Description

- **Function**: Computes one vocabulary rank for the selected token in each logits row. The rank counts real vocabulary entries whose logits are greater than or equal to the selected logit. A unique maximum ranks 1; both members of a two-way top tie rank 2. This operator does not select tokens or compute log probabilities. For log-probability calculation, see [topk_log_softmax](topk_log_softmax.md).
- **Formula**: For row `i`, vocabulary size `V`, logits `L`, and selected token ID `t_i`:
    - `x_i = L[i, t_i]`
    - `rank_i = sum(1[L[i, j] >= x_i] for j in range(V))`

  As in the upstream kernel, out-of-range lanes load negative infinity and are not explicitly excluded from the comparison. For a selected logit other than negative infinity, they do not contribute. For a selected negative-infinity logit, they are counted and the result can exceed `V`; correcting this boundary is outside this PR's scope.

- **Algorithm flow** (processed row by row, independently):
  1. The wrapper obtains the device Vector Core count, sets `P = min(batch_size, vector_cores)`, and launches a one-dimensional grid `(P,)` with `rows_per_core = ceil(batch_size / P)`.
  2. Program `core_id` processes rows from `core_id * rows_per_core` to `(core_id + 1) * rows_per_core`, skipping rows outside the batch.
  3. For each valid row, load the selected token ID and its logit `x_i`. Initialize an int32 counting vector of `BLOCK_SIZE` lanes.
  4. Scan the vocabulary in fixed `BLOCK_SIZE = 8192` tiles. Load real logits and substitute negative infinity outside the vocabulary, then accumulate `(logits >= x_i)` into the counting vector.
  5. Reduce the counting vector to a scalar and store the rank as int64.
- **Supported modes**: The operator is intended for Atlas A2, Atlas A3, and 950PR&950DT Products in the V2 score implementation. Single-operator tests run eagerly. End-to-end model ACL Graph and MTP coverage is not established by these tests.

  On the paired vLLM `24420b7a` source, standard MRV2 callers invoke `compute_topk_scores`, while the existing Ascend patch assigns the retired `compute_topk_logprobs` name. This rank-only change does not repair that binding. Direct calls to the Ascend wrapper and its operator tests execute this kernel; do not infer a changed standard-service result solely from this PR. See [the related score-binding work](https://github.com/vllm-project/vllm-ascend/pull/20434).

## Parameters

All eight kernel arguments are required. Grid size and `multibuffer` below are fixed by the current Python launcher rather than passed as kernel arguments.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` (`logits`) | Input | Logits with shape `[batch_size, vocab_size]`; vocabulary entries in each row are contiguous | fp32 / fp16 / bf16 | ND |
| `token_ids_ptr` (`sampled_token_ids`) | Input | One selected vocabulary ID per row, shape `[batch_size]` | int64 in the wrapper | ND |
| `logits_stride` | Input (attribute) | Distance between row starts, in elements: `logits.stride(0)` | integer scalar | scalar |
| `vocab_size` | Input (attribute) | Number of real vocabulary entries in each row | integer scalar | scalar |
| `batch_size` | Input (attribute) | Number of rows; marked `do_not_specialize` | integer scalar | scalar |
| `rows_per_core` | Input (attribute) | Rows assigned to each program; marked `do_not_specialize` | integer scalar | scalar |
| `BLOCK_SIZE` | Attribute (`tl.constexpr`) | Vocabulary tile width, fixed at 8192 by the wrapper | integer constant | scalar |
| `output_ptr` (`selected_token_ranks`) | Output | One rank per row, shape `[batch_size]`; internal counts use int32 | int64 | ND |
| Grid | Launch attribute | `(min(batch_size, vector_cores),)` | integer tuple | N/A |
| `multibuffer` | Launch attribute | Disabled by the wrapper to bound temporary-buffer use | bool (`False`) | scalar |

## Constraints

- `logits` is a two-dimensional NPU tensor with `batch_size > 0`, `vocab_size > 0`, and `logits.stride(-1) == 1`. A larger row stride is supported, provided every row exposes all `vocab_size` elements.
- `sampled_token_ids` and `selected_token_ranks` are contiguous one-dimensional NPU tensors of length `batch_size`, on the same device as `logits`. Every selected ID must satisfy `0 <= token_id < vocab_size`; the kernel does not mask the selected-token load.
- The current wrapper requires a nonempty batch to compute its grid and row assignment. Empty-batch handling is not added by this rank-only PR.
- Counts accumulate in int32; the scanned padded width must fit its positive range. The result is stored as int64, but this does not remove the internal counting limit.
- No vocabulary alignment is required. Partial tiles, multiple tiles, and an exactly aligned vocabulary all use the same formula. `BLOCK_SIZE` must remain a valid Triton tile width; the current launcher fixes it at 8192.
- For finite selected logits, the rank is in `[1, vocab_size]`. Ties count all equal entries; there is no token-ID tie-break. Selected negative infinity can include out-of-range lanes, as described above. NaN probability distributions are not validated by the committed tests.
- The kernel performs no explicit device-to-host transfer. Its wrapper reads shape metadata and the device core count to construct the launch. Eager correctness tests and a captured kernel microbenchmark do not establish model-level graph or scheduling invariance.
- There are no optional weights, bias tensors, or operator-specific normalization parameters: N/A.

## Origin and Differences

- **Origin**: Adapted from vLLM `vllm/v1/worker/gpu/sample/logprob.py::_ranks_kernel`. The upstream rank comparison is `>=`; the V1 helper `batched_count_greater_than` also uses `>=` despite its name.
- **Differences**:
    - NPU adaptation for performance: distribute rows over the available Vector Cores and loop over `rows_per_core`, rather than launch one program per row. Keep the 8192-lane counting vector and launch with `multibuffer=False`.
    - Align the former strict `>` comparison with upstream `>=`. Retain upstream behavior without an additional padding predicate or subtraction.
    - The output shape and dtype are unchanged. The change does not modify token sampling, logprob normalization, or the existing service bindings.

## Test Cases

The committed regression file is `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py`:

- Four wrapper cases use `(batch_size, vocab_size, num_logprobs)` of `(48, 1024, 5)`, `(96, 1024, 0)`, `(24, 1519, 1)`, and `(1, 320, 10)`. They compare token IDs, logprobs, and ranks against PyTorch. The batch of 96 also exercises the multi-row assignment on the validated A3's 48 Vector Cores.
- The one-based/tie case uses three rows and a vocabulary of 128, asserting ranks 1, 2, and 2 directly.

These small vocabularies are targeted contract regressions, not a claim of full-model inference coverage. Separate manual A3 rank-kernel checks also used the production-sized Qwen vocabulary of 151936. The captured rank microbenchmark used batch sizes 1, 32, and 128 with that vocabulary; these shapes are not additional parameter cases in the committed test file.

Ranks are integer counts: require exact equality (`rtol=0, atol=0`) against `(logits >= selected_logits).sum(-1)` on the same input values, independent of the input floating-point dtype. The committed wrapper tests use fp32 logits; their separate logprob comparison uses `rtol=1e-4, atol=1e-4`. This does not claim cross-dtype rank equality after logits have been rounded differently.

The file-level hardware guard skips unsupported products, including 310P. The current file contains five cases. Earlier A3 results included a padding-correction implementation that has since been removed; they are not a fresh validation of this revision.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py
```
