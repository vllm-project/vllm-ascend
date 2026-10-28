# Grammar bitmask (`_apply_grammar_bitmask_kernel`)

## Description

- **Function**: applies structured-output grammar constraints to logit rows in
  place. A zero vocabulary bit masks the corresponding logit to `-inf`; a set
  bit preserves its value.
- **Entry point**: `_apply_grammar_bitmask_kernel` in
  `vllm_ascend/ops/triton/v2/apply_grammar_bitmask.py`. Its launcher implements
  the upstream two-dimensional indexing interface and dispatches
  `_apply_grammar_bitmask_kernel_impl` on a one-dimensional VectorCore grid.
- **Supported modes**: Ascend NPU MRv2 structured-output sampling. The regression
  was validated on Atlas A3; this change does not establish additional hardware
  or graph-mode coverage.

For mask row `m`, the mapping and update are:

```text
request  = logits_indices[m] // MASK_STRIDE
position = logits_indices[m] % MASK_STRIDE
start    = cu_num_logits[request]
active   = position < cu_num_logits[request + 1] - start
row      = start + position
allowed  = (bitmask[m, token // 32] >> (token % 32)) & 1

if active and token < vocab_size:
    logits[row, token] = logits[row, token] if allowed else -inf
```

**Algorithm flow**:

1. Flatten `(mask row, vocabulary block)` into a task index.
2. Split the tasks into contiguous ranges across at most the available
   VectorCores; each program receives either `floor(tasks / programs)` tasks
   or one additional task.
3. Resolve the request and speculative position using `cu_num_logits` on
   device. Positions beyond that request's actual logit count are inactive.
4. Expand each int32 bitmask word into 32 vocabulary decisions, select original
   logits or `-inf`, and write the valid vocabulary block back.

## Parameters

All tensor pointers refer to NPU tensors. Strides are measured in elements.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input/Output | Logits updated in place. | fp16, bf16, or fp32 | `[num_logits, vocab_size]`, contiguous vocabulary dimension |
| `logits_stride` | Attribute | Row stride of logits. | integer | scalar |
| `logits_indices_ptr` | Input | Packed request/position mapping, one entry per mask row. | int32 | `[num_masks]` |
| `cu_num_logits_ptr` | Input | Prefix sum of actual per-request logit counts. | int32 | `[num_reqs + 1]` |
| `bitmask_ptr` | Input | Little-bit-order allowed-token mask; each word covers 32 tokens. | int32 | `[num_masks, bitmask_stride]`, contiguous words |
| `bitmask_stride` | Attribute | Mask row stride in int32 words. | integer | scalar |
| `vocab_size` | Attribute | Number of vocabulary entries. | integer | scalar |
| `MASK_STRIDE` | Attribute | Number of reserved mask positions per request. | constexpr integer | scalar |
| `BLOCK_SIZE` | Attribute | Vocabulary tile width, divisible by 32. | constexpr integer | scalar |

The launch grid is `(num_masks, ceil(vocab_size / BLOCK_SIZE))`. The launcher
supplies these internal kernel attributes:

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `total_tasks` | Attribute | Product of the logical grid dimensions. | integer | scalar |
| `NUM_PROGRAMS` | Attribute | Minimum of available VectorCores and total tasks. | constexpr integer | scalar |
| `NUM_VOCAB_BLOCKS` | Attribute | Number of vocabulary tiles per mask row. | constexpr integer | scalar |

## Constraints

- Callers own mapping validity and must provide distinct active output rows so
  separate tasks do not write the same logit block concurrently.
- `MASK_STRIDE` and `BLOCK_SIZE` are positive; `BLOCK_SIZE` is a power of two
  and a multiple of 32. The regression uses `MASK_STRIDE=6`, `BLOCK_SIZE=8192`.
- `cu_num_logits` is a non-decreasing prefix sum within the allocated logit
  rows; mapped request indices must have both prefix-sum entries available.
- Each mask row contains at least `ceil(vocab_size / 32)` words. Vocabulary
  tails are masked, and inactive speculative positions do not write logits.
- The launcher expects a non-empty logical grid. Empty work is handled by
  the caller before launch.
- The operator is inference-only. It adds no host synchronization; callers
  establish dependencies when preparing masks on a separate copy stream.
- `multibuffer=False` is retained to limit UB usage on A2/A3. Graph-mode
  acceptance is separate from the eager stream-overlap regression below.

## Origin and Differences

- **Origin**: Ascend implementation of the upstream MRv2
  `vllm.v1.worker.gpu.structured_outputs` grammar-bitmask kernel interface.
- **Launch adaptation**: distributes the logical grid over a bounded number
  of VectorCores instead of launching one program per logical tile.
- **Store adaptation**: the previous store predicate included arbitrary
  per-token grammar bits. The new code loads the active block and uses
  `tl.where(blocked, -inf, logits)`; the store predicate only covers the
  active row and contiguous vocabulary tail.
- Affected Triton-Ascend versions can generate an unordered lock workspace
  for the former store pattern. Avoiding that lowering removes the implicated
  workspace from the tested specialization without patching the Triton
  launcher or serializing asynchronous launches. See
  [Triton-Ascend #2817](https://github.com/triton-lang/triton-ascend/pull/2817).
- This is paired with the stop-logit recovery rewrite described in
  [logit_bias.md](sample/logit_bias.md). It does not claim to eliminate locks
  from every possible Triton kernel or specialization.

## Test Cases

The grammar accuracy test covers 64 rows, vocabulary size 151936, and packed
request/position mappings with inactive speculative positions. Fully allowed,
fully blocked, alternating, and random mask words are compared bit-exactly
against a CPU reference, including preservation of inactive output rows.

The stream-overlap regression uses the incident's shape: 18 logit rows,
129280 vocabulary entries, and six reserved positions for each of three
requests. It runs 20 iterations of asynchronous mask/mapping copies, grammar
masking, and logit-bias processing, with stream dependencies and one final
device synchronization. It compares the complete grammar output and final
biased output against CPU expectations with `rtol=0`, `atol=0`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_grammar_bitmask_triton.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_logit_bias.py
```

Separately, Atlas A3 validation ran three processes with 1000 iterations each
and inspected four compiled grammar/bias specializations, which reported
`has_unordered_sync_block_lock=false`. This is operator-level evidence,
not a claim that every PD service configuration has passed long-running tests.
