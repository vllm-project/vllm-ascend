# apply_grammar_bitmask

## Overview

`apply_grammar_bitmask` applies a packed grammar mask to logits in place for structured-output and grammar decoding. A mask bit of `1` allows the corresponding token and preserves its logit. A bit of `0` blocks the token and writes `-inf`.

The mask is stored as packed INT32 words, with one word representing 32 consecutive vocabulary entries. `logits_indices` maps each mask row to a request and position in the flattened logits tensor.

## Kernel organization

The Ascend implementation uses mask-major scheduling. Each VectorCore program owns a contiguous range of mask rows and processes all vocabulary blocks for each row. The request/position mapping and active-row check are therefore loaded once per mask row instead of once per vocabulary block. Host-known logits stride, bitmask stride, and vocabulary size are compile-time constants for each specialization, while request-dependent offsets and task counts remain runtime values.

For each vocabulary block, the kernel selects one of three paths:

- Fully allowed block: skip the writeback and preserve the existing logits.
- Fully blocked block: write `-inf` directly without loading logits.
- Mixed block: load logits, select the blocked elements, and store the result with a contiguous vector write.

Inactive speculative positions skip all vocabulary accesses. An empty mask grid returns without launching a Triton kernel.

## Host-side bitmask path

The Ascend structured-output manager starts parallel bitmask filling above 64 non-speculative requests. Speculative decoding, diffusion, and smaller batches retain the upstream serial or request-specific paths.

The Model Runner V2 worker preallocates pinned host buffers for the packed bitmask and request mapping. Consecutive non-speculative decode steps with an unchanged request set and order reuse the device mapping; every step still copies the new bitmask. A copy-stream event protects host buffers from being overwritten while an earlier asynchronous transfer is pending. Speculative positions and changed request mappings are rebuilt on every step.

## Semantics

For mask row `r` and vocabulary token `v`:

```text
word = floor(v / 32)
bit = v % 32
allowed = (bitmask[r, word] & (1 << bit)) != 0
```

The output is:

```text
logits[mapped_row, v] = logits[mapped_row, v]  if allowed
logits[mapped_row, v] = -inf                    otherwise
```

The implementation preserves the input dtype and allowed logit values, and writes `-inf` at blocked positions. The final vocabulary block is protected by `vocab_size`; padding words beyond the logical vocabulary are excluded from the fast-path decisions.

## Interface and constraints

| Argument | Description |
| --- | --- |
| `logits` | Two-dimensional `[num_logits, vocab_size]` tensor. The vocabulary dimension must be contiguous. |
| `logits_indices` | INT32 mapping entries packed as `request_index * MASK_STRIDE + position`. |
| `cu_num_logits` | INT32 cumulative row offsets with `num_requests + 1` entries. |
| `bitmask` | INT32 tensor with at least `ceil(vocab_size / 32)` words per row. Extra row stride is supported. |
| `vocab_size` | Logical vocabulary size. |
| `MASK_STRIDE` | Packing stride larger than the largest position value. |
| `BLOCK_SIZE` | Vocabulary tile size. The validated A2/A3 shape uses `8192`. |

The current validation scope is Ascend A2/A3 with `multibuffer=False`. A5/950-specific tile tuning is outside this change.

## Validation

The NPU test covers:

- business vocabulary size `151936`;
- a smaller vocabulary with a non-aligned tail;
- fully allowed, fully blocked, alternating, and random masks;
- inactive speculative positions;
- non-identity request/position mappings, including non-speculative decode;
- a padded bitmask row stride;
- BF16 logits.

Run the direct operator test with:

```bash
pytest -sv --noconftest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_grammar_bitmask_triton.py
```

Run the standalone latency benchmark on an Ascend NPU with:

```bash
python benchmarks/benchmark_grammar_bitmask.py \
  --output-dir /tmp/grammar-bitmask-profiles
```

The benchmark exports three profiled device latencies per shape after warmup and
supports loading a baseline kernel with `--kernel-source`. Use the same runtime,
input shapes, and mask patterns when comparing implementations.
