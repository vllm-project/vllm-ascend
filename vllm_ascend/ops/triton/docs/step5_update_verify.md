# Step5 update_verify (CSA region summary, MTP verify)

## Description

- **This will be removed once triton-ascend is compatible with upstream implementations**
- **Function**: Folds the `K+1` verified tokens of an MTP (speculative
  decoding) step per request into the CSA state, **in position order within
  each request**: each row is stored into the request's ring row, and any
  region whose eighth row lands inside the verified span is re-closed
  (replayed serially from the ring, including the just-stored rows) so the
  summary reflects the accepted sequence. A rejected draft that closed a
  region leaves a summary nobody reads; it is rewritten when the region
  closes again. Called from `AscendStep5DSAMetadata._update_summary_state`
  on uniform verify batches (`query_len > 1`).
- **Location**: `vllm_ascend/ops/triton/step5/summary_update.py` — kernel
  `_update_verify` (helpers `_replay_ring`, `_accumulate`, `_store_summary`),
  wrapper `update_verify`.
- **Formula**:
    - Per row `r` of request `q` (rows `[start, end)` in batch order):
      `position = seq_lens[q] - (end - r)`,
      `Ring[(base + position % capacity) * 512 + dim] = key`,
      `Ring[... + 256 + dim] = z`.
    - On `position % 8 == 7`: sequential online-softmax replay of ring rows
      `[position - 7, position + 1)` (the 7 earlier rows plus this row, in
      position order):
      `m' = max(m, z)`, `scale = den > 0 ? exp(m - m') : 0`,
      `num = num * scale + exp(z - m') * key`,
      `den = den * scale + exp(z - m')`, `m = m'`;
      finally `summary = num / max(den, 1e-20)` stored
      `fp32 -> bf16 -> fp8 e4m3` at the row's slot (same page addressing as
      `update_decode`).
- **Algorithm flow** (one program per request; its rows processed in order):
    1. Grid `(num_requests,)`; load the request's row span from
       `query_start_loc` and its ring `base` from the span's first row.
       Empty spans do nothing.
    2. For each row with `slot >= 0` and `base >= 0`: store the ring row,
       then close+summarize when the row is a region's eighth.
    3. `tl.debug_barrier()` between rows serializes the in-program row loop;
       inside `_replay_ring` a barrier per iteration keeps the Ascend
       vectorizer from reordering cross-iteration ring reads (the trip count
       is a runtime difference).
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend; execution
  evidence on Ascend950DT). Graph capturable: one program per request, all controls
  are device-side tensors, and the wrapper branches on
  `seq_lens.shape[0]` (static per capture bucket) — "uniform MTP verify
  update; graph capturable, no host metadata".

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key` | Input | Index K rows of the whole verify batch, `[total_rows, groups, 256]` | bf16 | ND; row and dim strides (`stride(0)`, `stride(2)`) passed to the kernel |
| `z` | Input | Index Z rows, same layout | bf16 | ND |
| `slots` | Input | Summary-cache slot per row (token units), `-1` skips | int64 | ND (1-D) |
| `query_start_loc` | Input | `[num_requests + 1]` cumulative row spans | int32/int64 | ND (1-D) |
| `seq_lens` | Input | `[num_requests]` full sequence length per request | int32/int64 | ND (1-D) |
| `ring_base` | Input | First ring row of the request block, per row, `-1` skips | int64 | ND (1-D) |
| `mean` | Output | Summary cache `[blocks, block_size // 8, 256]` | uint8 (fp8 e4m3 payload) | ND |
| `ring` | Output | Tail-ring rows, flat `[total_rows * 512]` | bf16 | 1-D flat |
| `capacity` | Attribute | Ring rows per request block (constexpr) | int | - |
| `MEAN_BLOCK`, `BLOCK_SIZE` | Attribute | From `summary_layout(mean)` (constexpr) | int | - |

## Constraints

- `seq_lens[q]` must be at least the request's verified span
  (`end - start <= seq_lens[q]`); positions are derived, not read.
- All rows of one request must share one ring `base` (the kernel loads it
  from the span's first row only); requests own disjoint ring blocks, so
  programs never contend.
- Rows with `slot < 0` / `base < 0` are skipped entirely (their ring rows
  are neither stored nor replayed).
- A region closed inside the verify span replays rows that may have been
  written by an earlier step AND earlier rows of this same span — hence the
  strict in-order row loop; the replay always reads this request's own ring
  block only.
- Kernel launches are skipped when `num_requests == 0`.
- Specialization: stride and geometry constexprs are fixed per deployment
  (`[t, 1, 256]` contiguous inputs, fixed ring capacity and cache layout),
  giving one serving variant; `enable_fp_fusion=False` for the softmax fold.
- In-place behavior: `ring` and `mean` are mutated in place; inputs are
  read-only.

## Origin and Differences

- **Origin**: Adapted from the upstream Triton implementation
  `vllm.models.step5.nvidia.ops.summary_update._update_verify` /
  `update_verify`; the kernel body and the sequential replay semantics are
  carried over verbatim.
- **Differences**:
    - `_replay_ring` adds `tl.debug_barrier()` per iteration (comment in the
      source): the Ascend vectorizer cannot prove cross-iteration safety for
      a runtime trip count, so the loop is explicitly serialized. The upstream
      version relies on the loop-carried dependency alone.
    - Address arithmetic keeps the int64 width of the slot and ring-base
      tensors (vLLM slot_mapping and the ring metadata are int64 end to
      end); the explicit casts are no-op guards against a future int32
      buffer, whose byte offsets could exceed 2 GiB.
    - `libdevice` from `tl.extra.cann` instead of `tl.extra.cuda`
      (functionally the same `expf`).

## Test Cases

Single-operator tests replay verify batches (draft length 1-3, several
requests, mixed `slot = -1` rows) against a pure-torch sequential reference
of the online-softmax replay. Ring stores bit-exact; FP8 summaries checked
with the FP8 tolerance (>= 99% bytes bit-equal, differences within one e4m3
quantization step). Cases include a region closing mid-span (replay reads
rows stored by the same span), ring wrap-around, and an empty batch.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_summary_update.py -k verify
```
