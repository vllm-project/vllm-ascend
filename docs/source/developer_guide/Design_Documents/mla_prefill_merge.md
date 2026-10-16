# MLA chunked prefill attention merging

MLA prefill combines the current query's attention result with results from
historical context chunks. The current prefix counts as one output/LSE pair.
Each time the list reaches 16 pairs, `_compute_prefill_context` calls
`torch_npu.npu_attention_update` with `update_type=1` and retains the merged
output and LSE as one pair. Subsequent chunks can trigger further updates.
A final list with multiple pairs uses `update_type=0`. A single pair after an
early update is already the complete result, including at 16, 31, and 46 total
pairs.

Empty chunks do not contribute pairs. If all historical chunks are empty, or
there is no historical context, the original prefix output and LSE are returned.
With valid historical context, the returned output has shape `[T, H, D]` and
FP32 dtype; the returned LSE is `None`, matching the existing caller contract.
The caller's existing output dtype conversion still applies.

## Relative LSE state

For each token/head row, merged output weights are proportional to `exp(LSE)`.
Adding the same constant to every LSE leaves the output unchanged. However,
adding a large common offset back into an intermediate FP32 LSE rounds away
small differences used by subsequent updates.

`merge_prefill_chunks` therefore keeps a finite per-row anchor separately from
the intermediate LSE. At the first early update it subtracts the largest finite
input LSE from every LSE. At later updates, the first pair contains an LSE
relative to the previous anchor, while newly appended pairs contain absolute
LSEs. The helper selects a new finite anchor and aligns the first pair using
`relative_lse + (previous_anchor - new_anchor)` before calling the native update.
It never reconstructs the absolute intermediate LSE.

The native update ignores infinite LSEs when a row also has finite LSEs, while
NaN LSEs and NaN/Inf outputs propagate through its arithmetic. A row containing
only infinite LSEs yields NaN output and LSE. To keep that undefined partial LSE
from poisoning a later valid chunk, intermediate updates temporarily use equal
scores for all-infinite rows and retain an empty-state LSE of `-inf`. Original
NaN/Inf outputs still propagate. At a final singleton flush, the loop restores
the native NaN output for rows whose aggregate LSE is `-inf`.

An empty partial result does not anchor a later valid chunk. Anchor selection
and alignment use device tensor operations and introduce no tensor-to-host
synchronization. Auxiliary LSE tensors scale with at most 16 input row vectors
per update.

## Validation

`tests/ut/attention/test_mla_prefill_merge.py` exercises the actual prefill loop
with a CPU model of the native update and an independent FP64 reference. It
covers 1, 2, 15, 16, 17, 30, 31, 32, 33, 46, 64, and 65 pairs, both supported LSE
layouts, skipped chunks, head padding, non-finite rows, extreme distributions,
and common LSE offsets of 1000. Intermediate outputs and LSEs use FP32 even when
the original attention output is FP16 or BF16.

| Original output dtype | Output atol | Output rtol |
| --- | --- | --- |
| FP32 | 2e-5 | 1e-4 |
| FP16 | 2e-3 | 2e-3 |
| BF16 | 2e-2 | 2e-2 |

Native NPU execution, capture/replay of the actual helper and loop, and
representative model workloads require hardware validation. CPU mocks alone do
not establish those results.
