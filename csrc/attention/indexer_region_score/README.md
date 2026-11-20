# Paged indexer region score

This operator accepts native FP16, BF16, or FP8 E4M3FN query and key summaries,
with FP32 accumulation. Query and key summaries must have the same dtype; weights can be FP16, BF16,
or FP32. `D=256`; query groups `G` can be 1 or a positive multiple of 4,
and heads per group `H` can be any multiple of 4 from 4 through 128.
Key summaries are shared by all query groups.

```python
import torch

selected_region_ids = query.new_empty((query.shape[1], query.shape[0], 512), dtype=torch.int32)
torch.ops._C_ascend.npu_indexer_region_score.ids_out(
    query,                    # [T, G, H, 256]
    weights,                  # [T, G, H]
    key_summary,             # [P, S, 1, 256]
    actual_kv_lengths,        # int32 [N], includes the current query tokens
    block_table,              # int32 [N, C], logical page -> physical page
    cu_seq_lens,              # int32 [N + 1], packed query offsets
    max_region_len=8192,
    top_k=512,
    region_size=8,
    cache_mode=1,
    mask_mode=3,
    sort_mode=1,
    selected_region_ids=selected_region_ids
)
```

The only Torch entry point is
`torch.ops._C_ascend.npu_indexer_region_score.ids_out`. It requires the
caller to preallocate `selected_region_ids`, writes that buffer in place,
and returns `None`. The buffer must be INT32 `[G, T, top_k]` on the same
NPU as the inputs. The Torch adapter does not allocate a score or IDs
output and always disables the native dense score output.

The buffer contains logical region IDs in ascending order. IDs are
selected by descending score, with the smaller logical ID winning ties.
Unused slots contain `-1`. These IDs do not include physical page mapping,
the current region, or downstream packing.
`top_k` is in `[1, 512]`. Selection assumes finite FP32 scores.

Inputs and the IDs buffer may be noncontiguous or unaligned. ACLNN converts
inputs as needed and copies results back into the caller's output view.
Aligned paged cache views with contiguous page contents, gaps between
pages, and nonzero layer offsets retain zero-copy access; other key
layouts are materialized. The IDs buffer must not overlap internally or
with any input.

The mutating schema and allocation-free Meta implementation describe the
caller-owned IDs output for tracing. Standard `torch.compile` and
NPUGraph compatibility require end-to-end validation on the target NPU
and software stack.

IDs-only execution uses a fixed **256 MiB IRS algorithm workspace limit**,
including its CANN library prefix, intrinsic auxiliary storage and alignment
reserve. It computes
into a `[G,batch_T,roundup8(R)]` FP32 pool, selects IDs, and reuses the pool for
the next token batch. Request positions remain global across batches. If even
one token batch cannot fit, the call fails. Input storage and final IDs are
outside this algorithm limit. Outer input materialization and output-view
writeback storage are additional allocations. ACLNN `GetWorkspaceSize` returns
the full executor requirement, which may therefore exceed 256 MiB; callers
must allocate that returned size. This is not a whole-model memory limit.

Compatible aligned page-inner-contiguous Key views, including inter-page
gaps and layer offsets, retain direct reads. Other layouts are materialized.
Unaligned or strided outputs use aligned temporary storage followed by
writeback to the caller view. Alignment copies read only the logical dense
span, even when the backing storage is much larger or the offset exceeds
the view's element count. Scoring and selection still use the bounded pool;
IDs-only execution does not allocate a dense score output.

At the native layer, passing `output=nullptr` to
`aclnnIndexerRegionScoreGetWorkspaceSize` selects the same bounded
IDs-only path. The native ACLNN entry point still supports its original
optional dense score output when a score buffer is supplied; that
capability is not exposed by the Torch registration.
The internal GE score slot has shape `[0]` when disabled. No public workspace
budget parameter is added. The migrated scope remains D=256, K<=512 and
cache/mask/sort modes 1/3/1.

For query token `j` within request `n`, the visible historical region count is
`clamp((actual_kv_lengths[n] - query_length[n] + j) // region_size,
0, max_region_len)`. For each visible region, the score is
`sum_h(weights[t, g, h] * head_scale * relu(dot(query[t, g, h], key[n, r])))`
for each query group `g`.
Only causally visible regions compete for the selected IDs. For native
ACLNN callers requesting scores, only visible score columns are defined;
masked columns and extra output capacity must not be read as candidates.

The mode parameters remain public, with defaults `cache_mode=1`,
`mask_mode=3`, and `sort_mode=1`. Only this combination is currently
supported; other values are rejected. Continuous keys, D=128, other
masks, and score-ordered output are omitted. The
Torch entry point exposes the operator for models that already produce paged
key summaries; existing quantized lightning-indexer calls use a different
data contract.

Build the native operator with the repository's CANN build, then rebuild the
Torch extension to register `_C_ascend.npu_indexer_region_score.ids_out`:

```bash
bash csrc/build.sh --pkg --soc=ascend950 --ops=indexer_region_score -j8 -O3
```

vLLM initializes the native extension during startup. Standalone callers can
initialize it with `vllm_ascend.utils.enable_custom_op()` before the first
operator call, after sourcing the installed OPP environment.

The scoring stage retains the native Cube/Vector pipeline and paged key
reuse across query groups. H=4 retains its compiled specialization; other
head counts use a common reduction with the same FP32 arithmetic order as
the source operator. The selection stage uses the source FP32 radix threshold primitive
with a 16384-region window and Ascend C `Sort` for logical IDs and retained
score keys. Short histories generate IDs directly. This keeps the large
history path efficient while removing unused layouts, proxy dimensions and sorting
policies. The operator builds with the CANN APIs already used by this
repository.

The source performance records cover 26 model cases, all with G=1, H=4,
D=256, `top_k=512`, and `region_size=8`. Those records use score-ordered output
and are historical reference data. A performance comparison for this migration
must measure both implementations with logical-ID ordering, the same inputs,
and the complete score-and-selection kernel sequence.

The [single-operator tests](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_indexer_region_score.py)
call `.ids_out` with preallocated INT32 IDs buffers and compare the resulting
IDs exactly against a CPU mathematical golden reference. After building and
loading the operator environment, run from the repository root on Ascend A5 (950):

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_indexer_region_score.py
```
