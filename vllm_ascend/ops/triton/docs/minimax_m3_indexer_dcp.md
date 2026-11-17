# MiniMax M3 index-key context sharding

## Description

This is a model-specific capacity feature for MiniMax-M3.1-Flash. The production
callers are `MiniMaxM3SparseAttention._insert_kv` and
`AscendMiniMaxM3IndexerImpl.forward`. Python wrappers live in
`vllm_ascend/models/minimax_m3/ops/indexer_dcp.py`; callers never launch the Triton
kernels directly. Fused preparation uses `minimax_qknorm_rope_cache` instead of
the standalone scatter. This is an experimental feature, not a temporary patch
with an announced replacement. It remains default-off pending acceptance evidence.

Main attention K/V remain head-sharded. Logical index page offset `t` belongs to
TP rank `t % size`; each rank stores only `128 / size` keys per logical page.
Decode gathers index Q heads, computes local BF16 QK GEMMs with FP32 output,
looks up the request's physical pages, applies sequence/causal masks, and takes
the maximum over each logical block's local keys. Scores are stored in FP16.
MAX ReduceScatter combines shard maxima and returns each TP rank's query heads.
The existing TopK wrapper widens scores before forced priorities are applied.
Prefill gathers only live pages, restores token-interleaved order, and invokes
the existing CANN scorer on the materialized cache.

## Parameters and side effects

| Wrapper | Inputs | Output and mutation |
| --- | --- | --- |
| `scatter_index_shard` | BF16 cache `[P,128/S,128]`, updates with `N*128` elements, contiguous int32/int64 logical slots `[N]`, `size=S`, rank | Writes owned nonnegative slots into cache in place; updates and slots are read-only |
| `index_shard_scores` | BF16 query `[B*Q,H,128]`, BF16 cache `[P,128/S,128]`, int32/int64 table `[B,L]`, lengths `[B]`, uniform `query_len=Q`, size/rank, causal flag | New contiguous FP16 scores `[H,B*Q,ceil(L/16)*16]`; no input mutation |
| `reduce_scatter_index_scores` | Contiguous-compatible scores and TP group; `H` divisible by TP size | New local-head scores; MAX collective over TP, no score mutation |
| `plan_prefill_index_gather` | CPU block table, final sequence lengths and logical block size | Unique live page IDs including safe page 0, and a remapped table; inputs unchanged |
| `gather_index_shards` | Local cache, matching TP group, selected physical page IDs | New full-page BF16 cache; collective order must match across TP ranks |

All tensor inputs to scatter/scoring must share a device. Scoring requires
unit-stride table columns and contiguous lengths; padded table rows are allowed.
Query/cache strides are supported through compact copies when necessary.
Scatter supports cache/update strides and skips negative slots. The caller must
keep referenced page IDs and slots in bounds and avoid duplicate owned writes;
device tensor contents are not synchronously scanned by the wrapper.
Writable cache views must not overlap internally or alias updates/slot inputs.
Invalid and padded score entries are `-inf`. Empty score tensors launch no kernel.

## Constraints and devices

The feature requires TP2/TP4, BF16 index keys, 128-token logical blocks,
128-dimensional keys, and installed CANN MsaIndexScore for prefill. Generic PCP
and DCP stay 1. PD requires matching TP and sharded cache layouts on both roles
with MooncakeConnectorV2. Spec `head_size=128/S` is byte accounting, not a change
to the physical key dimension; the backend constructs `[P,128/S,128]` storage.

The checklist tests were exercised on A3. A2/A5 compatibility of this new feature
is not established by the unsharded path or these tests. Record actual validation
and dependency constraints per applicable device before extending the support
claim. The unsharded feature-off path remains available. Do not silently select
the sharded path on an unvalidated device and infer correctness from imports.

## Origin, numerical reference and acceptance

The reference is the existing CANN MsaIndexScore on replicated keys. CANN's BF16
path publishes FP16-rounded scores in FP32. Compare valid scores exactly after
widening the compact FP16 result; normalize only CANN's invalid FP32-min sentinel
to `-inf`. Scatter, page restoration and integer IDs/counts require exact semantic
agreement. Forced-priority ties permit different equally ranked IDs, but require
equal selected scores, valid unique IDs, correct counts and ordering when requested.

The CANN wrapper pads a private block table to a multiple of 16 columns when
necessary. The tested CANN 9.2.1 batched path produces incorrect scores for an
unaligned logical width; increasing its row stride alone does not correct it.
Sequence lengths mask the added zero-page columns, and the already aligned score
width is unchanged. Aligned tables are passed through without a padding copy.
The caller's table and its layout are preserved. Compact scoring accepts the
original unaligned table directly and is also checked against CPU PyTorch.

Fused Norm/RoPE BF16 outputs must have at least 99% of elements within
`2**-6 + 2**-6 * abs(reference)` and maximum absolute error at most 1.
Untouched cache regions must be bit-exact. Operator equality does not replace
baseline/candidate model dataset accuracy, which must be reported separately.

## JIT specialization

| Kernel | Runtime, without value/alignment specialization | Intentional compile-time geometry |
| --- | --- | --- |
| `_scatter_index_shard` | Token count | TP size/rank, tensor layout strides, tile |
| `_gather_index_shard_scores` | Row start, GEMM row stride, table row stride/width, token count, query length, score width, block tile count | Heads, TP size/rank, causal mode, tile |
| Fused Norm/RoPE writer | Actual/padded token count | Head geometry, cache/input layout, TP size/rank, epsilon, dtype path |

Request lengths, batch sizes and cache pool sizes are indexing bounds, not tile
shapes, and must not create one variant per value. Tensor dtypes and pointer
alignment classes can still produce variants. Changing fixed geometry/layout or
compiler options can legitimately create a variant. Triton GEMM is not implemented
here: `torch.mm` has its own backend behavior and must be profiled separately.
The 128-row GEMM bound limits transient memory; 16 block columns per gather tile
reuse the fixed page lanes while bounding vector work. Retuning either requires
paired correctness and latency/memory evidence.

First validate reduced specialization for precision and performance. This patch
does not register warmup variants or claim model steady-state compilation is zero.
Use actual model shape/compile traces before deciding the necessary warmup space.

## Test and benchmark entry points

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_minimax_m3_indexer_dcp.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_minimax_m3_indexer_dcp_jit.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py
pytest -q tests/e2e/pull_request/four_card/test_minimax_m3_indexer_dcp.py
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_minimax_m3_indexer_dcp.py --iterations 200 --graph
```

Single-card tests exercise TP2/TP4 owners, padded/strided storage, 129/601/131077
contexts, Q=1/8/17, empty rows, poisoned unused pages/tails, FP16 rounding, fused
packed/separate projection writes and dynamic-length graph replay. Distributed
tests exercise actual Q AllGather and FP16 MAX ReduceScatter in eager/graph modes
against an independent CPU PyTorch oracle. The oracle gathers logical pages,
computes FP32 dot products and causal block maxima, then rounds to FP16. Apply
the FP16 acceptance gates (99% within mixed `2**-9` tolerance and maximum error
at most 0.1); infinity masks must agree exactly. Separate CANN cases require
exact equality and expose toolchain differences rather than relaxing tolerances.
The benchmark compares replicated/expanded/compact scoring on identical inputs
and gates timing on exact valid-score equality; it excludes TopK and model serving.
Report prefill, writer, TopK and full-model performance separately.

## Diagnostic model JIT traces

`tools/trace_minimax_m3_jit.py` provides an explicit diagnostic installer for
Triton 3.6. Load the file with `importlib.util.spec_from_file_location` from a
task-local `sitecustomize.py`, then call `install(output_dir, phase_file)`.
Put only that site's directory and the repository root on `PYTHONPATH`; adding
the repository's `tools` directory can shadow Python's standard `bisect` module.
Ensure every spawned worker inherits the site. This instrumentation changes JIT
methods only in the diagnostic process, writes one JSONL file per PID and requires
no production environment variable or warmup registration.

Start the phase file with `startup`, switch it to `serving` after engine startup,
and exercise the dataset's request lengths and concurrency. Use a fresh
`TRITON_CACHE_DIR` for a cold run and repeat in a fresh process with the same
cache to measure persistent-cache loading. Each record contains the phase,
kernel, input shapes/strides/dtypes, alignment class, scalar and constexpr values,
specialization controls and launch options. Complete cache resolutions and
frontend compilations are separate events: disk-cache hits resolve a kernel
without rebuilding its IR. Incomplete events indicate an interrupted or failed
operation, not a successful compilation. Count successful events per kernel,
phase and PID; do not merge worker-local caches into one variant count.

The synthetic JIT test checks in-process reuse for changing request counts,
query lengths, table widths and pool capacities. It does not certify GPQA or
model steady-state behavior. Model traces and clean, uninstrumented performance
runs remain separate delivery evidence. Necessary layout, dtype, TP ownership,
head geometry and compiler-option variants may still compile; determine their
reachable space before adding warmup wrappers and registration tests.
