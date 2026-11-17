# QSA preparation validation and JIT specialization

This is a white-box audit of the four preparation paths in PR #22086, starting
from revision `27bb374685245212222358b82bdfebb350eeba40`. It is not a record of
new NPU benchmark runs. Changes made after that revision require fresh checks.

## Interfaces and integration

| Path | Public interface and documentation | Production use |
| --- | --- | --- |
| Indexer Q | [QSA RMSNorm/MRoPE](qsa_rmsnorm_mrope.md) | `_fused_norm_rope` passes positions and the full table |
| Compressed K | [Compression](qsa_compress_groups_with_ratio.md) | `_update_and_compress`, then raw-ring updates |
| Indexer inputs | [Lightning preparation](lightning_prepare.md) | Before `npu_lightning_indexer` |
| Main Q/K | [Split QKV](split_qkv_rmsnorm_mrope.md) | `_project_qkv_gate`, excluding MTP |

Public wrappers own allocation and launch geometry. New paths do not read
device values with `.item()` or copy request metadata to the CPU. Shape/dtype
checks are host metadata checks, not checks of device tensor contents.

The compression model dispatch now separately checks ratio 4 and contiguous,
matching-width norm weights before selecting the combined kernel. The ordinary
pooling interface accepts other positive ratios, whereas the combined wrapper
rejects them. This distinction must remain separate from Q-only Norm eligibility.
Cache shape/device errors are still wrapper errors, not silent fallback.

## Coverage matrix and gaps

| Path | Existing new tests | Remaining evidence |
| --- | --- | --- |
| Indexer Q | 4 heads, width 128, empty/tail token counts, strided positions, clamp, replay | Map local head counts to actual TP configurations; run device tests |
| Compression | width 128, ratio 4, chunks 1/4/7, strided rings, 12-step updates, masked slots | Large prefill workloads, actual allocator layouts, hardware memory-safety validation |
| Lightning | 1..64 representative heads, row counts 0/1/32, padding boundaries, invalid owners, replay | Production request/table size distributions, longer batches, all applicable hardware |
| Main Q/K | Q heads 2/12, KV heads 1, width 256, BF16/FP32 norm weights, legacy comparison | Actual target TP geometry (including KV-head counts), shared-caller independent-reference runs |

The new device tests skip non-A3 devices. This is a test limitation, not a
demonstration of compatibility or a runtime device restriction. Determine the
supported model/device matrix before expanding test markers. Do not infer
support for every synthetic shape from test parameterization.

Exact comparisons test decomposition equivalence. Keep existing independent
reference tests for shared operators, and perform model-level evaluation as
well. CPU ABI tests cannot establish device compilation or numerical accuracy.

## JIT white-box analysis

### Explicit compile-time parameters

| Kernel | Parameters / expected domain | Reason and risk |
| --- | --- | --- |
| QSA RMSNorm/MRoPE | head width, block width, rotary width/sections, axis mode, `GATHER_CACHE` | Define reduction/vector shape and compile-time branches; QSA uses width 128, rotary 64, sections 11/11/10 |
| QSA RMSNorm/MRoPE | position axis/token strides, cache rows, epsilon, number of heads | Currently constexpr; buffer layouts or cache growth create variants. Address/clamp operands do not inherently require compile-time evaluation; evaluate runtime alternatives on NPU |
| Compression | ring size, compression ratio, head/block width, `LOAD_ROPE_POSITIONS`, `FUSE_NORM_STORE`, rotary layout | Select pooling loop, output path and vector shape. Fused ABI fixes ratio 4 and width 128 |
| Compression | rotary rows/stride, compressed strides/page size/capacity, epsilon | Allocation geometry is often stable per engine, but separate cache sizes/layouts specialize. Consider runtime address operands if measured variant counts justify it |
| Lightning | head/padded-head count, vector/table tile sizes, `PAD_QUERY`, width 128, ratio 4 | Determine vector allocation, masks and whether Q is copied; supported padded counts are 8/16/24/32/64 |
| Lightning | request count, exact table column count, score weight | Request/table shapes may vary by scheduler allocation. Request count is used for bounds and negative indexing; it is a candidate runtime scalar. Table tile size must remain compile-time, exact bounds need separate assessment |
| Split QKV | existing head/token partition and rotary configuration, new `GATHER_CACHE`, position strides, cache rows/stride | Existing core tiling already specializes. Direct lookup adds layout/cache variants; inspect complete legacy and new signatures, not only new parameters |

No claim is made that all existing constexpr annotations are optimal. This
change documents them without changing specialization policy before NPU
correctness/performance comparison. Newly added lowercase compile-time names
follow existing signatures but remain a naming cleanup item under the current
contribution guide; changing names alone does not reduce variants.

### Other specialization triggers

- Metadata int32 versus int64 and BF16 versus FP32 norm weights change pointer
  types. Optional positions/ring inputs and specialized branch modes also change
  signatures.
- Runtime stride/size integers may specialize by backend-inferred alignment or
  divisibility even when not explicitly constexpr. Tensor alignment can affect
  variants; full tensor addresses are not themselves the specialization key.
- Merely changing query, position, owner or table **contents** with unchanged
  metadata does not create a new specialization. Therefore replay correctness
  tests are not a recompilation stress test.
- Launch-grid changes alone are not proof of a new variant, but row counts can
  also affect explicitly passed token/tiling parameters, especially split QKV.

### Black-box validation plan (not yet executed)

1. Pin vLLM, vllm-ascend, torch_npu, CANN and triton-ascend versions, hardware,
   model/TP, graph mode and workload. Record whether compilation cache is cold.
2. Capture actual argument dtypes, shapes/strides, compile-time values and backend
   specialization/cache keys. Use instrumentation appropriate to that Triton
   version; do not treat every launch or cache file as a separate compilation.
3. Count distinct variants and actual compilations separately for startup,
   graph capture and steady-state serving. Include changing batches, request
   counts, context lengths, table layouts and graph buckets from a realistic
   evaluation workload.
4. Attribute unexpected variants to specific arguments. Evaluate runtime scalar
   conversion or specialization controls one at a time, with precision and
   latency checks, before adopting them.
5. Only then decide which required variants need warmup. If adding warmup,
   register in the active component window and use the same public wrapper and
   launch rules at runtime. No new warmup registration is included here.

## Evidence status

- CI on revision `27bb3746`: pre-commit, DCO and both CPU matrices passed in
  [run 44037858870](https://github.com/vllm-project/vllm-ascend/actions/runs/44037858870).
  NPU jobs were skipped and the approval gate failed. These results do not
  automatically validate later changes.
- The PR contains source-attributed A3 preparation-chain benchmarks. They are
  not measurements of this branch or end-to-end model speedups.
- User-reported integration evidence: three GPQA runs on A3 under the same
  configuration, including this PR, scored **90.4%, 91.9%, 92.42%**. These were
  not executed or independently verified by this audit. The exact combined
  revision, dataset subset, sampling settings and reproducible logs still need
  to be attached. No no-PR baseline was measured, so no non-regression claim is
  supported yet.
- Pending: final-head operator correctness on NPU, shared-operator regressions,
  applicable cross-device validation, matched model accuracy baseline, model
  latency/throughput, and startup/serving compilation counts.

## Reproduction entry points

On the configured NPU test image, run the three new tests plus shared baselines:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_direct_rope.py tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_compress_norm_store.py tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_lightning_prepare.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_rmsnorm_mrope.py tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
```

Use the repository CPU CI environment for ABI tests and `bash format.sh ci` for
all-file lint. Record commands, revisions, failures/skips and logs in the PR.
Do not replace hardware evidence with mocks or source-string assertions.
