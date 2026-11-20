# DeepSeek V4.1 QLI V2 and candidate integration

DeepSeek V4.1's indexer uses `QuantLightningIndexerV2` with paged INT8 index K.
The imported operator source is from `ops-transformer-qli_candidate.zip`,
SHA256 `771f0c16b9119c676c10cebef713b168f127ff194f3f2f98edcbcd0979c6f966`.
Its companion `QuantLightningIndexerV2Metadata` is built and registered too.

## Data flow

### Upstream planner, main128/SWA32

The current candidate uses unmodified vLLM at Ascend's verified main pin. All implementation changes stay in vllm-ascend. Main KV and
indexer manager blocks are 128 tokens; SWA pages are capped at 32 tokens,
matching upstream V4.1. Each manager block is one kernel page. Main block sizes
above 128 are rejected.

With one pipeline stage, V4.1 delegates grouping, placement, and capacity
accounting to the upstream planner through the Ascend platform builder.
Pipeline parallelism keeps Ascend's stage-local placement and uses the
configured block size for SWA.
The upstream scheduler owns block IDs, prefix caching, and recycling.
Ascend supplies storage specs, component views, strided C2 ring access, and
Mooncake registration of complete shared pool pages. Block tables and writer
slots retain the manager IDs without kernel-stripe remapping.

Runtime validation of this planner integration is pending. The regression
coverage exercises cache placement, component isolation, recycling, C1/C2 slot
coordinates, strided compressor rings, and remapped PD pages. Group counts,
usable capacity, operator correctness, and performance require validation on
the paired CANN/NPU environment.

`DeepseekV41Indexer.select` projects Q and head weights and applies RoPE.
`select_projected` quantizes each Q head to INT8 with a representable FP16
scale. Head weights and K scales are FP16. Existing source-owned K-cache
updates remain unchanged.

The operator consumes `TND` Q and `PA_BBND` index K directly. With main128,
C1 stores 128 rows per page and C2 stores 64. Index K and FP16 scales are
separate planes inside one resource page. The Torch adapter passes each page's
leading stride to ACLNN without a gather or a native operator ABI change.
Groups alias the same pool under scheduler ownership; group counts and padding
are determined by the current upstream planner rather than fixed slot tuples.

The C2 compressor owns one 32-row FP32 ring block per request. Pooling reads
and ring writes use the tensor's block stride. Pooling must finish before ring
updates overwrite residual rows; PCP gathered predecessors and tail writes
retain the same rule.

For a fixed-budget planning comparison on A2/A3, run:

```bash
python examples/deepseek_v41_cache_capacity.py MODEL --layout BLHNC --block-size 128 --swa-block-size 32
python examples/deepseek_v41_cache_capacity.py MODEL --layout BLHNC --block-size 128 --swa-block-size 128
```

The SWA128 option probes alternative geometry; the model launch uses SWA32.
Keep byte budget, context length, batch token limit, and concurrent batch count
identical. Compare pool blocks and per-request block demand. A higher raw block
count alone does not establish greater usable request capacity.

Metadata receives original query boundaries, compressed K lengths, and the
original sequence length modulo the compression ratio. The residual tensor
must be absent for ratio 1. On A2/A3, this repository compiles the model's
TND/PA_BBND layout only; BSND and nonpaged calls are rejected by host validation.
A5 retains its general QLI dtype, layout and quantization support.

The three candidate modes share one native operator:

| Mode | Meaning | Result |
| --- | --- | --- |
| 1 | Candidate source | Unfiltered position TopK and candidate block IDs |
| 2 | Candidate consumer | Rerank using this layer's Q and weights within source blocks |
| 3 | Candidate disabled | Ordinary position TopK |

Candidates are INT32 block IDs of shape `[tokens, 1, candidate_topk_blocks]`,
with `-1` padding. They are neither index-K vectors nor position TopK. A block
contains 8 compressed positions. Block scores use the maximum position score;
the last visible block is pinned. Shared attention state retains this tensor
within one forward and resets it on the next forward.

The model sorts returned position indices chronologically and moves `-1`
padding to the end before attention. Empty compressed contexts return empty
indices and, for a source, all-invalid candidates. A consumer without a source
raises an error.

## Current A3 contract

- INT8 Q/K, FP16 head weights and per-head Q/per-token K scales; quant mode 2.
- 32 or 64 replicated index heads, head dimension 128, one index-K head.
- DeepSeek V4.1 compression ratios 1 and 2, causal mask mode 3.
- Position TopK in `[1, 2048]`; candidate blocks a multiple of 64 in `[64, 2048]`.
- Candidate block size is exactly 8 in this kernel implementation.
- TND query and PA_BBND key layouts only in the compiled package.
- The A3 operator returns indices and candidate IDs, not score values.

The numerical reference follows the supplied INT8 golden: INT32 QK divided by
1024, FP16 ReLU and `weight * Q_scale`, FP32 head reduction, then K scaling.
Query quantization and FP16 weight rounding differ from the earlier floating-Q
small-operator path. Operator agreement with this quantized reference does not
establish full-model or dataset accuracy.

## Build and regression coverage

Build with `pip install -v -e . --no-deps --no-build-isolation` on the paired
CANN/NPU environment. Both new symbols are registered on PrivateUse1 and Meta.

Standalone operator-level E2E coverage for candidate generation/consumption,
mixed requests, paged views, Meta shapes, and slot-backed SparseFlashMla is
deferred. The end-to-end results below describe the original layout.

The imported host tiling needed one semantic fix: TND candidate size is
`T * N_k * blocks`, because T already includes every request. Multiplying by
batch size again incorrectly rejects mixed-request consumers. Kernel offsets
already use TND query prefixes and require no corresponding change.

Validation includes single-chip eager and Meta contracts, plus the full
40-layer W8A8 checkpoint on one A3 server with TP4/DP4/EP16 (Engram and
DSpark disabled). Four end-to-end cases were each run twice at temperature 0,
seed 7: arithmetic, capital-city lookup, multi-turn recall, and a 20,032-token
retrieval prompt. Both runs returned the expected answers and identical output
tokens. The long prompt exceeds the 16,384-position candidate block budget.
It exercises chunked prefill and decode with actual candidate filtering.

The checkpoint-provided `encoding.encode_messages(..., thinking_mode="chat")`
was used with `/v1/completions`; the checkpoint does not provide a standard
chat template. Repeated output agreement validates these fixed functional
cases, not a baseline-versus-candidate dataset accuracy comparison. Graph,
64k/128k requests, multi-node execution, Engram and DSpark remain unvalidated.

## Compilation scope

DeepSeek V4.1's A2/A3 path uses INT8 Q/K, FP16 weights/scales and quant mode 2. Its
compiled QLI V2 template matrix has one key, down from 4. A5 retains the full
16-key matrix, including paged BSND/TND queries and matching nonpaged BSND/TND
layouts. A5 dtype registration, host validation, and kernel dispatch retain
FP8, MXFP8, HiFloat8, MXFP4 and INT8 (quant modes 1/3/4/5/2 respectively).
The DeepSeek V4.1 INT8 call sites do not establish that other A5 paths are unused.
Both architectures retain their original template argument encodings.

Candidate modes 1/2/3, compression ratio, TopK and sequence lengths remain
runtime parameters. The A2/A3 candidate implementation and the A5 implementation
remain in separate architecture branches. This pruning does not add A5 candidate
support or change which operators the default A5 package builds: QLI V2 and
SparseFlashMla are currently included by the A2/A3 package lists. Explicit A5
builds of these operators use the A5 template selections.

SparseFlashMla is likewise restricted to the model's BF16 TND/PA_BBND SWA/CSA
calls. It compiles 6 keys on A2/A3 and 12 on A5; shared keys remain on both,
while single-head CSA specialization is A2/A3-only and split-G/vectorized
addressing is A5-only. HCA, independent original-KV sparse templates, other
layouts and FP16 are excluded on both architectures. Host validation rejects
pruned contracts before kernel lookup. The full compilation matrix is in
`csrc/attention/sparse_flash_mla/docs/ratio2_a2a3.md`.
HcPre already isolates A2/A3 key 0 from A5 keys 1000/1001; both A5 paths can
be selected by runtime token counts, so neither is removed.

The fused Compressor serves DeepSeek V4; V4.1 currently uses small operators
for compression. Compressor already selects four keys per architecture:
TH/BF16, interleaved RoPE, continuous cache, and `coff=1/2`, with FP32 RoPE.
A2/A3 selects EMPTY_X/PERF; A5 selects NORMAL/EMPTY_X. The two `coff` values
cover V4 compression ratios 128 and 4 respectively, and empty input remains
supported. A5 FULL_LOAD requires BSH, so it is already excluded. Host and kernel
sources are selected separately for arch32 (A2/A3) and arch35 (A5).

Compressor dtype registration now matches those existing selections: one
signature per architecture, down from four on A2/A3 and two on A5. Norm weights
remain BF16 on A2/A3 and FP32 on A5. Host validation likewise rejects uncompiled
layouts, dtypes, and modes. The empty-input entry uses a discarded `else` branch
so the compiler does not instantiate a computation kernel after an unconditional
return. The four keys themselves and their encodings remain unchanged.

The CPU-only template regression can be run without importing torch or CANN:

```bash
python3 -m unittest discover -s tests/ut/ops -p test_aurora_tiling_keys.py -v
```

This checks preprocessor selections, architecture isolation, model layout
calls, dtype registration and unchanged argument encodings. It also compiles
the real Compressor entry against stubs that reject unwanted template
instantiations, including any computation for EMPTY_X. It does not measure CANN build time
or execute NPU kernels. Rebuild the operator package before the numerical
regressions; retained key counts alone do not establish a wall-clock speedup.
