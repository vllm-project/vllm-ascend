# Quantized PLE host lookup

## Description

- **Function**: Gather selected Qwen3.8-Flash-Next PLE embedding rows directly
  from CANN-registered host memory into a caller-provided NPU BF16 output.
  The separate `ple_lookup.py` operator handles existing BF16 storage;
  `ple_lookup_quantized.py::lookup_quantized` fuses lookup and
  group-32 INT8 or MXFP8 dequantization without expanding the full host table.
- **Formula**: For an owned global ID `i`, local row `r = i - shard_start` and column
  `c`, quantized output is `BF16(float(code[r, c]) * scale[r, c // 32])`.
  INT8 uses signed codes and FP32 scales. MXFP8 decodes E4M3FN codes and
  unsigned E8M0 scales: byte `s` represents `2**(s - 127)`, including
  `s = 0`; `s = 255` represents NaN. Non-owned IDs produce zero.
- **Algorithm flow**: One program handles one requested row, checks shard
  ownership, selects the host allocation chunk through the device code pointer table,
  loads the requested columns from host and one scale per group from the rank-local NPU parameter, broadcasts scales
  across each group of 32 columns, and writes BF16 results. Columns
  outside the width are masked. The caller owns cross-rank reduction.
  Both INT8 and MXFP8 specializations branch on shard ownership before host
  gathers: non-owned rows only write zeros; owned rows use column-only masks.
  MXFP8 scale decoding preserves the E8M0 zero and NaN encodings.
- **Supported modes**: INT8 on Atlas A3 and 950DT; MXFP8 on 950DT. Existing BF16
  dispatch is unchanged. Atlas A2 and 950PR validation is not claimed here.
- **Production caller**: Model-specific
  `AscendPLEHostEmbedding._lookup` in `models/qwen4_exp/ple_embedding.py`, via
  upstream synchronous PLE lookup. NPU-resident quantized PLE uses a separate
  PyTorch selected-row implementation and does not call this kernel.

## Parameters

All parameters are required; `is_mxfp8` is keyword-only for `lookup_quantized`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `code_chunk_ptrs` | Input | Device addresses of registered host code chunks | NPU INT64 | Contiguous 1-D |
| `scales` | Input | Local scale shard in HBM | NPU UINT8 (MXFP8) or FP32 (INT8) | Contiguous 2-D |
| `embedding_ids` | Input | Global embedding IDs; non-owned IDs produce zero | NPU INT32/INT64 | Any shape; copied to contiguous if needed |
| `output` | Output | Requested rows, written in place | NPU BF16 | `(*embedding_ids.shape, width)`, contiguous for quantized lookup |
| `shard_start`, `shard_end` | Attribute | Inclusive/exclusive global bounds of the owned shard | Python int | `0 <= shard_start < shard_end` |
| `chunk_rows` | Attribute | Rows per host allocation chunk | Python int | Positive; production uses `CHUNK_ROWS = 1 << 22` |
| `is_mxfp8` | Attribute | Select MXFP8 instead of INT8 storage | Python bool | Quantized entry only |

The wrapper retains the existing lazy `init_device_properties_triton()` call
to populate cached core counts and UB size before launch. This lookup uses a
fixed tile size and does not use those properties to select its grid or tile.

## Constraints

- Quantized width must be positive and divisible by 32. The target model uses
  width 160 (five scale groups); width 256 is an additional supported test case.
- All device tensors must be on the same NPU. The code pointer table must contain
  at least `ceil((shard_end - shard_start) / chunk_rows)` addresses. Code chunks contain rows of
  `width` elements; the scale tensor contains rows of `width // 32` elements.
- Host code storage is INT8 with FP32 scales, or E4M3FN with UINT8 E8M0 scales.
  Host allocations must remain registered and alive until all eager and graph
  work completes. Pointer-table contents are an owner contract, not validated
  by reading back addresses on the hot path. Metadata errors raise `ValueError`.
- The output must not alias the code pointer table, scales, IDs or host storage. Only the
  output is modified. Empty IDs perform no launch after metadata validation.
- IDs outside the shard, including negative IDs, are zeroed. No error is raised
  for a non-owned ID because other ranks may own it. Quantized host integration
  requires BF16 activations and ETP=TP. Lookup is synchronous; upstream async
  pinned-host prefetch is unsupported. NPU graph replay is supported; graphs
  must be torn down before closing host buffers.

## Origin and Differences

- **Origin**: Based on the existing Ascend BF16 `_lookup` implementation in
  `ops/triton/qwen4_exp/ple_lookup.py`. The quantized kernel and its public
  wrapper live independently in `ops/triton/qwen4_exp/ple_lookup_quantized.py`;
  the model caller selects the operator before invocation. `is_mxfp8` selects
  the INT8 or MXFP8 quantized format at compile time, not BF16 versus quantized
  lookup at runtime.
- **Differences**: Keeps compressed codes in registered host memory and scales in rank-local HBM,
  decodes selected rows, preserves out-of-shard zeroing, and retains upstream
  reduction and host-buffer ownership. It is a model-support feature; there is
  no unconditional throughput improvement claim. No autotuning is used.

## Test Cases

The CPU reference gathers the same code/scale rows, multiplies in FP32 and
casts to BF16. Tests require exact BF16 agreement (`rtol=0`, `atol=0`), with
matching NaNs for E8M0 special encodings. Cases cover INT8 extrema, FP32 zero
scale, E8M0 zero/NaN, actual host chunk boundaries, empty/non-owned IDs,
INT32/INT64 and sliced IDs, changed IDs during graph replay, metadata rejection,
and decode/batched-decode/MTP-shaped/prefill row counts at width 160.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_ple_quant_lookup.py
```

Model-level dataset accuracy and latency/throughput must be reported separately
in the PR, with software revisions, checkpoint, hardware, parallelism,
offload settings, graph/MTP settings and evaluation parameters. Historical
results or sampled-row correctness do not replace final-revision model tests.

## Performance comparison

### Measured operator results

The following historical results used host-resident scales. They were measured using CANN 9.2.1 r89, torch 2.10.0, torch_npu
2.10.0.post7 and triton-ascend 3.6.0.post2. Full registered host tables remain
resident in the same process. Codes and scales are synthetic; the BF16 table
contains exactly their dequantized values. IDs and outputs are checked for
exact BF16 agreement before timing. Measurements use fixed-ID NPU graph replay
and report median device-event microseconds per lookup, excluding model work,
TP reduction and network latency. A3 uses 50 calls per graph and 9 samples;
A5 uses 100 calls and 15 samples, alternating candidate, previous quantized
and BF16 measurement order. Five graph replays prime each measurement.

| Hardware / format | ID distribution | IDs | Quantized (us) | BF16 (us) | Latency reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| A3 INT8 | Fully owned | 1 | 6.188 | 5.828 | -6.19% |
| A3 INT8 | TP8-uniform | 128 | 6.702 | 6.267 | -6.94% |
| A3 INT8 | TP8-uniform | 256 | 8.032 | 7.913 | -1.50% |
| A3 INT8 | TP8-uniform | 4096 | 25.002 | 35.363 | 29.30% |
| A3 INT8 | TP8-uniform | 65536 | 320.183 | 459.063 | 30.25% |
| A5 MXFP8 | TP4-uniform | 128 | 4.270 | 3.538 | -20.68% |
| A5 MXFP8 | TP4-uniform | 256 | 5.336 | 4.804 | -11.07% |
| A5 MXFP8 | TP4-uniform | 1024 | 12.484 | 11.925 | -4.68% |
| A5 MXFP8 | TP4-uniform | 4096 | 41.700 | 41.541 | -0.38% |
| A5 MXFP8 | TP4-uniform | 65536 | 614.551 | 685.004 | 10.29% |
| A5 MXFP8 | TP4-uniform | 131072 | 1263.867 | 1401.373 | 9.81% |

Negative reduction means a slowdown. Small gathers can pay extra scale-read
and dequantization overhead; the feature does not guarantee lower latency for
every shape. Fixed-ID steady-state results do not establish cold-cache behavior
or full-model speedup. With service prefill chunks of 4096 tokens, the large
131072-ID row is an additional operator test rather than one service invocation.

At these table sizes, INT8 codes/scales use 7,200,034,560 bytes per rank versus
12,800,061,440 bytes for BF16 (43.75% less); MXFP8 uses 13,200,063,360 bytes
versus 25,600,122,880 bytes (48.4375% less). These are logical table bytes,
excluding allocation overhead, not total model memory savings.

Per 32 elements, BF16 uses 64 bytes, INT8+FP32 scale uses 36, and MXFP8+E8M0
uses 33, excluding allocation overhead. Reduced logical bytes do not guarantee
reduced physical traffic or latency. Report model performance against matching
settings, distinguishing the accuracy effect of quantization from the effect
of host placement. Do not compare different weight recipes as an offload-only
performance result.

## JIT specialization and startup

- `EMBEDDING_WIDTH` controls row addressing and masks; `CHUNK_ROWS` controls pointer-table
  addressing; `GROUP_SIZE=32` controls scale addressing; `BLOCK_SIZE=next_power_of_2(EMBEDDING_WIDTH)`
  defines vector lanes; `IS_MXFP8` selects different pointer types and decoding.
  These are compile-time values. Width 160 uses block 256. E8M0 decoding uses
  named constants for the exponent bias (127), NaN encoding (255), minimum
  scale (`2**-127`) and FP32 mantissa width (23 bits).
- `shard_start` and `shard_end` are runtime scalars with `do_not_specialize`, so shard-offset
  value/alignment classes do not create unnecessary variants. An integer type
  change (for example INT32 to INT64 inference) can still specialize.
- ID count changes only the launch grid. ID values do not specialize. INT32
  versus INT64 IDs, tensor alignment classes, constexpr changes, compiler flags,
  target, source/compiler changes and cache state can affect artifact reuse.
- This kernel uses normal eager Triton JIT; no custom warmup registry or global
  runtime cache is added. The NPU test primes one real specialization, forbids
  compiler entry points, then checks changing shapes, offsets and graph replay.
  This verifies reuse after priming, not compile-only registry warmup or absence
  of first-request compilation in a full service.
- If real service workloads expose required variants not reached during startup,
  first reduce unnecessary specialization, then use the existing Ascend
  `AscendTritonJitKernel`/`JitWarmupRegistry` mechanism for required variants, with
  registration-window, key-space and runtime-coverage tests.

Worker tracing of the model's registered-host path recorded one quantized-kernel
compiler entry per worker during startup and zero additional entries during
the observed serving phase: 8 INT8 workers on A3 and 4 MXFP8 workers on A5.
A3 covered 16 distinct ID shapes per worker before its partial run ended;
A5 covered 42 across 205 completed 131072-input/1024-output requests with
concurrency 1/8/32. Observed IDs were INT64, stride `(16, 1)`, with a 16-wide
ngram dimension; output width was 160, group size 32, block size 256 and
chunk size 4,194,304. Shard bounds varied across workers without introducing
additional serving variants. Compiler entries include cache resolution and
must not be described as guaranteed cold compilation. Counts describe the
measured serving snapshot; source/module renames can invalidate cached artifacts. This tracing used the
r89 serving adapter with the tested operator; it is not a full latest-main
framework installation or a completed model performance comparison.
