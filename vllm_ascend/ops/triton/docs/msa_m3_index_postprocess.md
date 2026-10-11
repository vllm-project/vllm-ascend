# MSA index decode TopK postprocessing

## Description

- **Function**: `index_decode_topk_fused` selects sparse logical block IDs from
  AscendC MsaIndexScore output, masks invisible blocks, forces initial/local
  blocks, and returns int32 indices and valid counts.
- **Formula**: For query token `q`, request `b = q // decode_query_len` and
  offset `o = q % decode_query_len`, the visible block count is
  `V = min(logical_block_count, max(0, (context_lens[b] + o + 128) // 128))`.
  Invisible columns receive `-inf`. Visible initial blocks receive `1e30`;
  visible local blocks receive `1e29`, overriding initial-block scores on overlap.
  TopK selects the resulting scores. Selected IDs outside `[0, V)` become `-1`.
- **Algorithm flow**:
  1. Triton prepare kernel applies visibility, initial/local masks and padding.
  2. `torch.topk` selects block IDs along the last dimension.
  3. Triton postprocess kernel casts indices to int32, replaces invalid IDs by
     `-1`, and counts valid selections.
- **Supported modes**: A2/A3 MiniMax-M3 decode path; verified on Ascend910_9362
  (A3). A2 execution was not tested in this validation. 950: N/A for this path.
  The interface supports single-token decode and uniform multi-token decode.

## Parameters

All parameters are required except `init_blocks` and `local_blocks`, which default to zero.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `score` | Input/workspace | Scores `[H, B * Q, W]`; may be modified in place | float32 | ND, strided supported |
| `context_lens` | Input | Context lengths before the current queries, `[B]` | int32/int64 | ND, strided supported |
| `decode_query_len` | Attribute | Uniform query length `Q > 0` | Python int | Scalar |
| `topk` | Attribute | Number of output slots `K > 0` | Python int | Scalar |
| `logical_block_count` | Attribute | Logical block-table width, from 0 to `W` | Python int | Scalar |
| `init_blocks` | Attribute | Non-negative initial-block count | Python int | Scalar |
| `local_blocks` | Attribute | Non-negative local-block count | Python int | Scalar |
| `indices` | Output | Logical block IDs `[H, B * Q, K]`; `-1` for invalid slots | int32 | ND |
| `counts` | Output | Valid selections per query/head, `[H, B * Q]` | int32 | ND |

## Constraints

- Block size is fixed at 128 tokens. It is not an interface argument.
- Both input tensors must be on the same NPU. Input rank, dtype, batch/query
  relationship and scalar ranges are checked before launching kernels.
- Empty head or token dimensions return correctly shaped empty outputs without
  launching kernels. Zero score width is supported when logical block count is
  zero; nonempty rows then produce only `-1` indices and zero counts.
- `topk` may exceed the score width or visible block count: score storage is
  padded and extra output slots are invalid. Negative context values are useful
  for dummy rows; visibility is clamped at zero.
- Scores in visible blocks must be initialized. Padding/invisible scores are
  never loaded. Tests use finite scores below the forced-score sentinels;
  selection among equal scores has no stable index-order guarantee.
- A contiguous score buffer can be reused and modified in place. Clone it if
  the caller needs original scores after this call. Non-contiguous scores use
  a separate contiguous preparation buffer.
- Dynamic widths, query length and batch-dependent/input-view strides use
  `do_not_specialize`. Model constants and fixed output strides use
  `tl.constexpr`. Changing model constants or input dtype can still require a
  different compiled specialization.
- The caller supplies correctly updated metadata for ACL graph replay. This
  validation measures eager execution; it does not certify all graph shapes.

## Origin and Differences

- **Origin**: MiniMax-M3 A2/A3 decode helpers originally under
  `models/minimax_m3/ops`, moved to `ops/triton/msa` for shared operator ownership.
- **Differences**: Fuses score preparation and index/count postprocessing into
  two Triton kernels around `torch.topk`. AscendC score computation, the existing
  `msa_m3_triton.py` kernels and TP projection logic are unchanged.

## Test Cases

Accuracy and interface cases are in
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py`.

| Category | Cases |
| --- | --- |
| Empty shape | `[0,0,0]`, `[0,3,32]`, `[4,0,32]`, `[4,0,0]` |
| Zero logical width | Nonempty queries with score widths 0 and 32 |
| Boundary visibility | Context lengths `-Q, 0, 127, 128, 255, 256, 4095`; `Q=1,4` |
| TopK padding | `(K,W)=(1,1),(16,7),(16,33),(32,16)` |
| Input types/layout | int32/int64 context; contiguous and strided score/context |
| Forced blocks | Disabled and overlapping initial/local windows |
| Invalid inputs | Rank/dtype, device mismatch, query length/batch mismatch, invalid TopK/block counts |

Integer outputs and selected scores are checked exactly (`rtol=0, atol=0`).
For ties introduced by forced scores, compare selected scores, uniqueness and
counts instead of requiring a particular ordering of equally ranked block IDs.

```bash
ASCEND_RT_VISIBLE_DEVICES=8 python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py
```

### Performance plan

The following matrix was specified before running benchmarks. Configuration
matches the supplied MiniMax-M3 model: `H=4`, `K=16`, block size 128,
`Q=1`, `init_blocks=0`, `local_blocks=1`, float32 contiguous scores.

| Axis | Values |
| --- | --- |
| Batch size | 1, 8, 16, 32, 64, 128, 256, 512 |
| Logical/allocated blocks | 256, 512, 1024, 2048, 4096 |
| Shape count | Cartesian product: 40 |
| Warmup and samples | 20 warmup calls, 100 measured calls per stage/shape |
| Dynamic replay | `(512,4096),(1,256),(256,1024),(8,2048),(32,512),(1,4096)` |

Measure the full interface and prepare/TopK/postprocess separately using NPU
Events; report median/P95 in microseconds. Eager event intervals include launch
gaps and are not isolated hardware kernel durations. Stage TopK uses preallocated
outputs; the full interface retains its real allocation behavior, so stage
medians need not sum to the full-interface median. Data creation, copies and
correctness checks are outside measured intervals. Preparation is idempotent
for these fixed contexts, permitting repeated calls on its writable score input.

Record synchronized first-call wall time separately, including cache loading or
compilation. Track in-process Triton specialization-cache entries before/after
each shape and dynamic replay; unsupported cache introspection reports null.
This does not count `torch.topk`/CANN internal variants. Validate both full and
split-stage outputs for every shape and full outputs for every replay case.

The benchmark script is retained on the test host at
`/home/cxy-pr18294-20261011/msa_index_decode_topk.py`, mounted as
`/test-run/msa_index_decode_topk.py` in the test container. It is not part of this
repository. Run the following inside that container:

```bash
ASCEND_RT_VISIBLE_DEVICES=8 python /test-run/msa_index_decode_topk.py \
  --output-dir /test-run/artifacts/msa-benchmark --warmup 20 --repeats 100
```

The script writes `results.csv` and `summary.json`, including device/software
versions, measured latencies and cache-entry counts.

### Measured results (2026-10-11)

Environment: host `192.168.13.198`, one free logical NPU (device 8),
Ascend910_9362, image `f475d0b09e7b`, PyTorch `2.10.0+cpu` with torch-npu,
triton-ascend `3.2.0`. Other devices hosted an unrelated service. No model
server or C++ rebuild was needed for these operator measurements.

- Interface/accuracy tests: **41 passed**; MiniMax integration unit tests:
  **112 passed**.
- All **40** benchmark shapes passed reference checks for both the public
  interface and split stages; all **6** dynamic replay shapes also passed.
- Prepare and postprocess each retained **one** in-process specialization
  throughout the matrix and dynamic replay. This observation applies to the
  tested dtype/layout/model configuration, not every possible caller.
- First interface call: **1150.10 ms**, including initialization and compiled
  kernel loading/compilation. It is excluded from steady-state statistics.

Full-interface median latency in **microseconds**:

| Batch size | 256 blocks | 512 blocks | 1024 blocks | 2048 blocks | 4096 blocks |
| --- | --- | --- | --- | --- | --- |
| 1 | 226.94 | 224.83 | 222.57 | 223.02 | 222.50 |
| 8 | 220.08 | 225.63 | 223.25 | 227.20 | 250.70 |
| 16 | 226.19 | 222.91 | 232.64 | 262.67 | 492.62 |
| 32 | 224.25 | 246.12 | 290.09 | 516.36 | 914.44 |
| 64 | 246.65 | 360.33 | 599.29 | 956.89 | 1741.91 |
| 128 | 417.23 | 705.56 | 1098.84 | 1899.75 | 3495.10 |
| 256 | 803.18 | 1284.43 | 2182.67 | 3769.38 | 6886.55 |
| 512 | 1448.67 | 2535.32 | 4231.26 | 7562.76 | 13718.27 |

Selected stage medians and full-interface P95, in **microseconds**:

| Batch | Blocks | Prepare median | TopK median | Postprocess median | Full P95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 256 | 89.17 | 39.12 | 88.60 | 245.98 |
| 16 | 1024 | 135.83 | 37.27 | 87.16 | 242.20 |
| 64 | 2048 | 913.44 | 46.86 | 84.06 | 974.48 |
| 512 | 4096 | 13157.55 | 382.18 | 112.25 | 13840.78 |

At the largest shape, prepare dominates the measured cost. These results are
measurements of the current implementation, not a speedup claim: no alternative
implementation was benchmarked. The eager event measurements include launch
gaps, particularly significant for small shapes. There is no performance
acceptance threshold in this benchmark.

Raw CSV, JSON and logs are archived on the test host under
`/home/cxy-pr18294-20261011/artifacts/msa-benchmark/` (results) and
`/home/cxy-pr18294-20261011/artifacts/msa-benchmark.log` (per-case log).
