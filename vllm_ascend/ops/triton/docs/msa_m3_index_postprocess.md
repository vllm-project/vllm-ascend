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
- **Supported modes**: shared CANN MiniMax-M3 decode producer on A2/A3/A5,
  subject to installed Triton/runtime sort support. Hardware results must
  identify the tested revision/device; importability does not establish validation.
  The interface supports single-token decode and uniform multi-token decode.

## Parameters

All parameters are required except `init_blocks` and `local_blocks`, which default
to zero, and keyword-only `sort_indices`, which defaults to False.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `score` | Input/workspace | Scores `[H, B * Q, W]`; may be modified in place | float32 | ND, strided supported |
| `context_lens` | Input | Context lengths before the current queries, `[B]` | int32/int64 | ND, strided supported |
| `decode_query_len` | Attribute | Uniform query length `Q > 0` | Python int | Scalar |
| `topk` | Attribute | Number of output slots `K > 0` | Python int | Scalar |
| `logical_block_count` | Attribute | Logical block-table width, from 0 to `W` | Python int | Scalar |
| `init_blocks` | Attribute | Non-negative initial-block count | Python int | Scalar |
| `local_blocks` | Attribute | Non-negative local-block count | Python int | Scalar |
| `sort_indices` | Attribute | Ascending logical-ID order, default False | bool | Scalar |
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
- Contexts must fit int32 arithmetic including the query offset and block
  rounding. MiniMax-M3's tested maximum is 263000 tokens.
- Context values and input dtype, shape, strides and storage addresses are
  read-only. Internal `.contiguous()` copies do not rewrite caller metadata.
- Widths, query length and strides use Triton's default specialization.
  Model/algorithm constants (`TOPK`, `INIT_BLOCKS`, `LOCAL_BLOCKS`, `BLOCK_SIZE`,
  `TILE`, `ROWS`, `SORT_INDICES`, `SENTINEL`, where applicable) use `tl.constexpr`.
  Pointer dtype, scalar value-one and alignment hints can still create variants.
  Tensor contents do not become compile-time keys. No zero-recompilation claim
  follows merely from a missing `tl.constexpr` annotation.
- The caller supplies correctly updated metadata for ACL graph replay. This
  validation measures eager execution; it does not certify all graph shapes.

## Origin and Differences

- **Origin**: MiniMax-M3 A2/A3 decode helpers originally under
  `models/minimax_m3/ops`, moved to `ops/triton/msa` for shared operator ownership.
- **Differences**: Fuses score preparation and index/count postprocessing into
  two Triton kernels around `torch.topk`; ordered mode also incorporates the
  GBSA valid-prefix sort into postprocessing. AscendC score computation, the existing
  `msa_m3_triton.py` kernels and TP projection logic are unchanged.
- **Reason**: remove separate valid-ID normalization, sort and conversion
  launches before GBSA, without changing the selected TopK set. This is a
  permanent optimization; temporary patch replacement plan: N/A.

## Test Cases

Model integration and performance acceptance target the M3/M3.1 configuration
`sparse_topk_blocks=16`. Vary query length, batch size, context, dtype and padding
within that model contract. Existing helper boundary fixtures are supplementary;
they do not establish support for other models or arbitrary TopK configurations.

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
| Model geometry | Local B=5/Q=8/128K; B=10/Q=8/64K; historical B=11/Q=4/64K; maximum context 263000; Q=9/16 |
| Repeated model TopK | H=4/B=2/Q=8/K=16/W=33, repeated allocations, independent CPU oracle, failure tensor artifact |
| Input side effects | Allowed score-workspace writes; strided backing guards; context and input metadata unchanged |

Integer IDs/counts require 100% exact agreement (`rtol=0, atol=0`). Floating-point
tolerances must not excuse a wrong block ID. Selected scores are checked exactly.
For ties introduced by forced scores, compare selected scores, uniqueness and
counts instead of requiring a particular ordering of equally ranked block IDs.

```bash
ASCEND_RT_VISIBLE_DEVICES=8 python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py
```

## Ordered output and shared sentinel

`sort_indices=True` orders valid block IDs inside the existing postprocess
launch using FP32 keys, then restores trailing `-1`. TopK=16 uses four rows per
program and 32 internal sort slots; other widths use a general per-row kernel.
TopK=1 skips sorting because a single selected ID is already ordered.
`logical_block_count <= FP32_INDEX_SENTINEL` is required. A3/A5 share the single
definition in `ops/triton/msa/msa_m3_index_postprocess.py`: FP32 represents integers through
`2**24` exactly; that finite endpoint is reserved for invalid keys, and valid IDs
are strictly below it. It is supported by the Ascend sort extension and is not
an attention score. IDs and counts remain int32.

Both the CANN producer and A5 Triton fallback enable ordered output. Only matching
GBSA consumers skip normalization; direct callers and prefill retain their
correctness fallback. The A5-specific contract is documented in
[decode index ordering](msa_m3_index_topk.md).

## Performance and compilation validation

Compare legacy postprocessing plus the production valid-prefix normalizer with
fused ordering; separately compare complete CANN TopK producer paths. Inputs and
compiler/runtime must match. Exclude compilation, input generation and correctness
checks from timing; alternate baseline/candidate order and report latency distributions.
Model accuracy and clean GSM8K throughput require separate evidence. Reused
GPQA results must identify the tested source and backend; a score from the A5
Triton fallback does not certify a different CANN-default model configuration.

For each kernel record shapes, strides, dtypes, scalar/model constants, JIT
cache misses, completed compilation/load events and distinct specialization keys.
Use a new private cache for cold counts and mark warmup/capture/inference phases.
Report new keys after warmup; do not equate a filesystem cache-file count with
actual compilations. Graph replay can bypass Python JIT launches entirely.
Shape/compile instrumentation must not contaminate clean performance measurements.
Do not add `do_not_specialize` solely to hide unexplained variants: verify the
installed compiler API and measure the effects first.

Record source hashes, device/compiler versions, numerical results and timing
scope with each validation report. Operator-only traces do not establish model
shape coverage, and completed trace shutdown alone does not prove kernel calls.
