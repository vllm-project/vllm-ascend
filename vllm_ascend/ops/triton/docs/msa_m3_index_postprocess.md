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
- Widths, query length and strides use Triton's default specialization.
  Model constants (`TOPK`, `INIT_BLOCKS`, `LOCAL_BLOCKS`, `BLOCK_SIZE` and
  `TILE`, where applicable) use `tl.constexpr`. Changes to input properties or
  model constants can require a different compiled specialization.
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
