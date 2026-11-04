# MiniMax-M3 index TopK: decode index ordering

## Description

- **Implementation**: `vllm_ascend/ops/triton/msa/msa_m3_index_topk.py`.
- **Function**: `_decode_index_postprocess_sorted`, used by
  `minimax_m3_index_decode(sort_indices=True)`, fuses valid-ID filtering,
  counts, ascending logical-ID ordering and int32 output into one Triton launch.
  This is the A5 Triton indexer fallback; the current model default is CANN.
- **Formula**: For row `q`, request `b=q//Q` and offset `o=q%Q`,
  `V=ceil(max(seq_lens[b]-Q+o+1,0)/128)`. A slot is valid if
  `slot < min(K,V)` and `0 <= raw_id < V`. Output contains those IDs ascending,
  then `-1`; counts are the number of valid slots.
- **Algorithm flow**: normalize private input views if needed; mask/load each
  row; count valid IDs; sort FP32 keys using the shared `FP32_INDEX_SENTINEL`;
  restore invalid entries and store through the supplied output strides.
  TopK=16 processes four rows together; other widths use one row per program.
- **Supported modes**: Ascend A5 (950PR/950DT), subject to installed Triton and
  CANN support. A2/A3 model dispatch for this A5 indexer: N/A. Single-card
  correctness/performance can run independently of a multi-node deployment.

## Parameters

All helper parameters are required. The public producer additionally owns
index-score computation, TopK selection, max context and optional output.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `raw` | Input | TopK IDs `[H,B*Q,K]` | int64 from TopK; int32 supported by kernel | ND, strided supported |
| `seq_lens` | Input | Sequence lengths including queries, `[B]` | int32/int64 | ND, strided supported |
| `indices` | Output | Caller-owned sorted IDs `[H,B*Q,K]` | int32 | ND, supplied strides retained |
| `counts` | Output | Caller-owned valid counts `[H,B*Q]` | int32 | ND, supplied strides retained |
| `decode_query_len` | Attribute | Positive uniform query width `Q` | int | Scalar |

## Constraints

- Do not confuse `seq_lens` (includes queries) with the shared CANN helper's
  `context_lens` (before queries). All buffers share the same NPU; the helper
  is internal and relies on the public producer's input/shape contracts.
- Raw IDs and sequence values are read-only. Internal contiguous copies do
  not modify input dtype, data pointer or strides. Only `indices` and `counts`
  are written; they must not overlap read-only inputs. Public `out` keeps its
  storage address and only active rows/slots are overwritten.
- Zero-token input launches no kernel. FULL graph padded rows with zero
  sequence length yield `-1` and zero counts. Graph replay must refresh values
  at stable addresses; strided output guard regions remain untouched.
- The public producer rejects invalid output dtype/shape/device and an ID
  domain larger than `2**24`. The same named constant as the CANN helper is
  passed as a compile-time float key; valid IDs are below it and exact in FP32.
  External IDs/counts remain int32, with exact integer comparison required.
- Compile-time arguments: `BLOCK_SIZE`, `TOPK`, `ROWS`, `TILE`, `SENTINEL`.
  Tensor contents and ordinary scalar sizes/strides are runtime arguments;
  dtype/alignment/value-one specialization can still create extra binaries.
  Log actual keys and shapes on A5 before claiming bounded compilation.

## Origin and Differences

- **Origin**: MiniMax-M3 A5 Triton decode indexer.
- **Reason**: replace legacy invalid-ID postprocessing followed by a separate
  valid-prefix FP32 sort with fused GBSA-ready output. Permanent optimization;
  temporary patch replacement plan: N/A.
- **Differences**: score computation, `torch.topk`, KV dtype/quantization and
  model dispatch are unchanged. Existing caller output buffers are preserved.
  No global dtype/layout mutation or implicit production tracing is added.

## Test Cases

Model acceptance uses M3/M3.1's `sparse_topk_blocks=16` with their runtime shapes.
Other TopK helper fixtures are supplementary and are not a model-support claim.

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_topk.py
```

Tests cover heads 1/4; Q=1/4/8; TopK=1/3/16/31/32/64; block/context
boundaries; contiguous and strided inputs/outputs/counts; inactive guard slots;
input immutability; and changed inputs between graph replays. IDs and counts
must match exactly. Public guard/dispatch/empty-batch tests also live in
`tests/ut/models/minimax_m3/test_msa_m3.py`.

A5 single-machine model validation must hold TP/DP/PP, KV dtype, model/draft,
input files, seeds, operator libraries and generation parameters fixed between
baseline and candidate. The model must fit entirely; BF16/FP8 KV are separate
configurations. Both the CANN default and explicitly selected Triton fallback
need attribution. This does not validate multi-node communication or throughput.
Record the tested source hashes and runtime with each result. Reused model
accuracy scores must identify their source and actual backend; single-operator
exactness and trace coverage are reported separately.
