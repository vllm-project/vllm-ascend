# MiniMax-M3 A5 indexer and sparse attention

## Description

- **Function:** A5 block scoring, prefill/decode TopK, and paged sparse attention.
- **Formula:** Same causal index-block maximum and selected-block softmax
  attention as [msa_m3_triton.md](msa_m3_triton.md).
- **Algorithm flow:** Compute block scores, force initial/local windows, select
  TopK, mask invalid IDs, and compute sparse attention. Decode also produces the
  valid selection count per head/query.
- **Supported modes:** Ascend A5; runner-independent model kernels.

## Parameters

The five public wrapper names and common arguments are documented in
[msa_m3_triton.md](msa_m3_triton.md#parameters). A5 differences are:

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| Decode return value | Output | `(topk_idx, select_num_idx)` | int32 | `[kv_heads, tokens, topk]`, `[kv_heads, tokens]` |
| `out` | Output | Optional caller-owned ID buffer | int32 | `[kv_heads, >=tokens, topk]`; active rows retain the buffer address |
| `kv_cache` | Input | Fused or split main caches, normalized to KV axis 1 | BF16 or supported FP8 | `[pages, 2, 128, kv_heads, head_dim]`, KV axis 0, or a K/V pair |
| `max_decode_query_len` | Attribute | Bound for the runtime decode width | int | Scalar; defaults to `decode_query_len` |
| Decode sharding / `return_scores` | Attribute | N/A: these A2/A3 optional arguments are not supported | N/A | N/A |

## Constraints

- Sparse block size is 128, and index-head count equals KV-head count. Decode
  token count equals `requests * decode_query_len`.
- Invalid IDs are `-1`; counts equal the number of valid IDs, including padded
  requests and contexts crossing a block boundary.
- Production-shaped tests use one 128-wide index/KV head, eight query heads,
  and TopK16. The separate A5 API and launch policy are intentionally retained.
- Graph replay tests keep geometry and buffer addresses fixed while changing
  sequence-length tensor values. Caller-owned output guards must remain intact.

## Origin and Differences

- **Origin:** MiniMax M3 `common/ops/sparse_attn.py`.
- **Differences:** A5 score and split-K launch policy, fused-cache adaptation,
  and decode valid-count output. Relocation changes no kernel bodies or APIs.

## Test Cases

The A5-specific suite directly imports this implementation. It compares selected
scores and counts exactly against a CPU reference, tests strided output guards,
and replays graphs with changed lengths. Sparse attention is compared with CPU
softmax attention for prefill/decode and three KV layouts, using the same BF16
mean/max absolute error limits (`2.5e-4` / `1.7e-2`) as the existing M3 suite.
CPU-only layout and output-buffer tests also exercise the A5 launch wrappers.

```bash
pytest -q tests/ut/ops/test_msa_m3_triton.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_triton_a5.py
```
