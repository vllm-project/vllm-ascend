# MiniMax-M3 indexer and sparse attention

## Description

- **Function:** Compute paged index-key block scores, select visible sparse blocks,
  and apply grouped-query attention to the selected main K/V blocks.
- **Formula:** For query position `p` and block `b`, the index score is
  `max(q_index @ k_index[t])` over visible tokens `t` in block `b`.
  Sparse attention computes `softmax(q @ K_selected.T * sm_scale) @ V_selected`.
- **Algorithm flow:** Compute causal block scores; prioritize initial/local
  blocks; select TopK IDs and mask invalid entries; gather selected KV pages;
  compute attention. Decode uses split-K and combines partial attention outputs.
- **Supported modes:** A2/A3 model path; runner-independent Triton implementation.
  The A5 production implementation is documented in [msa_m3_triton_a5.md](msa_m3_triton_a5.md).

## Parameters

The module exports `minimax_m3_index_score`, `minimax_m3_index_topk`,
`minimax_m3_index_decode`, `minimax_m3_sparse_attn`, and
`minimax_m3_sparse_attn_decode`. Attributes apply to the corresponding wrapper.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `idx_q` / `q` | Input | Index queries / attention queries | Floating point | `[tokens, heads, head_dim]` |
| `index_kv_cache` | Input | Paged index keys | Floating point | `[pages, 128, head_dim]`; a singleton head axis or split-cache input is also accepted |
| `kv_cache` | Input | Paged main keys and values | BF16 or supported FP8 | Split `[pages, 128, kv_heads, head_dim]` tensors, or a fused tensor with KV axis 0 or 1 |
| `block_table` | Input | Logical-to-physical page IDs | int32 | `[requests, max_blocks]` |
| `cu_seqlens_q` | Input | Cumulative query lengths | int32 | `[requests + 1]` |
| `seq_lens`, `prefix_lens` | Input | Total / previously computed token counts | Integer | `[requests]` |
| `score` | Input/Output | Block-score workspace; TopK preparation mutates it | FP32 | `[index_heads, tokens, aligned_blocks]` |
| `topk_idx` | Input | Selected logical blocks; invalid entries are `-1` | int32 | `[kv_heads, tokens, topk]` |
| `out` / `output` | Output | Optional caller-owned index buffer / required attention output | int32 / query dtype | `[index_heads, >=tokens, topk]` / `[tokens, query_heads, head_dim]` |
| `topk`, `init_blocks`, `local_blocks` | Attribute | Selection width and forced windows | int | Scalars |
| `max_query_len`, `max_seq_len` | Attribute | Query/context bounds | int | Scalars |
| `num_kv_heads`, `sm_scale` | Attribute | KV heads / attention scale; index ordering ignores the scale | int / float | Scalars |
| `decode_query_len`, `max_decode_query_len` | Attribute | Decode query width / padded bound | int | Scalars; the padded bound defaults to the query width |
| `block_offset`, `block_count`, `global_seq_lens` | Attribute/Input | Optional sharded score range and original sequence lengths | int / integer tensor | Scalars / `[requests]` |
| `return_scores` | Attribute | Return selected scores alongside decode IDs | bool | Scalar; default false |

## Constraints

- Sparse block size is 128. Index-head count equals KV-head count. Decode token
  count is `requests * decode_query_len`, with the query width within its bound.
- Model regression uses head dimension 128 and TopK16, including TP8 and TP16
  per-rank shapes. Floating-point attention uses the BF16 tolerance below.
- TopK outputs are int32 IDs with `-1` padding. Optional output buffers retain
  their address and unselected guard regions; missing selected scores use `-inf`.
- Graph replay requires compatible shapes and caller-managed input/output
  lifetimes. Changing the model geometry can specialize a new Triton variant.

## Origin and Differences

- **Origin:** MiniMax M3 `common/ops/index_topk.py` and `sparse_attn.py`.
- **Differences:** Ascend launch sizing and split-cache handling. The modules
  were relocated from the model package without changing kernel bodies or APIs.

## Test Cases

- CPU wrapper tests cover cache layouts, invalid shapes, short/empty TopK rows,
  strided caller-owned buffers, guards, and score padding.
- NPU index IDs are checked against an independent reference. BF16 attention
  requires mean absolute error at most `2.5e-4` and maximum at most `1.7e-2`.

```bash
pytest -q tests/ut/ops/test_msa_m3_triton.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_minimax_m3_sparse_attn.py
```
