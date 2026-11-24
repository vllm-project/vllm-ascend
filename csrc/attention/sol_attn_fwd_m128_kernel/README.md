# Sol-Attn on Ascend 910B

Sol-Attn is a noncausal self-attention operator for contiguous BF16 Q, K, and V tensors with the same `[B, T, H, 128]` shape on one Ascend 910B device. It computes exact attention for selected 64-token KV blocks and represents the other blocks by their key and value means.

```python
from vllm_ascend.ops.sol_attn import sol_attn

output, lse = sol_attn(q, k, v, tau=1.0, return_lse=True)
```

The output has the same shape and dtype as Q. The optional LSE is base-2 log-sum-exp with shape `[B, H, T]` and dtype FP32. `scale` defaults to `128**-0.5`. `tau` sets the diagonal-statistics routing threshold. `sink_tokens` and `sink_start` specify a range of tokens whose KV blocks are always exact. When `sink_start` is omitted, the range ends at the last token.

The Python entry point selects preparation and forward launches from the sequence length, batch-head count, and autocast state. Eligible short inputs use a single mixed Cube/Vector launch with direct K/V reads. Other inputs with one KV group can fuse QKV preparation and routing. Inputs with two to four KV groups can fuse diagonal statistics, scoring, and route packing after QKV summaries. The general path computes centroid scores with head-chunked `torch.bmm` before route packing. The general forward uses paired 128-key Cube tiles and a four-slot Cube/Vector pipeline.

The underlying `torch.ops._C_ascend` operators write to caller-owned buffers.

| Operator | Output buffers |
| --- | --- |
| `sol_attn_prepare_qkv` | Padded K/V, BF16 K/V means, FP32 Q/K means, and tail block lengths |
| `sol_attn_prepare_diag_thresh` | FP32 threshold per query block |
| `sol_attn_prepare_route_pack` | UINT64 approximate block masks, INT32 exact block counts, and UINT16 exact block indices |
| `sol_attn_prepare_short` | Padded K/V, K/V means, tail block lengths, thresholds, and routes for one KV group |
| `sol_attn_prepare_route_stats` | FP32 thresholds and routes for two to four KV groups |
| `sol_attn_short_fwd_m128` | BF16 output and FP32 LSE from a single launch |
| `sol_attn_fwd_m128` | BF16 output and FP32 LSE, with the LSE token stride rounded up to a multiple of eight |

For direct calls, the threshold, route pack, and forward operators use `scale_log2 = scale * log2(e)`. The wrapper computes the buffer dimensions and exposes LSE over the valid token range.
