# SPDX-License-Identifier: Apache-2.0
"""Compact C8 attention for sparse and 512-token sliding-window Decode.

Each row contains 512 E4M3 latent bytes, 64 BF16 RoPE coordinates and four
FP32 latent scales. Reinterpret its bytes for the CANN C8 ABI instead of
materializing FP32 KV tensors in HBM. Valid candidates must form a prefix;
all remaining slots are padding, and completely padded queries return zero.
"""

import torch
from cann_ops_transformer import kv_quant_sparse_flash_attention

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES


def attend_compact_c8(query, packed, selected, sink, scale):
    """Treat each verification slot as one independent sparse-attention batch."""
    queries, heads, _ = query.shape
    width = selected.shape[-1]
    lengths = (selected >= 0).sum(dim=-1, dtype=torch.int32)
    indices = torch.arange(width, device=query.device, dtype=torch.int32)
    indices = indices.view(1, 1, 1, width).expand(queries, 1, 1, width).contiguous()
    # Valid candidates precede padding in both the sorted sparse history and
    # current partial group. Explicit lengths exclude padded tail positions.
    kv = packed.view(queries, width, 1, C8_ROW_BYTES).view(torch.float8_e4m3fn)
    context, _, _ = kv_quant_sparse_flash_attention(
        query.view(queries, 1, heads, 576),
        kv,
        kv,
        indices,
        scale,
        2,
        2,
        actual_seq_lengths_kv=lengths.clamp_min(1),
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        sinks=sink.contiguous(),
    )
    return context[:, 0].masked_fill(lengths[:, None, None] == 0, 0)
