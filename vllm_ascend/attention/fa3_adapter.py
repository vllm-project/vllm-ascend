"""
flash-attention-npu (FA3) adapter for vllm-ascend.

Maps vllm-ascend's attention parameter formats to the flash-attention-npu
API so FA3 can replace CANN's npu_fused_infer_attention_score (V1 FIA) for
decode attention. Prefill keeps the CANN FIA path (see attention_v1.py);
learnable-sink layers always stay on CANN FIA v2.

Data format differences bridged here:
  - vllm cumulative seq lengths WITHOUT a leading zero  -> FA3 cu_seqlens WITH leading zero
  - vllm ``(num_blocks, block_size, -1)`` flat cache    -> FA3 ``(num_blocks, block_size, H, D)``
  - vllm ``sliding_window``                             -> FA3 ``window_size``
"""

from typing import List, Optional

import torch

try:
    from flash_attn_npu_3 import (
        flash_attn_with_kvcache as fa3_kvcache,
        get_scheduler_metadata,
    )
    HAS_FLASH_ATTN_NPU = True
except ImportError:
    HAS_FLASH_ATTN_NPU = False

# FA3 kernel compilation limit (round_up_headdim in flash-attention-npu).
FA3_MAX_HEAD_DIM = 256


def _to_cu_seqlens(actual_seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """vllm cumulative lengths (no leading 0) -> FA3 cu_seqlens (leading 0)."""
    return torch.tensor([0] + actual_seq_lengths, dtype=torch.int32, device=device)


def _max_seqlen(cumulative: List[int]) -> int:
    """Maximum *individual* sequence length from a cumulative list."""
    max_len = cumulative[0]
    for prev, cur in zip(cumulative, cumulative[1:]):
        max_len = max(max_len, cur - prev)
    return max_len


def fa3_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_metadata,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window: Optional[int] = None,
    causal: bool = True,
    block_table: Optional[torch.Tensor] = None,
    seq_lens_list: Optional[List[int]] = None,
    scheduler_metadata=None,
) -> torch.Tensor:
    """Paged-KV-cache decode attention via ``flash_attn_with_kvcache``.

    *key*/*value* are paged cache views ``(num_blocks, block_size, -1)``;
    *block_table* and *seq_lens_list* (per-request KV lengths) are required.
    *scheduler_metadata* pre-computed via :func:`get_scheduler_metadata`
    avoids re-computation inside the FA3 op.

    Returns a tensor of shape ``(total_q_tokens, num_heads, head_size)``.
    """
    if not HAS_FLASH_ATTN_NPU:
        raise ImportError("flash-attention-npu is not installed")
    if head_size > FA3_MAX_HEAD_DIM:
        raise ValueError(
            f"flash-attention-npu supports head_dim <= {FA3_MAX_HEAD_DIM}, "
            f"got {head_size}"
        )
    assert block_table is not None, "block_table is required for FA3 decode"
    assert seq_lens_list is not None, "seq_lens_list is required for FA3 decode"

    device = query.device
    # Non-causal attention matches CANN's sparse_mode=0 mapping: always a full
    # window, ignoring sliding_window (avoids a local bias on cross-attention).
    if causal and sliding_window is not None:
        window_size = (sliding_window, 0)
    else:
        window_size = (-1, -1)

    num_blocks, bs = key.shape[0], key.shape[1]
    k_fa = key.view(num_blocks, bs, num_kv_heads, head_size)
    v_fa = value.view(num_blocks, bs, num_kv_heads, head_size)

    return fa3_kvcache(
        query,
        k_fa,
        v_fa,
        cache_seqlens=torch.tensor(seq_lens_list, dtype=torch.int32, device=device),
        page_table=block_table.contiguous(),
        cu_seqlens_q=_to_cu_seqlens(attn_metadata.actual_seq_lengths_q, device),
        max_seqlen_q=_max_seqlen(attn_metadata.actual_seq_lengths_q),
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        scheduler_metadata=scheduler_metadata,
    )
