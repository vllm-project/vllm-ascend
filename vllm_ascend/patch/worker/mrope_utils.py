from typing import TypeGuard

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def can_use_fused_mrope(rotary_emb: torch.nn.Module, num_kv_heads: int) -> TypeGuard[AscendMRotaryEmbedding]:
    if not HAS_TRITON or not isinstance(rotary_emb, AscendMRotaryEmbedding):
        return False
    sections = rotary_emb.mrope_section
    # The Triton kernel rotates two half rows and uses unpadded power-of-two tiles.
    return (
        rotary_emb.is_neox_style
        and sections is not None
        and len(sections) == 3
        and min(sections) >= 0
        and 2 * sum(sections) == rotary_emb.rotary_dim
        and 0 < rotary_emb.rotary_dim <= rotary_emb.head_size
        and all(
            size > 0 and (size & (size - 1)) == 0
            for size in (rotary_emb.head_size, rotary_emb.rotary_dim, num_kv_heads)
        )
    )


def get_fused_mrope_inputs(
    rotary_emb: AscendMRotaryEmbedding,
    positions: torch.Tensor,
    qkv: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return the cached rows or the positions and frequencies for the fused op."""
    inv_freq = rotary_emb.inv_freq
    if (
        positions.ndim == 2
        and positions.device == qkv.device
        and inv_freq is not None
        and inv_freq.device == qkv.device
        and inv_freq.dtype == torch.float32
    ):
        return None, positions, inv_freq

    if positions.ndim == 1:
        # Text positions share the same index on all three axes.
        positions = positions.unsqueeze(0).expand(3, -1)
    return rotary_emb.cos_sin_cache[positions].to(qkv), None, None
