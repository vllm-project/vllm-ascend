# SPDX-License-Identifier: Apache-2.0
"""Q-only RoPE helper for Gemma4 MTP sliding-attention layers."""

import torch


def gemma4_q_only_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    """Apply RoPE to query only when key is None (K/V come from target cache)."""
    import torch_npu
    from vllm.triton_utils import HAS_TRITON

    query_shape = query.shape
    num_tokens = query.shape[0]
    if HAS_TRITON:
        from vllm_ascend.ops.triton.rope import rope_forward_triton

        query, _ = rope_forward_triton(
            query.view(num_tokens, -1, head_size),
            # Dummy key to satisfy the API; only query is rotated.
            torch.empty(num_tokens, 0, head_size, dtype=query.dtype, device=query.device),
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            rope_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        return query.view(query_shape)
    # Non-Triton fallback: rotate only query via the NPU rotary op.
    if rotary_dim < head_size:
        query = query.view(num_tokens, -1, head_size)
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        q_rot = q_rot.contiguous().view(num_tokens, -1)
        k_dummy = torch.empty_like(q_rot)
        torch_npu._npu_rotary_embedding(positions, q_rot, k_dummy, rotary_dim, cos_sin_cache, is_neox_style)
        q_rot = q_rot.view(num_tokens, -1, rotary_dim)
        return torch.cat((q_rot, q_pass), dim=-1).view(query_shape)
    query = query.contiguous().view(num_tokens, -1)
    k_dummy = torch.empty_like(query)
    torch_npu._npu_rotary_embedding(positions, query, k_dummy, head_size, cos_sin_cache, is_neox_style)
    return query.view(query_shape)
