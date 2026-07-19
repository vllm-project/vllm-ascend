# SPDX-License-Identifier: Apache-2.0
"""Q-only RoPE helper for Gemma4 MTP sliding-attention layers.

Lives in the ops layer (not the spec_decode proposer) so that
``ops/rotary_embedding`` does not have to import from the business layer.
"""

import torch


def gemma4_q_only_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    """Apply RoPE to query only, for Gemma4 MTP Q-only attention layers.

    Gemma4 MTP's sliding-attention layers are Q-only: K/V come from the
    target model's KV cache, so the draft passes key=None to RoPE.  This is
    Gemma4-MTP-specific (no other model passes key=None), hence the helper
    is isolated in its own ops module rather than the shared rotary op.

    Returns the rotated query; the caller discards the (None) key.
    """
    import torch_npu
    from vllm.triton_utils import HAS_TRITON

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
        return query
    # Non-Triton fallback: reuse the NPU rotary op with a throwaway key buffer
    # so only the query is rotated. Mirrors the normal (key is not None) path
    # and avoids hand-rolling cos/sin pairs, which is easy to get wrong for
    # both neox and interleaved styles.
    if rotary_dim < head_size:
        query = query.view(num_tokens, -1, head_size)
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        q_rot = q_rot.contiguous().view(num_tokens, -1)
        k_dummy = torch.empty_like(q_rot)
        torch_npu._npu_rotary_embedding(positions, q_rot, k_dummy, rotary_dim, cos_sin_cache, is_neox_style)
        q_rot = q_rot.view(num_tokens, -1, rotary_dim)
        return torch.cat((q_rot, q_pass), dim=-1).flatten(-2, -1)
    query = query.contiguous().view(num_tokens, -1)
    k_dummy = torch.empty_like(query)
    torch_npu._npu_rotary_embedding(positions, query, k_dummy, head_size, cos_sin_cache, is_neox_style)
    return query.view(num_tokens, -1, head_size).flatten(-2, -1)
