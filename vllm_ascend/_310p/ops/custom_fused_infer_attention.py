"""Ascend 310P custom fused infer attention (PagedAttention / incre-flash) op wrapper.

This wraps the custom CANN operator ``CustomFusedInferAttentionV310`` (exposed as
``torch.ops._C_ascend.npu_custom_fused_infer_attention_v310``) ported from the
Ascend custom-op project. It also provides the staircase compress-mask generator
used by the compress attention mode.
"""

import torch
from torch._dynamo import allow_in_graph

# head_dim -> (block_size, q_step) configuration for the compress staircase mask.
_FIA_COMPRESS_MASK_CFG: dict[int, tuple[int, int]] = {
    128: (128, 32),
    256: (64, 16),
}


@allow_in_graph
def gen_custom_fia_compress_mask(head_dim: int) -> torch.Tensor:
    """Generate the staircase compress mask for ``custom_fused_infer_attention_v310``.

    The mask controls attention during the compression stage. ``0.0`` keeps a
    position, ``torch.finfo(torch.float16).min`` (i.e. -inf) masks it out.

    Args:
        head_dim: per-head dimension, only 128 or 256 are supported.
            - head_dim = 128 -> block_size = 128, q_step = 32, shape (191, 128)
            - head_dim = 256 -> block_size = 64,  q_step = 16, shape (95, 64)

    Returns:
        A float16 mask tensor of shape ``(q_step + block_size - 1 + q_step,
        block_size)``.
    """
    if head_dim not in _FIA_COMPRESS_MASK_CFG:
        raise ValueError(f"Unsupported head_dim: {head_dim}. Only 128 or 256 are supported.")
    block_size, q_step = _FIA_COMPRESS_MASK_CFG[head_dim]

    block_b_rows = block_size - 1
    total_rows = q_step + block_b_rows + q_step

    # Initialize everything to -inf (masked).
    compress_mask = torch.full(
        (total_rows, block_size),
        torch.finfo(torch.float16).min,
        dtype=torch.float16,
    )

    # Lower-left triangular staircase for the middle rows.
    b_row_idx = torch.arange(block_b_rows).unsqueeze(1)  # (block_b_rows, 1)
    b_col_idx = torch.arange(block_size).unsqueeze(0)  # (1, block_size)
    staircase_mask = b_col_idx <= b_row_idx

    staircase_start = q_step
    staircase_end = q_step + block_b_rows
    compress_mask[staircase_start:staircase_end][staircase_mask] = 0.0
    # Bottom rows are fully kept.
    compress_mask[staircase_end:] = 0.0

    return compress_mask


@allow_in_graph
def custom_fused_infer_attention_v310(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    actual_seq_lengths_q: list[int] | None = None,
    actual_seq_lengths_kv: list[int] | None = None,
    block_table: torch.Tensor | None = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    input_layout: str = "BSND",
    num_key_value_heads: int = 0,
    block_size: int = 0,
) -> torch.Tensor:
    """Ascend 310P custom fused infer attention (PagedAttention / incre-flash).

    Wraps ``CustomFusedInferAttentionV310`` which is a 310P-specific attention
    kernel supporting paged KV-cache with block-table addressing. ONLY head-dim
    256 or 128 supported.

    Args:
        query: Query tensor.
            - dtype: float16
            - format: ND
            - TND layout: shape ``(T_q, num_heads, head_dim)``
            - BSND layout: shape ``(B, max_q_len, num_heads, head_dim)``
        key: Key cache tensor (single layer).
            - dtype: float16
            - format: ND or FRACTAL_NZ
            - ND shape: ``(num_blocks, C//16, block_size, 16)``
              where ``C = num_key_value_heads * head_dim``
            - NZ shape: same shape, format tag FRACTAL_NZ
            - Internally wrapped as ``[key]`` for the TensorList input.
        value: Value cache tensor (single layer). Same dtype/format/shape
            constraints as ``key``.
        attn_mask: Optional attention mask.
            - dtype: float16
            - format: ND
            - ``None`` means no mask (causal or full attention depending on
              the kernel path).
        actual_seq_lengths_q: Per-batch actual query sequence lengths.
            - List of ``B`` integers.
            - Default ``None`` treated as all ones (decode).
        actual_seq_lengths_kv: Per-batch actual KV sequence lengths.
            - List of ``B`` integers.
            - Default ``None`` treated as all ones.
        block_table: Paged-attention block table.
            - dtype: int32
            - format: ND
            - shape: ``(B, max_num_blocks_per_seq)``
            - ``-1`` marks unused (padding) entries.
        num_heads: Number of query heads. Must be >= num_key_value_heads and
            ``num_heads % num_key_value_heads == 0``.
        scale_value: Attention scale factor, typically ``head_dim ** -0.5``.
        input_layout: Input tensor layout string. Must be ``"BSND"`` or
            ``"TND"``.
        num_key_value_heads: Number of key/value heads (GQA). Defaults to 0
            (will be set to ``num_heads`` internally if not provided).
        block_size: KV-cache block size in tokens. Must be a multiple of 16.
            head_dim * block_size <= 128 * 128.

    Returns:
        Attention output tensor.
            - dtype: float16, format: ND
            - TND layout: shape ``(T_q, num_heads, head_dim)``
            - BSND layout: shape ``(B, max_q_len, num_heads, head_dim)``

    Note:
        ``inner_precise`` is fixed to 2 (high-precision softmax accumulate)
        and not exposed in this API.  It should not be changed without
        verifying numerical correctness on 310P.
    """
    return torch.ops._C_ascend.npu_custom_fused_infer_attention_v310(
        query,
        [key],
        [value],
        attn_mask=attn_mask,
        actual_seq_lengths_q=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=block_table,
        num_heads=num_heads,
        scale_value=scale_value,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        block_size=block_size,
        inner_precise=2,
    )
