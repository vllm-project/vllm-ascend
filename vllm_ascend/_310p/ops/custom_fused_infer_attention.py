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
        raise ValueError(
            f"Unsupported head_dim: {head_dim}. Only 128 or 256 are supported."
        )
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
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    block_size: int = 0,
    inner_precise: int = 1,
) -> torch.Tensor:
    """Ascend 310P custom fused infer attention.

    Args:
        query: query tensor.
        key / value: key/value (single) cache tensors; passed as single-element
            lists to the dynamic-list inputs of the custom op.
        attn_mask: optional attention mask.
        actual_seq_lengths_q / actual_seq_lengths_kv: optional per-batch actual
            sequence lengths.
        block_table: optional paged-attention block table.
        num_heads: number of query heads.
        scale_value: attention scale.
        input_layout: input layout, e.g. ``"BSH"``/``"BSND"``/``"TND"``.
        num_key_value_heads: number of key/value heads (GQA).
        block_size: paged-attention block size.
        inner_precise: inner-precise flag.

    Returns:
        Attention output tensor.
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
        inner_precise=inner_precise,
    )
