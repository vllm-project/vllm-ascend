"""Lazy A5 FlashAttention bindings from the CANN transformer package."""

from typing import Any

import torch


def flash_attn_metadata_size(batch_size: int, num_heads_kv: int) -> int:
    """Return the int32 element count required by ``flash_attn_metadata``."""
    metadata_size = ((36 + 72) * batch_size * num_heads_kv + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


def build_flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    output_buffer: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Build tiling metadata outside the attention graph.

    The import stays local because ``cann_ops_transformer`` is an A5 CANN
    component and is not available in every A2/A3 environment.
    """
    from cann_ops_transformer.ops import flash_attn_metadata  # type: ignore[import-not-found]  # noqa: PLC0415

    metadata = flash_attn_metadata(num_heads_q, num_heads_kv, head_dim, **kwargs)
    if output_buffer is None:
        return metadata
    if output_buffer.shape != metadata.shape:
        raise ValueError(
            "FlashAttention metadata buffer shape mismatch: "
            f"expected {tuple(metadata.shape)}, got {tuple(output_buffer.shape)}."
        )
    output_buffer.copy_(metadata)
    return output_buffer


def run_flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the A5 CANN FlashAttention operator with prebuilt metadata."""
    from cann_ops_transformer.ops import flash_attn  # type: ignore[import-not-found]  # noqa: PLC0415

    return flash_attn(query, key, value, **kwargs)
