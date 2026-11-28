"""PCA primitives: custom paged-cache prepare/restore and CANN addmm.

Quantization belongs to the worker, after projection and before reconstruction.
These functions always accept/return tensors; they have no codec mode options.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def kvtc_compress(
    caches: Sequence[torch.Tensor],
    block_ids: torch.Tensor,
    projected_bias_bf16: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Prepare values and project to BF16 [S,B,C]."""
    if not caches:
        raise ValueError("kvtc_compress: caches must not be empty")
    prepared = torch.ops._C_ascend.kvtc_prepare(list(caches), block_ids)
    return torch.addmm(projected_bias_bf16, prepared, projection).reshape(
        block_ids.numel(), caches[0].shape[1], projection.shape[1]
    )


def kvtc_compress_rope(
    caches: Sequence[torch.Tensor],
    block_ids: torch.Tensor,
    token_starts: torch.Tensor,
    inv_cos_sin_cache: torch.Tensor,
    projected_bias_bf16: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse NeoX RoPE and project keys to BF16 [S,B,C]."""
    if not caches:
        raise ValueError("kvtc_compress_rope: caches must not be empty")
    prepared = torch.ops._C_ascend.kvtc_prepare_rope(list(caches), block_ids, token_starts, inv_cos_sin_cache)
    return torch.addmm(projected_bias_bf16, prepared, projection).reshape(
        block_ids.numel(), caches[0].shape[1], projection.shape[1]
    )


def kvtc_decompress(
    compressed: torch.Tensor,
    dst_block_ids: torch.Tensor,
    mu_bf16: torch.Tensor,
    basis_t: torch.Tensor,
    caches: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct values and scatter into caches in-place."""
    if not caches:
        raise ValueError("kvtc_decompress: caches must not be empty")
    reconstructed = torch.addmm(mu_bf16, compressed.reshape(-1, compressed.shape[-1]), basis_t)
    return torch.ops._C_ascend.kvtc_restore(reconstructed, dst_block_ids, list(caches))


def kvtc_decompress_rope(
    compressed: torch.Tensor,
    dst_block_ids: torch.Tensor,
    token_starts: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mu_bf16: torch.Tensor,
    basis_t: torch.Tensor,
    caches: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct keys, apply forward NeoX RoPE and scatter in-place."""
    if not caches:
        raise ValueError("kvtc_decompress_rope: caches must not be empty")
    reconstructed = torch.addmm(mu_bf16, compressed.reshape(-1, compressed.shape[-1]), basis_t)
    return torch.ops._C_ascend.kvtc_restore_rope(
        reconstructed, dst_block_ids, token_starts, cos_sin_cache, list(caches)
    )


__all__ = [
    "kvtc_compress",
    "kvtc_compress_rope",
    "kvtc_decompress",
    "kvtc_decompress_rope",
]
