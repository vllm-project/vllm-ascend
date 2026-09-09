# SPDX-License-Identifier: Apache-2.0
"""Peripheral address conversion for parent-backed SFA caches."""

import torch


def paged_cache_token_addresses(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Return addresses for validated physical slots without packing a cache."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[1] <= 0:
        raise ValueError("SFA cache must have shape (blocks, tokens, 1, channels)")
    if slots.dtype != torch.int64:
        raise ValueError("SFA physical slots must be int64")
    block, token = slots.div(cache.shape[1], rounding_mode="floor"), slots.remainder(cache.shape[1])
    return cache.data_ptr() + (block * cache.stride(0) + token * cache.stride(1)) * cache.element_size()


def remap_sfa_source_addresses(
    addresses: torch.Tensor,
    num_misses: int,
    *,
    k_base: int,
    rope_base: int,
    k_token_bytes: int,
    rope_token_bytes: int,
) -> None:
    """Convert legacy split-cache source descriptors to token-concat addresses.

    The existing CPU planner still generates split-cache offsets and unchanged
    destination/length descriptors. Only its two active source slices are
    rewritten here, before sparse_copy consumes them. No operator ABI changes.
    """
    if addresses.device.type != "cpu" or addresses.dtype != torch.int64 or addresses.ndim != 1:
        raise ValueError("Source descriptors must be a CPU int64 vector")
    if num_misses < 0 or 2 * num_misses > addresses.numel():
        raise ValueError("Invalid source descriptor count")
    if k_token_bytes <= 0 or rope_token_bytes <= 0:
        raise ValueError("SFA component byte sizes must be positive")
    parent_token_bytes = k_token_bytes + rope_token_bytes
    for source, base, token_bytes in (
        (addresses[:num_misses], k_base, k_token_bytes),
        (addresses[num_misses : 2 * num_misses], rope_base, rope_token_bytes),
    ):
        source.sub_(base).div_(token_bytes, rounding_mode="floor").mul_(parent_token_bytes).add_(base)
