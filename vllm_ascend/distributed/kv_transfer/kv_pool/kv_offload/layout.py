# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Physical page views shared by Ascend offload connectors."""

from collections.abc import Sequence

import torch


def _canonicalize_physical_pages(
    cache_parts: Sequence[torch.Tensor],
    num_blocks: int,
    page_size_bytes: int,
) -> torch.Tensor:
    """Validate kernel-block geometry and expose complete scheduler pages."""
    if any(not isinstance(part, torch.Tensor) for part in cache_parts):
        raise TypeError("An Ascend KV cache must contain only tensors")
    parts = tuple(part for part in cache_parts if part.numel())
    if not parts:
        raise ValueError("An Ascend KV cache must contain a non-empty tensor")
    if num_blocks <= 0 or page_size_bytes <= 0 or any(part.ndim == 0 for part in parts):
        raise ValueError("Physical KV cache pages require positive sizes and a block dimension")

    first = min(parts, key=lambda part: part.data_ptr())
    factor, remainder = divmod(first.shape[0], num_blocks)
    stride_bytes = first.stride(0) * first.element_size()
    storage_ptr = first.untyped_storage().data_ptr()
    if remainder or not factor or stride_bytes <= 0 or page_size_bytes > stride_bytes * factor:
        raise ValueError("Physical KV cache pages must fit complete, non-overlapping blocks")
    for part in parts:
        offset = part.data_ptr() - first.data_ptr()
        span_bytes = (
            1 + sum((size - 1) * stride for size, stride in zip(part.shape[1:], part.stride()[1:]))
        ) * part.element_size()
        if (
            part.device != first.device
            or part.untyped_storage().data_ptr() != storage_ptr
            or part.stride(0) * part.element_size() != stride_bytes
            or part.shape[0] != num_blocks * factor
            or offset + (factor - 1) * stride_bytes + span_bytes > page_size_bytes
        ):
            raise ValueError("Physical KV cache components must share one page layout and backing storage")

    raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(first.untyped_storage())
    return torch.as_strided(
        raw,
        (num_blocks, page_size_bytes),
        (stride_bytes * factor, 1),
        first.storage_offset() * first.element_size(),
    )
