# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Raw KV storage allocation with allocator ownership and alignment preserved."""

from collections.abc import Callable
from typing import TypeAlias

import torch

ByteAllocator: TypeAlias = Callable[[int], torch.Tensor]


def allocate_zeroed_byte_tensor(
    size: int,
    device: torch.device,
    *,
    alignment: int = 1,
    allocate: ByteAllocator | None = None,
) -> torch.Tensor:
    """Allocate zeroed INT8 storage, retaining a supplied allocator's backing.

    The callback receives the full physical byte count, including alignment
    slack, and owns allocation policy. Returned views retain that storage;
    no replacement allocation or copy is used to repair an invalid callback.
    Typed K/V and scale views are created separately by the cache layout.
    """
    if size < 0 or alignment < 1:
        raise ValueError("KV storage size must be nonnegative and alignment must be positive.")
    device = torch.device(device)
    physical_size = size + alignment if alignment > 1 else size
    raw = torch.empty(physical_size, dtype=torch.int8, device=device) if allocate is None else allocate(physical_size)
    if not isinstance(raw, torch.Tensor):
        raise TypeError("KV byte allocator must return a Tensor.")
    if raw.dtype not in (torch.int8, torch.uint8) or raw.numel() != physical_size or not raw.is_contiguous():
        raise ValueError("KV byte allocator must return the requested number of contiguous one-byte elements.")
    if raw.device.type != device.type or (device.index is not None and raw.device.index != device.index):
        raise ValueError("KV byte allocator returned storage on the wrong device.")
    raw = raw.view(torch.int8).view(-1)
    raw.zero_()
    offset = (-raw.data_ptr()) % alignment
    return raw.narrow(0, offset, size)
