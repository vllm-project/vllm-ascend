# SPDX-License-Identifier: Apache-2.0
"""Explicit byte-view diagnostics for tests; never used on the transfer path."""

import torch


def registered_page(backing: torch.Tensor, base: int, stride: int, length: int, block_id: int) -> torch.Tensor:
    """View registered bytes through owned storage, while the caller pins KV."""
    if block_id < 0:
        raise ValueError("block_id must be nonnegative")
    storage = backing.untyped_storage()
    offset = base + block_id * stride - storage.data_ptr()
    if offset < 0 or offset + length > storage.nbytes():
        raise ValueError("KV block is outside backing storage")
    return torch.empty(0, dtype=torch.uint8, device=backing.device).set_(storage, offset, (length,), (1,))
