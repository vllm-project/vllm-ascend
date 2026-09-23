# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place writes to paged caches with padded physical storage."""

import torch
import torch_npu


def write_pooled_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Scatter through a contiguous storage view without copying padded pages."""
    block_size, head_dim = cache.shape[1], cache.shape[-1]
    page_stride = cache.stride(0)
    if cache.stride()[-3:] != (head_dim, head_dim, 1) or page_stride % head_dim:
        raise ValueError("GLM KPool cache requires packed rows and a head-aligned page stride.")
    rows_per_page = page_stride // head_dim
    storage_rows = (cache.shape[0] - 1) * rows_per_page + block_size
    storage = cache.as_strided((storage_rows, head_dim), (head_dim, 1))
    valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
    storage_slots = slots
    if rows_per_page != block_size:
        storage_slots = torch.div(slots, block_size, rounding_mode="floor") * rows_per_page + slots % block_size
    storage_slots = torch.where(valid, storage_slots, -1)
    # Native scatter skips negative indices, including NaN update rows whose
    # pools have not completed. Physical slot zero remains a valid write.
    torch_npu.npu_scatter_nd_update_(storage, storage_slots.view(-1, 1), values.reshape(-1, head_dim))
