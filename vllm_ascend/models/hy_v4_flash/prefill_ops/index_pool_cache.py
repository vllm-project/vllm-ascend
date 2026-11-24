# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM

from .paged_cache_copy import copy_paged_cache_lanes, supports_paged_cache_copy


def _pool_key_addresses(cache, block_table, groups, live, block_size, ratio):
    lanes_per_row = INDEX_DIM // ratio
    lane = torch.arange(INDEX_DIM, device=cache.device)
    token = (groups * ratio).masked_fill(~live, 0)
    block = block_table.index_select(0, token // block_size).long()
    slot = block * block_size + token % block_size
    row_width = cache.shape[-1]
    row = slot[:, None] + lane[None, :] // lanes_per_row
    column = row_width - lanes_per_row + lane % lanes_per_row
    return row * row_width + column


def read_prefill_pool(cache, block_table, groups, block_size, ratio):
    """Read full IndexPool groups from the BF16 lanes at the end of each cache row."""
    lanes_per_row = INDEX_DIM // ratio
    groups_per_page = block_size // ratio
    if (
        ratio == 4
        and 0 < groups <= block_table.numel() * groups_per_page
        and supports_paged_cache_copy(cache, block_table, block_size)
    ):
        row_bytes = lanes_per_row * torch.bfloat16.itemsize
        packed = copy_paged_cache_lanes(cache, block_table, groups * ratio, cache.size(-1) - row_bytes, row_bytes)
        return packed.view(torch.bfloat16).view(groups, INDEX_DIM)
    cache = cache.view(torch.bfloat16)
    pages_to_read = (groups + groups_per_page - 1) // groups_per_page
    page_lanes = cache.view(-1, block_size, cache.shape[-1])[..., -lanes_per_row:]
    pages = page_lanes.index_select(0, block_table[:pages_to_read])
    return pages.reshape(-1, INDEX_DIM)[:groups]


def gather_pool_keys(cache, block_table, groups, live, block_size, ratio):
    """Gather the IndexPool lanes that are spread across each group's rows."""
    cache = cache.view(torch.bfloat16)
    addresses = _pool_key_addresses(cache, block_table, groups, live, block_size, ratio)
    addresses = addresses.masked_fill(~live[:, None], 0)
    values = cache.flatten().index_select(0, addresses.flatten()).view(-1, INDEX_DIM)
    return values.masked_fill(~live[:, None], 0)


def copy_pool_keys(cache, block_table, keys, first_group, block_size, ratio, *, write, end_group=None):
    if not keys.shape[0]:
        return
    if (
        write
        and end_group is None
        and ratio == 4
        and isinstance(first_group, int)
        and first_group >= 0
        and (first_group + keys.shape[0]) * ratio <= block_table.numel() * block_size
        and keys.ndim == 2
        and keys.shape[1] == INDEX_DIM
        and keys.is_contiguous()
        and keys.dtype in (torch.bfloat16, torch.float32)
        and keys.device == cache.device
        and supports_paged_cache_copy(cache, block_table, block_size)
    ):
        from ..index_pool_cache import copy_pool_keys as write_paged_pool

        write_paged_pool(cache, block_table, keys, first_group, block_size, ratio, write=True)
        return
    groups = torch.arange(keys.shape[0], device=cache.device) + first_group
    live = groups >= 0 if end_group is None else (groups >= 0) & (groups < end_group)
    if write:
        cache = cache.view(torch.bfloat16)
        addresses = _pool_key_addresses(cache, block_table, groups, live, block_size, ratio)
        # Offset zero is outside the IndexPool lanes, even on a dummy page.
        addresses = addresses.masked_fill(~live[:, None], 0)
        flat = cache.flatten()
        values = torch.where(live[:, None], keys.bfloat16(), flat[:1])
        flat.index_copy_(0, addresses.flatten(), values.flatten())
    else:
        keys.copy_(gather_pool_keys(cache, block_table, groups, live, block_size, ratio))
