# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM


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
