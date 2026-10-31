# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read individual tokens from strided ND MLA caches without repacking the full cache."""

import torch

from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache

PCP_COPY_MAX_TILE_ROWS = 8


def load_mla_cache_rows_by_slots(cache_k, cache_v, slots):
    """Read selected ND MLA rows without making the source cache contiguous."""
    slots = slots.reshape(-1).to(torch.int64)
    num_tokens = slots.numel()
    # The existing A3 batched copy can raise MTE errors on a partial final tile.
    # Repeat the first slot to fill the tile; the kernel zero-fills negative
    # slots and clamps their addresses, including when every slot is -1.
    padded_tokens = (num_tokens + PCP_COPY_MAX_TILE_ROWS - 1) // PCP_COPY_MAX_TILE_ROWS * PCP_COPY_MAX_TILE_ROWS
    if padded_tokens != num_tokens:
        slots = torch.cat((slots, slots[:1].expand(padded_tokens - num_tokens)))
    return copy_pcp_kv_cache((cache_k, cache_v), slots)[:num_tokens]
