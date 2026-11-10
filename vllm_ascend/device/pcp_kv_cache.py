# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Select a supported implementation before importing PCP cache kernels."""

import torch

from vllm_ascend.utils import supports_triton


def copy_pcp_kv_cache(cache, slots):
    """Pack selected cache rows, preserving packed bytes and zeroing padding."""
    if supports_triton():
        from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache as triton_copy

        return triton_copy(cache, slots)

    assert len(cache) in (1, 2)
    key = cache[0]
    assert key.ndim == 4 and key.shape[2] == 1
    if len(cache) == 1:
        assert key.element_size() == 1
        parts = [key.view(torch.int8)]
    else:
        rope = cache[1]
        assert rope.ndim == 4 and rope.shape[2] == 1
        assert key.dtype == rope.dtype and key.shape[:3] == rope.shape[:3]
        parts = [key, rope]
    slots = slots.reshape(-1).to(torch.int64)
    safe_slots = slots.clamp_min(0)
    blocks, offsets = safe_slots // key.shape[1], safe_slots % key.shape[1]
    rows = [part[blocks, offsets, 0] for part in parts]
    packed = torch.cat(rows, dim=-1)
    return packed.masked_fill(slots.lt(0).unsqueeze(-1), 0)
