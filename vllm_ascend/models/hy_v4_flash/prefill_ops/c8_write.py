# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.functional import pad
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.hisparse.c8 import pack_c8_components

if HAS_TRITON:
    from .c8_write_triton import write_c8_rows as write_c8_rows_triton


def write_c8_rows(cache, slots, latent, rope, scales, keys=None, positions=None, *, index_norm=None):
    """Pack quantizer outputs and write them to their unique physical slots."""
    if not slots.numel():
        return
    if HAS_TRITON and cache.device.type == "npu" and cache.is_contiguous():
        if keys is not None:
            assert positions is not None
            positions = positions.to(torch.int64)
            if positions.stride(-1) != 1:
                # A singleton slice can report contiguous with a non-unit stride.
                positions = positions.clone(memory_format=torch.contiguous_format)
        return write_c8_rows_triton(cache, slots, latent, rope, scales, keys, positions, index_norm=index_norm)

    if index_norm is not None:
        weight, inverse = index_norm
        keys = ((keys.float() * inverse[:, None]) * weight).bfloat16()
    payload = pack_c8_components(latent, rope.contiguous(), scales)
    if keys is not None:
        assert positions is not None
        coords = positions[:, None].expand(-1, 3) if positions.ndim == 1 else positions.T
        coords = coords.to(torch.int64).contiguous().view(torch.uint8)
        payload = torch.cat((payload, keys.contiguous().view(torch.uint8), coords), dim=-1)

    rows = pad(payload, (0, cache.shape[-1] - payload.shape[-1]))
    # vLLM reserves page zero. All padding writes restore identical bytes there.
    rows = torch.where(slots[:, None] >= 0, rows, cache[:1])
    cache.index_copy_(0, slots.long().clamp_min(0), rows)
    if index_norm is not None:
        return keys
