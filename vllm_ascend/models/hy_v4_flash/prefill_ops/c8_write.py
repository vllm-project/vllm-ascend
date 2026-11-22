# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.functional import pad

from vllm_ascend.hisparse.c8 import pack_c8_components


def write_c8_rows(cache, slots, latent, rope, scales, keys=None, positions=None):
    """Pack quantizer outputs and write them to their unique physical slots."""
    if not slots.numel():
        return
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
