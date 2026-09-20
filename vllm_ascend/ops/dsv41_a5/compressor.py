# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packaged A5 CompressorV2 adapter."""

from __future__ import annotations

import torch

from .package_loader import import_packaged_a5_module


def compressor_v2(x, wkv, wgate, state_cache, metadata, out):
    """Run CompressorV2 and restore its packed rows to token alignment."""
    import_packaged_a5_module("cann_ops_transformer.ops.attention.compressor_v2.compressor")
    controls = metadata.c2_ring_metadata
    complete = metadata.c2_complete_mask[: x.shape[0]]
    packed = torch.ops.cann_ops_transformer.ds41.compressor(
        x.contiguous(),
        wkv.detach().contiguous(),
        wgate.detach().contiguous(),
        state_cache,
        controls[4].contiguous(),
        metadata.query_start_loc.contiguous(),
        controls[1].contiguous(),
        controls[0].contiguous(),
        2,
    )
    if x.shape[0]:
        packed_index = complete.to(torch.int64).cumsum(0) - 1
        gathered = packed[packed_index.clamp_min(0)]
        out.copy_(torch.where(complete[:, None], gathered, torch.zeros_like(gathered)))
    return out
