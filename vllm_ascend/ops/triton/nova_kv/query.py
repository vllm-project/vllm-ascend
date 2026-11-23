# SPDX-License-Identifier: Apache-2.0
"""Project decode queries into the NOVA-KV key space with mean correction."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _unpack_projection(PROJECTED, QUERY, MEAN, B: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    offset = ((row // G) % H * B * G + row // (H * G) * G + row % G) * (D + 16)
    lane = tl.arange(0, D)
    tl.store(QUERY + row * D + lane, tl.load(PROJECTED + offset + lane))
    tl.store(MEAN + row, tl.load(PROJECTED + offset + D))


def prepare_query_projection(codec):
    if not hasattr(codec, "_query_projection"):
        weight = codec.key_inverse.transpose(-1, -2)
        mean = codec.key_mean.unsqueeze(-1)
        tail = torch.zeros((*mean.shape[:-1], 15), dtype=mean.dtype, device=mean.device)
        codec._query_projection = torch.cat((weight, mean, tail), dim=-1)


def project_query(query, codec, fuse_outputs=False):
    batch, heads, groups, dim = query.shape
    by_head = query.permute(1, 0, 2, 3).reshape(heads, batch * groups, dim)
    result = torch.bmm(by_head, codec._query_projection)
    if fuse_outputs:
        rotated = torch.empty_like(query)
        mean = torch.empty(query.shape[:-1], dtype=query.dtype, device=query.device)
        _unpack_projection[(batch * heads * groups,)](
            result,
            rotated,
            mean,
            batch,
            heads,
            groups,
            dim,
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
        )
        return rotated, mean
    result = result.reshape(heads, batch, groups, dim + 16).permute(1, 0, 2, 3)
    return result[..., :dim].contiguous(), result[..., dim].contiguous()
