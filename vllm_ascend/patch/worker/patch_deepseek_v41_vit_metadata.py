# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.deepseek_v4.common.vision import _compute_vision_cos_sin, get_vision_cos_sin
from vllm.models.deepseek_v41.common import vl_cudagraph as _deepseek_v41_vl_cudagraph
from vllm.utils.torch_utils import async_tensor_h2d


def build_packed_vit_metadata(
    grids: list[list[int]] | list[tuple[int, int]],
    *,
    rope_dim: int,
    rope_theta: float,
    device: torch.device,
    max_seqlen_override: int | None = None,
    cached: bool = True,
) -> dict[str, torch.Tensor]:
    cos_sin = get_vision_cos_sin if cached else _compute_vision_cos_sin
    cos_list: list[torch.Tensor] = []
    sin_list: list[torch.Tensor] = []
    cu_seqlens = [0]
    max_seqlen = 0
    for n_h, n_w in grids:
        cos, sin = cos_sin(n_h, n_w, rope_dim, rope_theta)
        cos_list.append(cos)
        sin_list.append(sin)
        n = n_h * n_w
        cu_seqlens.append(cu_seqlens[-1] + n)
        max_seqlen = max(max_seqlen, n)
    if cos_list:
        cos = async_tensor_h2d(torch.cat(cos_list), device)
        sin = async_tensor_h2d(torch.cat(sin_list), device)
    else:
        cos = torch.zeros((0, 1, rope_dim), dtype=torch.float32, device=device)
        sin = torch.zeros((0, 1, rope_dim), dtype=torch.float32, device=device)
    if max_seqlen_override is not None:
        max_seqlen = max_seqlen_override
    return {
        "vit_cos": cos,
        "vit_sin": sin,
        # Keep cu_seqlens on CPU; FIA consumes host endpoints at task update.
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device="cpu"),
        "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),
    }


_deepseek_v41_vl_cudagraph.build_packed_vit_metadata = build_packed_vit_metadata
