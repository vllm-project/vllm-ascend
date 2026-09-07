#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Fused head-major assembly of the DCP query-gather input.

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_heads",
        "nope_dim",
        "total_dim",
        "qn_stride_t",
        "qn_stride_h",
        "qp_stride_t",
        "qp_stride_h",
    ]
)
def _q_gather_prep_head_major_kernel(
    qn_ptr,  # [T, H, nope_dim] ql_nope (any strides)
    qp_ptr,  # [T, H, rope_dim] q_pe (any strides)
    out_ptr,  # [H, T, nope_dim + rope_dim] contiguous, head-major gather input
    num_tokens,
    num_heads,
    nope_dim,
    total_dim,
    qn_stride_t,
    qn_stride_h,
    qp_stride_t,
    qp_stride_h,
    BLOCK: tl.constexpr,
):
    """Assemble the fused query directly in head-major layout.

    One program per (head, token) row of the output. Reads the nope fragment
    from strided ``ql_nope`` and the rope fragment from strided ``q_pe`` and
    writes them into ``out[h, t, :]`` (contiguous), producing exactly the
    buffer that ``all_gather_into_tensor`` needs for the native-DCP head
    gather. This replaces the torch ``cat -> permute -> contiguous`` chain
    with a single kernel.
    """
    row = tl.program_id(0)
    head_idx = row // num_tokens
    token_idx = row % num_tokens
    offs = tl.arange(0, BLOCK)
    n_mask = offs < nope_dim
    p_mask = (offs >= nope_dim) & (offs < total_dim)
    src_base_n = token_idx * qn_stride_t + head_idx * qn_stride_h
    src_base_p = token_idx * qp_stride_t + head_idx * qp_stride_h
    dst_base = row * total_dim

    qn = tl.load(qn_ptr + src_base_n + offs, mask=n_mask, other=0)
    tl.store(out_ptr + dst_base + offs, qn, mask=n_mask)
    qp = tl.load(qp_ptr + src_base_p + (offs - nope_dim), mask=p_mask, other=0)
    tl.store(out_ptr + dst_base + offs, qp, mask=p_mask)


def prep_query_head_major(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
) -> torch.Tensor:
    """Return the fused query as a contiguous [H, T, nope+rope] tensor.

    Args:
        ql_nope: [T, H, nope_dim] partial query (rope-excluded).
        q_pe: [T, H, rope_dim] rope part.

    Returns:
        [H, T, nope_dim + rope_dim] contiguous tensor, in the layout
        ``all_gather_into_tensor`` produces for a head gather (each rank's
        head chunk is contiguous along dim 0).
    """
    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise RuntimeError(
            f"query gather prep requires matching (T, H), got {tuple(ql_nope.shape)} and {tuple(q_pe.shape)}"
        )
    if ql_nope.dtype != q_pe.dtype:
        raise RuntimeError("query gather prep requires ql_nope and q_pe to share a dtype")
    num_tokens, num_heads, nope_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    total_dim = nope_dim + rope_dim

    out = torch.empty(
        (num_heads, num_tokens, total_dim),
        dtype=ql_nope.dtype,
        device=ql_nope.device,
    )
    grid = (num_tokens * num_heads,)
    _q_gather_prep_head_major_kernel[grid](
        ql_nope,
        q_pe,
        out,
        num_tokens,
        num_heads,
        nope_dim,
        total_dim,
        ql_nope.stride(0),
        ql_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        BLOCK=next_power_of_2(total_dim),
        multibuffer=False,
    )
    return out
