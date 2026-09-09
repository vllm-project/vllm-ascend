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
        "rope_dim",
        "total_dim",
        "qn_stride_t",
        "qn_stride_h",
        "qp_stride_t",
        "qp_stride_h",
    ]
)
def _q_gather_prep_head_major_kernel(
    qn_ptr,  # [T, H, nope_dim] ql_nope (any non-negative strides)
    qp_ptr,  # [T, H, rope_dim] q_pe (any non-negative strides)
    out_ptr,  # [H, T, nope_dim + rope_dim] contiguous, head-major gather input
    num_tokens,
    num_heads,
    nope_dim,
    rope_dim,
    total_dim,
    qn_stride_t,
    qn_stride_h,
    qp_stride_t,
    qp_stride_h,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Assemble the fused query directly in head-major layout.

    One program per (head, token) row of the output. Writes the fused row
    ``[nope | rope]`` into ``out[h, t, :]`` (contiguous), producing exactly
    the buffer that ``all_gather_into_tensor`` needs for the native-DCP head
    gather. This replaces the torch ``cat -> permute -> contiguous`` chain
    with a single kernel.

    Each fragment uses its own ``tl.arange`` index space, so pointer
    arithmetic is never negative (no ``offs - nope_dim``) and no per-lane
    offset selects are required. The wrapper only launches this kernel for
    shapes it can prove legal (all addresses, including masked tail lanes,
    inside the tensor storages); any other shape falls back to the torch
    assembly in the caller (``prep_query_head_major`` raises).
    """
    row = tl.program_id(0)
    head_idx = row // num_tokens
    token_idx = row % num_tokens
    dst_base = row * total_dim
    src_base_n = token_idx * qn_stride_t + head_idx * qn_stride_h
    src_base_p = token_idx * qp_stride_t + head_idx * qp_stride_h

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < nope_dim
    qn = tl.load(qn_ptr + src_base_n + offs_n, mask=n_mask, other=0)
    tl.store(out_ptr + dst_base + offs_n, qn, mask=n_mask)

    offs_p = tl.arange(0, BLOCK_R)
    p_mask = offs_p < rope_dim
    qp = tl.load(qp_ptr + src_base_p + offs_p, mask=p_mask, other=0)
    tl.store(out_ptr + dst_base + nope_dim + offs_p, qp, mask=p_mask)


def _qualifies_fast_path(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    block_n: int,
    block_r: int,
    rope_dim: int,
    total_dim: int,
) -> bool:
    """Host-side proof that every address the fast kernel computes is legal.

    Loads and stores only use non-negative offsets (per-fragment index
    spaces), so the remaining risk is a masked tail lane pointing past the
    end of a tensor storage. Returns True iff no such lane exists for any
    (head, token) row:
      - strides are non-negative (no mirrored views);
      - the two store tails stay inside the [H, T, total] allocation, i.e.
        BLOCK_N <= total_dim and nope_dim + BLOCK_R <= total_dim;
      - loads: the last row base + BLOCK - 1 never passes the storage end,
        for both fragments.
    """
    if (
        ql_nope.stride(0) < 0
        or ql_nope.stride(1) < 0
        or q_pe.stride(0) < 0
        or q_pe.stride(1) < 0
    ):
        return False
    if block_n > total_dim or ql_nope.shape[-1] + block_r > total_dim:
        return False
    num_tokens, num_heads = ql_nope.shape[:2]
    for tensor, block in ((ql_nope, block_n), (q_pe, block_r)):
        last_base = (num_tokens - 1) * tensor.stride(0) + (num_heads - 1) * tensor.stride(1)
        if last_base + block > tensor.numel():
            return False
    return True


def prep_query_head_major(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
) -> torch.Tensor | None:
    """Return the fused query as a contiguous [H, T, nope+rope] tensor.

    Args:
        ql_nope: [T, H, nope_dim] partial query (rope-excluded).
        q_pe: [T, H, rope_dim] rope part.

    Returns:
        [H, T, nope_dim + rope_dim] contiguous tensor, in the layout
        ``all_gather_into_tensor`` produces for a head gather (each rank's
        head chunk is contiguous along dim 0); or ``None`` when the shapes
        cannot be proven legal for the kernel (see ``_qualifies_fast_path``),
        in which case callers fall back to the torch ``cat -> permute ->
        contiguous`` assembly. No guarded Triton kernel is used.

    Raises:
        RuntimeError: on genuinely invalid inputs (mismatched (T, H) or
        dtype); those cannot be assembled by the torch fallback either.
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
    if num_tokens == 0 or num_heads == 0:
        return out
    block_n = next_power_of_2(nope_dim)
    block_r = next_power_of_2(rope_dim)
    if not _qualifies_fast_path(ql_nope, q_pe, block_n, block_r, rope_dim, total_dim):
        return None
    grid = (num_tokens * num_heads,)
    _q_gather_prep_head_major_kernel[grid](
        ql_nope,
        q_pe,
        out,
        num_tokens,
        num_heads,
        nope_dim,
        rope_dim,
        total_dim,
        ql_nope.stride(0),
        ql_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        multibuffer=False,
    )
    return out
