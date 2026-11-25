# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Single-launch GDN decode prepare kernel.

The eager GDN decode step submits five host dispatches per layer (three QKV
split-contiguous copies, one fused-gating Triton launch and two l2norm Triton
launches). Profiling on 910B2 C16 speculative decode showed the host work
between those dispatches leaves the NPU idle (~49% of all NPU idle time sits
inside ``AscendGatedDeltaNetAttention._forward_core``). This kernel folds the
whole prepare sequence into one Triton launch.

Numerics replicate ``l2norm_fwd_kernel2`` (per-head fp32 L2 norm with
``rsqrt``) and ``fused_gdn_gating_kernel`` (softplus gating with the same
beta/threshold formula) exactly, including output dtypes.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "beta", "threshold", "eps"])
def gdn_decode_prepare_kernel(
    mixed_ptr,  # [T, QKV_ROW] conv output, per token: q heads | k heads | v heads
    a_ptr,  # [T, NV]
    b_ptr,  # [T, NV]
    a_log_ptr,  # [NV]
    dt_bias_ptr,  # [NV]
    q_out_ptr,  # [T, NK, DK] l2-normalized
    k_out_ptr,  # [T, NK, DK] l2-normalized
    v_out_ptr,  # [T, NV, DV] plain copy
    g_out_ptr,  # [T, NV] fp32 gating output
    beta_out_ptr,  # [T, NV]
    num_tokens,
    NK: tl.constexpr,
    NK_BLOCK: tl.constexpr,
    NV: tl.constexpr,
    NV_BLOCK: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QKV_ROW: tl.constexpr,
    beta,
    threshold,
    eps,
):
    token = tl.program_id(0)
    d_off = tl.arange(0, DK)[None, :]
    hq = tl.arange(0, NK_BLOCK)[:, None]
    hq_mask = hq < NK
    qk_off = hq * DK + d_off
    row_base = token * QKV_ROW

    # q: per-head L2 norm, math identical to l2norm_fwd_kernel2.
    q_src = tl.load(mixed_ptr + row_base + qk_off, mask=hq_mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q_src * q_src, axis=1)
    q_norm = q_src * tl.rsqrt(q_var + eps)[:, None]
    tl.store(q_out_ptr + token * NK * DK + qk_off, q_norm.to(q_out_ptr.dtype.element_ty), mask=hq_mask)

    # k: same normalization on the k slice.
    k_src = tl.load(mixed_ptr + row_base + NK * DK + qk_off, mask=hq_mask, other=0.0).to(tl.float32)
    k_var = tl.sum(k_src * k_src, axis=1)
    k_norm = k_src * tl.rsqrt(k_var + eps)[:, None]
    tl.store(k_out_ptr + token * NK * DK + qk_off, k_norm.to(k_out_ptr.dtype.element_ty), mask=hq_mask)

    # v: plain copy of the v slice.
    hv = tl.arange(0, NV_BLOCK)[:, None]
    hv_mask = hv < NV
    v_off = hv * DV + d_off
    v_src = tl.load(mixed_ptr + row_base + 2 * NK * DK + v_off, mask=hv_mask, other=0.0)
    tl.store(v_out_ptr + token * NV * DV + v_off, v_src, mask=hv_mask)

    # gating: g = -exp(A_log) * softplus(a + dt_bias), beta_out = sigmoid(b).
    hh = tl.arange(0, NV_BLOCK)
    hh_mask = hh < NV
    gate_off = token * NV + hh
    a_val = tl.load(a_ptr + gate_off, mask=hh_mask, other=0.0).to(tl.float32)
    b_val = tl.load(b_ptr + gate_off, mask=hh_mask, other=0.0).to(tl.float32)
    a_log = tl.load(a_log_ptr + hh, mask=hh_mask, other=0.0).to(tl.float32)
    bias = tl.load(dt_bias_ptr + hh, mask=hh_mask, other=0.0).to(tl.float32)
    x = a_val + bias
    softplus_x = tl.where(beta * x <= threshold, (1.0 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    tl.store(g_out_ptr + gate_off, (-tl.exp(a_log) * softplus_x).to(g_out_ptr.dtype.element_ty), mask=hh_mask)
    tl.store(beta_out_ptr + gate_off, tl.sigmoid(b_val).to(beta_out_ptr.dtype.element_ty), mask=hh_mask)


def fused_gdn_decode_prepare(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    eps: float = 1e-6,
):
    """Fold QKV split + l2norm(q, k) + GDN gating into one launch.

    Returns ``(q, k, v, g, beta)`` with the same shapes/dtypes the original
    sequence produces (``[1, T, H, D]`` tensors and ``[1, T, NV]`` gates), or
    ``None`` when the inputs do not match the supported decode layout or run
    off the NPU so the caller falls back to the original path.
    """
    if mixed_qkv.device.type != "npu":
        # The kernel targets triton-ascend; CPU unit tests and profile runs
        # must keep the original dispatch sequence.
        return None
    if mixed_qkv.dim() != 2 or not mixed_qkv.is_contiguous():
        return None
    if a.dim() != 2 or not a.is_contiguous() or b.dim() != 2 or not b.is_contiguous():
        return None
    if A_log.dim() != 1 or dt_bias.dim() != 1:
        return None
    num_tokens = mixed_qkv.shape[0]
    qkv_row = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    if (
        mixed_qkv.shape[1] != qkv_row
        or a.shape[0] != num_tokens
        or b.shape[0] != num_tokens
        or a.shape[1] != num_v_heads
        or b.shape[1] != num_v_heads
        or A_log.shape[0] != num_v_heads
        or dt_bias.shape[0] != num_v_heads
        or num_tokens == 0
    ):
        return None

    q = torch.empty((num_tokens, num_k_heads, head_k_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty_like(q)
    v = torch.empty((num_tokens, num_v_heads, head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    g = torch.empty((num_tokens, num_v_heads), dtype=torch.float32, device=a.device)
    beta_out = torch.empty((num_tokens, num_v_heads), dtype=b.dtype, device=b.device)
    gdn_decode_prepare_kernel[(num_tokens,)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta_out,
        num_tokens,
        NK=num_k_heads,
        NK_BLOCK=triton.next_power_of_2(num_k_heads),
        NV=num_v_heads,
        NV_BLOCK=triton.next_power_of_2(num_v_heads),
        DK=head_k_dim,
        DV=head_v_dim,
        QKV_ROW=qkv_row,
        beta=beta,
        threshold=threshold,
        eps=eps,
    )
    return (
        q.view(1, num_tokens, num_k_heads, head_k_dim),
        k.view(1, num_tokens, num_k_heads, head_k_dim),
        v.view(1, num_tokens, num_v_heads, head_v_dim),
        g.view(1, num_tokens, num_v_heads),
        beta_out.view(1, num_tokens, num_v_heads),
    )
