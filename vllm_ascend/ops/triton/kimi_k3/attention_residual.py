"""Fused Kimi K3 attention-residual mixture.

This is the supplied Kimi K3 implementation, adapted only to use the
vLLM-Ascend Triton device-property helper.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["N", "B"])
def _apply_attn_res_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    N,
    H: tl.constexpr,
    B,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= N:
        return
    tok1 = tl.minimum(tok0 + block_size, N)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    s_idx = tl.arange(0, NB)

    norm_w = tl.load(norm_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = norm_w * proj_w

    br_stride = B * H

    for tok in range(tok0, tok1):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(
                    block_residual_ptr + tok * br_stride + s * H + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols, mask=mask, other=0.0).to(tl.float32)
            ms = tl.sum(v * v) / H
            rstd = tl.rsqrt(ms + EPS)
            k = v * rstd
            scores = tl.where(s_idx == s, tl.sum(k * w), scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        weights = exp_scores / tl.sum(exp_scores)

        out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(
                    block_residual_ptr + tok * br_stride + s * H + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols, mask=mask, other=0.0).to(tl.float32)
            w_s = tl.sum(tl.where(s_idx == s, weights, 0.0))
            out += w_s * v

        tl.store(out_ptr + tok * H + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: torch.nn.Module,
    norm: torch.nn.Module,
) -> torch.Tensor:
    """Return K3's learned softmax mixture of residual streams."""
    num_tokens, hidden_size = prefix_sum.shape
    num_blocks = block_residual.shape[1]
    proj_w = proj.weight.squeeze(0)
    norm_w = norm.weight
    eps = norm.variance_epsilon

    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=prefix_sum.dtype,
        device=prefix_sum.device,
    )
    num_streams = triton.next_power_of_2(num_blocks + 1)
    block_size = triton.next_power_of_2(hidden_size)
    init_device_properties_triton()
    num_vectorcore = get_vectorcore_num()
    _apply_attn_res_kernel[(num_vectorcore,)](
        block_residual,
        prefix_sum,
        norm_w,
        proj_w,
        out,
        N=num_tokens,
        H=hidden_size,
        B=num_blocks,
        EPS=eps,
        NUM_CORES=num_vectorcore,
        NB=num_streams,
        BLOCK_SIZE=block_size,
        multibuffer=True,
    )
    return out
