"""Fused Kimi K3 attention-residual mixture.

For every token, the operator RMS-normalizes each valid block residual and the
prefix-sum residual, projects them to scalar scores, applies a softmax across
those streams, and returns their weighted sum. ``block_residual`` follows
vLLM's preallocated ``[num_tokens, block_capacity, hidden_size]`` contract;
``num_valid_blocks`` identifies the initialized prefix of that capacity.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _apply_attn_res_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    BLOCK_CAPACITY: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    tl.static_assert(NB >= B + 1, "NB must include all block residuals and prefix_sum")
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= N:
        return
    tok1 = tl.minimum(tok0 + block_size, N)

    cols = tl.arange(0, H)
    s_idx = tl.arange(0, NB)
    block_residual_stride = BLOCK_CAPACITY * H

    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    for tok in range(tok0, tok1):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(block_residual_ptr + tok * block_residual_stride + s * H + cols).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols).to(tl.float32)
            ms = tl.sum(v * v) / H
            rstd = tl.rsqrt(ms + EPS)
            k = v * rstd
            scores = tl.where(s_idx == s, tl.sum(k * w), scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        weights = exp_scores / tl.sum(exp_scores)

        out = tl.zeros([H], dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(block_residual_ptr + tok * block_residual_stride + s * H + cols).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols).to(tl.float32)
            w_s = tl.sum(tl.where(s_idx == s, weights, 0.0))
            out += w_s * v

        tl.store(out_ptr + tok * H + cols, out.to(out_ptr.dtype.element_ty))


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: torch.nn.Module,
    norm: torch.nn.Module,
    num_valid_blocks: int,
) -> torch.Tensor:
    """Return K3's learned softmax mixture of residual streams."""
    num_tokens, hidden_size = prefix_sum.shape
    block_capacity = block_residual.shape[1]
    proj_w = proj.weight.squeeze(0)
    norm_w = norm.weight
    eps = norm.variance_epsilon

    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=prefix_sum.dtype,
        device=prefix_sum.device,
    )
    # The extra stream is prefix_sum, so NB must cover num_valid_blocks + 1.
    num_streams = triton.next_power_of_2(num_valid_blocks + 1)
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
        B=num_valid_blocks,
        BLOCK_CAPACITY=block_capacity,
        EPS=eps,
        NUM_CORES=num_vectorcore,
        NB=num_streams,
        multibuffer=True,
    )
    return out
