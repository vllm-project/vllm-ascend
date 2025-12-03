# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

MAX_CORES = 65535
UNIFIED_BUFFER_SIZE = 1572864

class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x,
                weight,
                bias,
                z=None,
                eps=1e-6,
                group_size=None,
                norm_before_gate=True,
                is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)
    

def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M, ), dtype=torch.float32,
                       device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()

    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")

    # SUB_BLOCK_M = 128 # 1835008 when N = 128, bf16 (overflow) -> assume factor 56
    # Assume large M
    HEURISTIC_FACTOR = 56
    SUB_BLOCK_M = triton.next_power_of_2(triton.cdiv(UNIFIED_BUFFER_SIZE, HEURISTIC_FACTOR * N) // x.element_size()) // 2
    num_cores = get_vectorcore_num()
    BLOCK_M = max(triton.next_power_of_2(triton.cdiv(triton.cdiv(M, SUB_BLOCK_M), num_cores)) * SUB_BLOCK_M, SUB_BLOCK_M)

    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (triton.cdiv(M, BLOCK_M), ngroups)
    layer_norm_fwd_kernel[grid](x,
                                out,
                                weight,
                                bias,
                                z,
                                mean,
                                rstd,
                                x.stride(0),
                                out.stride(0),
                                z.stride(0) if z is not None else 0,
                                M,
                                group_size,
                                eps,
                                BLOCK_N=BLOCK_N,
                                BLOCK_M=BLOCK_M,
                                SUB_BLOCK_M=SUB_BLOCK_M,
                                HAS_BIAS=bias is not None,
                                HAS_Z=z is not None,
                                NORM_BEFORE_GATE=norm_before_gate,
                                IS_RMS_NORM=is_rms_norm,
                                num_warps=num_warps)
    return out, mean, rstd


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SUB_BLOCK_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row_blk_idx = tl.program_id(0)
    group = tl.program_id(1)

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    W_inner = W + group * N + cols
    if HAS_BIAS:
        B_inner = B + group * N + cols

    w = tl.load(W_inner, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B_inner, mask=col_mask).to(tl.float32)

    for sub_row_blk in range(0, BLOCK_M // SUB_BLOCK_M):
        row = row_blk_idx * BLOCK_M + sub_row_blk * SUB_BLOCK_M + tl.arange(0, SUB_BLOCK_M)
        row_mask = row < M
        blk_mask = row_mask[:, None] & col_mask[None, :]

        X_inner = X + row[:, None] * stride_x_row + (group * N + cols)[None, :]
        Y_inner = Y + row[:, None] * stride_y_row + (group * N + cols)[None, :]
        if HAS_Z:
            Z_inner = Z + row[:, None] * stride_z_row + (group * N + cols)[None, :]
        if not IS_RMS_NORM:
            Mean_inner = Mean + group * M + row[:, None]
        Rstd_inner = Rstd + group * M + row[:, None]

        x = tl.load(X_inner, mask=blk_mask, other=0.).to(tl.float32)

        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_inner, mask=blk_mask).to(tl.float32)
            x *= z * tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1, keep_dims=True) / N
            tl.store(Mean_inner, mean, mask=row_mask[:, None])
            xbar = x - mean
            var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / N
        else:
            xbar = x
            var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / N

        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_inner, rstd, mask=row_mask[:, None])
        # Normalize and apply linear transformation

        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

        y = x_hat * w + b if HAS_BIAS else x_hat * w

        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_inner, mask=blk_mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(Y_inner, y, mask=blk_mask)