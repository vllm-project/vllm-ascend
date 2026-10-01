# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.layernorm_gated_dispatch import (
    FT16_MAX_N_GROUP,
    DispatchConfigError,
    DispatchParams,
    _select_layernorm_launch,
)
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, get_vectorcore_num


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(do_not_specialize=["stride_x_row", "stride_y_row", "stride_z_row", "M", "N", "eps"])
def _layer_norm_fwd_1pass_kernel_npu(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    # Compute row indices for this program
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Mask for valid rows and cols
    row_mask = rows < M
    col_mask = cols < N

    # Load weight once (broadcasted over rows)
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    # Load X: shape [BLOCK_M, BLOCK_N]
    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

    # Load Z if needed
    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    # Compute statistics per row
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # [BLOCK_M]
        xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        tl.store(Mean + rows, mean, mask=row_mask)
    else:
        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
    tl.store(Rstd + rows, rstd, mask=row_mask)

    # Normalize
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]
    if HAS_BIAS:
        y += b[None, :]

    # Post-gate
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    # Store output
    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(
    do_not_specialize=[
        "stride_x_row",
        "stride_y_row",
        "stride_z_row",
        "M",
        "N",
        "eps",
        "num_m_blocks",
        "ngroups",
    ]
)
def _layer_norm_fwd_persistent_kernel_npu(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    num_m_blocks,
    ngroups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = num_m_blocks * ngroups

    for tile_id in range(pid, num_tiles, num_programs):
        pid_m = tile_id // ngroups
        group = tile_id - pid_m * ngroups
        mean_ptr = Mean
        rstd_ptr = Rstd + group * M
        w_ptr = W + group * N
        b_ptr = B
        if not IS_RMS_NORM:
            mean_ptr = Mean + group * M
        if HAS_BIAS:
            b_ptr = B + group * N

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N
        w = tl.load(w_ptr + cols, mask=col_mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + cols, mask=col_mask).to(tl.float32)

        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
            z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x *= z * tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            tl.store(mean_ptr + rows, mean, mask=row_mask)
        else:
            xbar = tl.where(col_mask[None, :], x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N

        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + rows, rstd, mask=row_mask)
        if not IS_RMS_NORM:
            x_hat = (x - mean[:, None]) * rstd[:, None]
        else:
            x_hat = x * rstd[:, None]
        y = x_hat * w[None, :]
        if HAS_BIAS:
            y += b[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y *= z * tl.sigmoid(z)
        y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(
    do_not_specialize=[
        "stride_x_row",
        "stride_y_row",
        "stride_z_row",
        "M",
        "N",
        "eps",
        "num_m_blocks",
    ]
)
def _layer_norm_fwd_persistent_hoist_kernel_npu(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    num_m_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    for tile_id in range(pid, num_m_blocks, num_programs):
        rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :]
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :]
            z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x *= z * tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            tl.store(Mean + rows, mean, mask=row_mask)
        else:
            xbar = tl.where(col_mask[None, :], x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)
        if not IS_RMS_NORM:
            x_hat = (x - mean[:, None]) * rstd[:, None]
        else:
            x_hat = x * rstd[:, None]
        y = x_hat * w[None, :]
        if HAS_BIAS:
            y += b[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y *= z * tl.sigmoid(z)
        y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :]
        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
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
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    runtime_p = None
    ub_bytes = None
    if getattr(getattr(x, "device", None), "type", None) == "npu":
        runtime_p = get_vectorcore_num()
        if 128 < group_size <= FT16_MAX_N_GROUP:
            ub_bytes = get_ub_size_bytes()
    spec = _select_layernorm_launch(
        M,
        group_size,
        ngroups,
        runtime_p,
        _LAYERNORM_GATED_EXPERIMENTAL_PARAMS,
        ub_bytes=ub_bytes,
    )

    # BASE selections reuse the upstream kernel and feature-dimension guard.
    # Non-NPU and unqualified wide-N inputs retain BLOCK_M=64; qualified NPU
    # inputs may use a smaller row tile.
    if spec.impl == "FT_BASE":
        max_fused_size = 65536 // x.element_size()
        block_n = min(max_fused_size, triton.next_power_of_2(group_size))
        if group_size > block_n:
            raise RuntimeError(
                f"layer_norm_fwd_npu: Feature dim too large, got {group_size}, max supported is {block_n}."
            )
        grid = (triton.cdiv(M, spec.block_m), ngroups)
        _layer_norm_fwd_1pass_kernel_npu[grid](
            x,
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
            BLOCK_M=spec.block_m,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    if runtime_p is None:
        raise DispatchConfigError("persistent selection requires an initialized vector-core count")
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(group_size))

    if spec.impl == "FT_PERSIST":
        num_m_blocks = triton.cdiv(M, spec.block_m)
        num_tiles = num_m_blocks * ngroups
        grid = (min(runtime_p, num_tiles),)
        _layer_norm_fwd_persistent_kernel_npu[grid](
            x,
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
            num_m_blocks,
            ngroups,
            BLOCK_M=spec.block_m,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    if spec.impl == "FT_PERSIST_HOIST":
        if ngroups != 1:
            raise DispatchConfigError("FT_PERSIST_HOIST requires ngroups == 1")
        num_m_blocks = triton.cdiv(M, spec.block_m)
        grid = (min(runtime_p, num_m_blocks),)
        _layer_norm_fwd_persistent_hoist_kernel_npu[grid](
            x,
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
            num_m_blocks,
            BLOCK_M=spec.block_m,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    raise DispatchConfigError(f"impl {spec.impl} is unsupported or unmaterialized")


def _layer_norm_gated_experimental_params():
    return DispatchParams(
        bm_small=16,
        bm_multi=32,
        k_persist_num=1,
        k_persist_den=4,
        n_persist_min=128,
        hoist_qualified=True,
        persist_single_qualified=True,
    )


_LAYERNORM_GATED_EXPERIMENTAL_PARAMS = _layer_norm_gated_experimental_params()
