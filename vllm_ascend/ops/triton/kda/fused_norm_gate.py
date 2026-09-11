# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Fused RMSNorm and sigmoid output gate for Kimi KDA on Ascend.

The Triton kernels below follow the supplied ``fused_norm_gate.py``
implementation.  The thin KDA wrapper at the end only adapts it to vLLM's
opaque custom-op boundary and K3's [1, T, H, D] output layout.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import direct_register_custom_op

# Maximum rows per Triton block for layernorm gated kernel.
MAX_ROWS_PER_BLOCK = 4


@triton.jit
def layer_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    num_valid_rows,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    # Fold ``_zero_padded_spec_output`` into the input load: rows beyond
    # ``num_valid_rows`` (graph-padding slots the recurrent kernel left
    # uninitialized) are zeroed here so RMSNorm drives them to 0 exactly as the
    # old ``torch.where(arange<n, x, 0)`` pre-pass did. ``num_valid_rows`` is a
    # pointer to a 0-d int tensor so the value can vary per graph replay.
    if HAS_NUM_VALID_ROWS:
        nvr = tl.load(num_valid_rows)
        valid_row = (i_t * BT + tl.arange(0, BT)) < nvr
        b_x = tl.where(valid_row[:, None], b_x, 0.0)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    if IS_RMS_NORM:
        b_x_hat = b_x * b_rstd[:, None]
    else:
        b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    num_valid_rows,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    # See ``layer_norm_gated_fwd_kernel``: zero graph-padding rows (one row
    # per program here) so the norm emits 0 for them, replacing the old
    # ``_zero_padded_spec_output`` pre-pass.
    if HAS_NUM_VALID_ROWS:
        nvr = tl.load(num_valid_rows)
        b_x = tl.where(i_t < nvr, b_x, 0.0)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    if IS_RMS_NORM:
        b_x_hat = b_x * b_rstd
    else:
        b_x_hat = (b_x - b_mean) * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    has_num_valid_rows = num_valid_rows is not None
    # Triton needs a real pointer even when the branch is unused; fall back to x.
    nvr_ptr = num_valid_rows if has_num_valid_rows else x

    if D <= 512:
        BT = 32
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_valid_rows=nvr_ptr,
            HAS_NUM_VALID_ROWS=has_num_valid_rows,
            num_warps=4,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_valid_rows=nvr_ptr,
            HAS_NUM_VALID_ROWS=has_num_valid_rows,
            num_warps=4,
        )
    return y, mean, rstd, residual_out if residual_out is not None else x


def _kda_rms_norm_sigmoid_gate_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt the supplied fused kernel to K3's per-head output layout.

    ``num_valid_rows`` (optional 0-d int tensor) is the number of *valid token
    rows* in the spec-decode contiguous path; the kernel folds
    ``_zero_padded_spec_output`` into the input load by zeroing rows beyond it.
    It is in token units and is scaled by the head count here, because the
    kernel iterates rows of the reshaped ``[num_tokens * num_heads, D]`` view
    (token-major, so valid rows = num_valid_tokens * num_heads).
    """
    if x.shape[-1] != gate.shape[-1]:
        raise ValueError(f"KDA output and gate head dimensions differ: {x.shape[-1]} != {gate.shape[-1]}")
    if x.numel() != gate.numel():
        raise ValueError(f"KDA output and gate element counts differ: {x.numel()} != {gate.numel()}")

    output_shape = x.shape
    feature_dim = output_shape[-1]
    if num_valid_rows is not None:
        assert x.ndim == 4 and x.shape[0] == 1, (
            "num_valid_rows expects KDA's [1, T, H, D] output layout")
        num_tokens = output_shape[1]
        num_heads = x.numel() // (num_tokens * feature_dim)
        nvr_rows = num_valid_rows * num_heads
    else:
        nvr_rows = None
    output, _, _, _ = layer_norm_gated_fwd(
        x=x.reshape(-1, feature_dim),
        g=gate.reshape(-1, feature_dim),
        weight=weight.contiguous(),
        bias=None,
        activation="sigmoid",
        eps=eps,
        out_dtype=x.dtype,
        is_rms_norm=True,
        num_valid_rows=nvr_rows,
    )
    return output.reshape(output_shape)


def _kda_rms_norm_sigmoid_gate_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    del gate, weight, eps, num_valid_rows
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="kda_rms_norm_sigmoid_gate",
    op_func=_kda_rms_norm_sigmoid_gate_impl,
    fake_impl=_kda_rms_norm_sigmoid_gate_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def apply_kda_rms_norm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call K3's opaque fused RMSNorm and sigmoid-gate operator."""
    return torch.ops.vllm.kda_rms_norm_sigmoid_gate(x, gate, weight, eps, num_valid_rows)
