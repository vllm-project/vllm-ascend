# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""Fused RMSNorm and output-gate kernel for Kimi KDA on Ascend.

The forward kernel is synced from flash-linear-attention commit 31d15f7554,
file ``fla/modules/backends/triton_ascend/fused_norm_gate.py``.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .ascend_ub_utils import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    compute_ub_block_size,
    get_multiprocessor_count,
)

_BD_MEM_MULT = 6.0
_FWD_MEM_MULT = 3.0
_UB_SAFETY_MARGIN = 0.85
_FALLBACK_MAX_BD = 65536 // 4
_MAX_BT = 128
_LARGE_BD = 2048
_LARGE_BD_FWD_MEM_MULT = 4.0

_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SWISH
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def _fwd_memory_multiplier(block_dim: int) -> float:
    if block_dim >= _LARGE_BD:
        return max(_FWD_MEM_MULT, _LARGE_BD_FWD_MEM_MULT)
    return _FWD_MEM_MULT


def _get_layer_norm_gated_tiles(feature_dim: int) -> tuple[int, int]:
    block_dim = compute_ub_block_size(
        feature_dim,
        _BD_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=triton.next_power_of_2(feature_dim),
    )
    if feature_dim > block_dim:
        raise RuntimeError(
            f"LayerNormGated feature dim {feature_dim} exceeds "
            f"UB-safe block size {block_dim}. Column-tiled kernels are "
            "not yet implemented for this size."
        )
    block_rows = compute_row_tile_block_size(
        1 << 20,
        block_dim,
        _fwd_memory_multiplier(block_dim),
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=16,
        min_block=1,
        max_block=_MAX_BT,
    )
    return block_dim, block_rows


def _launch_config(
    num_rows: int,
    feature_dim: int,
    device_index: int,
) -> tuple[int, int, int]:
    block_dim, block_rows = _get_layer_norm_gated_tiles(feature_dim)
    num_tiles = triton.cdiv(num_rows, block_rows)
    num_streams = max(
        1,
        min(
            get_multiprocessor_count(device_index),
            num_tiles,
            ASCEND_MAX_GRID_DIM,
        ),
    )
    return block_dim, block_rows, num_streams


@triton.jit(do_not_specialize=["T"])
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
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Grid-stride forward: each program owns a BT-row tile stream."""
    stream_index = tl.program_id(0)
    columns = tl.arange(0, BD)
    column_mask = columns < D

    if HAS_WEIGHT:
        block_weight = tl.load(w + columns, mask=column_mask).to(tl.float32)
    if HAS_BIAS:
        block_bias = tl.load(b + columns, mask=column_mask).to(tl.float32)

    num_tiles = tl.cdiv(T, BT)
    for tile_index in range(stream_index, num_tiles, NS):
        rows = tile_index * BT + tl.arange(0, BT)
        row_mask = rows < T
        mask = row_mask[:, None] & column_mask[None, :]
        row_offsets = rows[:, None] * D + columns[None, :]

        block_x = tl.load(x + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            block_x += tl.load(residual + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL_OUT:
            tl.store(
                residual_out + row_offsets,
                block_x.to(residual_out.dtype.element_ty),
                mask=mask,
            )

        if not IS_RMS_NORM:
            block_mean = tl.sum(block_x, axis=1) / D
            tl.store(mean + rows, block_mean, mask=row_mask)
            block_xbar = tl.where(mask, block_x - block_mean[:, None], 0.0)
            block_var = tl.sum(block_xbar * block_xbar, axis=1) / D
        else:
            block_xbar = tl.where(mask, block_x, 0.0)
            block_var = tl.sum(block_xbar * block_xbar, axis=1) / D
        block_rstd = 1 / tl.sqrt(block_var + eps)
        tl.store(rstd + rows, block_rstd, mask=row_mask)

        block_x_hat = (
            (block_x - block_mean[:, None]) * block_rstd[:, None] if not IS_RMS_NORM else block_x * block_rstd[:, None]
        )
        block_y = block_x_hat * block_weight[None, :] if HAS_WEIGHT else block_x_hat
        if HAS_BIAS:
            block_y = block_y + block_bias[None, :]

        block_gate = tl.load(g + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == 0:
            block_y = block_y * block_gate * tl.sigmoid(block_gate)
        else:
            block_y = block_y * tl.sigmoid(block_gate)

        tl.store(y + row_offsets, block_y.to(y.dtype.element_ty), mask=mask)


def layer_norm_gated_fwd_npu(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    num_rows, feature_dim = x.shape
    if g.shape != (num_rows, feature_dim):
        raise ValueError(f"gate shape must be {(num_rows, feature_dim)}, got {tuple(g.shape)}")
    if residual is not None and residual.shape != (num_rows, feature_dim):
        raise ValueError(f"residual shape must be {(num_rows, feature_dim)}, got {tuple(residual.shape)}")
    if weight is not None and weight.shape != (feature_dim,):
        raise ValueError(f"weight shape must be {(feature_dim,)}, got {tuple(weight.shape)}")
    if bias is not None and bias.shape != (feature_dim,):
        raise ValueError(f"bias shape must be {(feature_dim,)}, got {tuple(bias.shape)}")

    output = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_output = torch.empty(
            num_rows,
            feature_dim,
            device=x.device,
            dtype=residual_dtype,
        )
    else:
        residual_output = None
    mean = torch.empty((num_rows,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((num_rows,), dtype=torch.float, device=x.device)

    block_dim, block_rows, num_streams = _launch_config(
        num_rows,
        feature_dim,
        x.device.index,
    )
    activation_id = _activation_id(activation)
    layer_norm_gated_fwd_kernel[(num_streams,)](
        x=x,
        g=g,
        y=output,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_output,
        mean=mean,
        rstd=rstd,
        eps=eps,
        T=num_rows,
        NS=num_streams,
        D=feature_dim,
        BD=block_dim,
        BT=block_rows,
        ACTIVATION=activation_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_output is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return output, mean, rstd, residual_output if residual_output is not None else x


def _kda_rms_norm_sigmoid_gate_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Kimi's per-head RMSNorm and sigmoid output gate."""
    if x.shape[-1] != gate.shape[-1]:
        raise ValueError(f"KDA output and gate head dimensions differ: {x.shape[-1]} != {gate.shape[-1]}")
    if x.numel() != gate.numel():
        raise ValueError(f"KDA output and gate element counts differ: {x.numel()} != {gate.numel()}")

    output_shape = x.shape
    feature_dim = output_shape[-1]
    x_2d = x.contiguous().reshape(-1, feature_dim)
    gate_2d = gate.contiguous().reshape(-1, feature_dim)
    output, _, _, _ = layer_norm_gated_fwd_npu(
        x=x_2d,
        g=gate_2d,
        weight=weight.contiguous(),
        bias=None,
        activation="sigmoid",
        eps=eps,
        is_rms_norm=True,
    )
    return output.reshape(output_shape)


def _kda_rms_norm_sigmoid_gate_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del gate, weight, eps
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
) -> torch.Tensor:
    """Call the KDA-specific opaque op backed by the FLA Ascend kernel."""
    return torch.ops.vllm.kda_rms_norm_sigmoid_gate(x, gate, weight, eps)
