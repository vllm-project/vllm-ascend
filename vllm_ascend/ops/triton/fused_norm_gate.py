# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""Inference-only fused LayerNorm/RMSNorm and gate for Ascend Triton.

Adapted from FLA's ``triton_ascend/fused_norm_gate.py``. The grid-stride
row tiling keeps the norm parameters resident and caps the launch at the
available vector-core count. Kimi K3 uses the RMSNorm + sigmoid-gate path.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.layernorm_gated import (
    layer_norm_fwd_npu as legacy_layer_norm_fwd_npu,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_ASCEND_910_UB_BYTES = 192 * 1024
_UB_SAFETY_MARGIN = 0.85
_BD_MEMORY_MULTIPLIER = 6.0
# The K3 path can gate before normalization, keeping both the gate and
# normalized-value tiles live. Ascend910 compilation measured just over 6x
# [block_rows, block_feature] at D=128, so use the same conservative budget as
# the single-row feature check instead of FLA's post-norm-only 3x budget.
_FWD_MEMORY_MULTIPLIER = 6.0
_LARGE_BD = 2048
_LARGE_BD_MEMORY_MULTIPLIER = 4.0
_MAX_BT = 128
_ASCEND_MAX_GRID_DIM = 65535

_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SWISH
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def _largest_power_of_two_at_most(value: int) -> int:
    return 1 << max(0, value.bit_length() - 1)


def _get_layer_norm_gated_tiles(feature_size: int) -> tuple[int, int]:
    block_feature = triton.next_power_of_2(feature_size)
    safe_bytes = int(_ASCEND_910_UB_BYTES * _UB_SAFETY_MARGIN)
    max_feature = int(safe_bytes // (_BD_MEMORY_MULTIPLIER * 4))
    if block_feature > max_feature:
        raise RuntimeError(
            f"LayerNormGated feature dim {feature_size} exceeds the UB-safe "
            f"block size {_largest_power_of_two_at_most(max_feature)}."
        )
    row_multiplier = _LARGE_BD_MEMORY_MULTIPLIER if block_feature >= _LARGE_BD else _FWD_MEMORY_MULTIPLIER
    max_rows = int(safe_bytes // (row_multiplier * block_feature * 4))
    block_rows = min(_MAX_BT, _largest_power_of_two_at_most(max(1, max_rows)))
    return block_feature, block_rows


@triton.jit(do_not_specialize=["num_rows"])
def _fused_norm_gate_kernel(
    x,
    gate,
    output,
    weight,
    bias,
    mean,
    rstd,
    eps,
    num_rows,
    num_programs,
    feature_size: tl.constexpr,
    block_feature: tl.constexpr,
    block_rows: tl.constexpr,
    activation: tl.constexpr,
    norm_before_gate: tl.constexpr,
    is_rms_norm: tl.constexpr,
    has_bias: tl.constexpr,
):
    program_id = tl.program_id(0)
    columns = tl.arange(0, block_feature)
    column_mask = columns < feature_size
    norm_weight = tl.load(weight + columns, mask=column_mask).to(tl.float32)
    if has_bias:
        norm_bias = tl.load(bias + columns, mask=column_mask, other=0.0).to(tl.float32)

    num_tiles = tl.cdiv(num_rows, block_rows)
    for tile_id in range(program_id, num_tiles, num_programs):
        rows = tile_id * block_rows + tl.arange(0, block_rows)
        row_mask = rows < num_rows
        mask = row_mask[:, None] & column_mask[None, :]
        offsets = rows[:, None] * feature_size + columns[None, :]

        values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid_gate = tl.sigmoid(gate_values)
        if activation == 0:
            gate_multiplier = gate_values * sigmoid_gate
        else:
            gate_multiplier = sigmoid_gate
        if not norm_before_gate:
            values *= gate_multiplier
        if not is_rms_norm:
            row_mean = tl.sum(values, axis=1) / feature_size
            centered = tl.where(mask, values - row_mean[:, None], 0.0)
            variance = tl.sum(centered * centered, axis=1) / feature_size
            tl.store(mean + rows, row_mean, mask=row_mask)
        else:
            variance = tl.sum(tl.where(mask, values * values, 0.0), axis=1) / feature_size
        reciprocal_std = 1.0 / tl.sqrt(variance + eps)
        tl.store(rstd + rows, reciprocal_std, mask=row_mask)

        if is_rms_norm:
            normalized = values * reciprocal_std[:, None]
        else:
            normalized = (values - row_mean[:, None]) * reciprocal_std[:, None]
        normalized *= norm_weight[None, :]
        if has_bias:
            normalized += norm_bias[None, :]

        if norm_before_gate:
            normalized *= gate_multiplier
        tl.store(output + offsets, normalized.to(output.dtype.element_ty), mask=mask)


def fused_norm_gate_fwd_npu(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    activation: str = "swish",
    eps: float = 1e-5,
    is_rms_norm: bool = False,
    norm_before_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    num_rows, feature_size = x.shape
    assert gate.shape == x.shape
    assert x.stride(-1) == gate.stride(-1) == 1
    assert weight.shape == (feature_size,)
    if bias is not None:
        assert bias.shape == (feature_size,)

    output = torch.empty_like(x)
    mean = None if is_rms_norm else torch.empty((num_rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((num_rows,), dtype=torch.float32, device=x.device)
    block_feature, block_rows = _get_layer_norm_gated_tiles(feature_size)
    num_tiles = triton.cdiv(num_rows, block_rows)
    num_programs = max(
        1,
        min(get_vectorcore_num(), num_tiles, _ASCEND_MAX_GRID_DIM),
    )
    _fused_norm_gate_kernel[(num_programs,)](
        x,
        gate,
        output,
        weight,
        bias,
        mean,
        rstd,
        eps,
        num_rows,
        num_programs,
        feature_size=feature_size,
        block_feature=block_feature,
        block_rows=block_rows,
        activation=_activation_id(activation),
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        has_bias=bias is not None,
    )
    return output, mean, rstd


def layer_norm_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    z: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
):
    if z is None or out is not None:
        return legacy_layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            out=out,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
    if group_size not in (None, x.shape[-1]):
        return legacy_layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
    return fused_norm_gate_fwd_npu(
        x,
        z,
        weight,
        bias,
        activation=activation,
        eps=eps,
        is_rms_norm=is_rms_norm,
        norm_before_gate=norm_before_gate,
    )
