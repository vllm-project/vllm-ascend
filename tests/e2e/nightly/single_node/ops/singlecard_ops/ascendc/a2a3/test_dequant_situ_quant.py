# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
SITU_ROUTED_INPUT_WIDTH = 6144
SITU_LOCAL_EXPERTS = 14
SITU_SHARED_TP_CASES = (
    (1, 12288),
    (2, 6144),
    (4, 3072),
    (8, 1536),
    (16, 768),
)


def _situ(values: torch.Tensor) -> torch.Tensor:
    gate, up = values.float().chunk(2, dim=-1)
    gate = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    up = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return gate * up


def _dynamic_int8_quant(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] == 0:
        return (
            torch.empty((0, values.shape[-1]), dtype=torch.int8),
            torch.empty((0,), dtype=torch.float32),
        )
    scale = values.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    output = torch.round(values / scale[:, None]).clamp(-128, 127).to(torch.int8)
    return output, scale


def _shared_inputs(rows: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulator = (torch.arange(rows * width, dtype=torch.int64) * 19) % 30001 - 15000
    accumulator = accumulator.to(torch.int32).reshape(rows, width)
    weight_scale = torch.linspace(0.0007, 0.0017, width, dtype=torch.float32)
    activation_scale = torch.linspace(0.025, 0.075, rows, dtype=torch.float32).reshape(rows, 1)
    return accumulator, weight_scale, activation_scale


def _bf16_input(rows: int, width: int) -> torch.Tensor:
    if rows == 0:
        return torch.empty((0, width), dtype=torch.bfloat16)
    values = torch.arange(rows * width, dtype=torch.int64)
    values = ((values * 37) % 4097).float() / 64.0 - 32.0
    return values.to(torch.bfloat16).reshape(rows, width)


def _routed_input(rows: int) -> torch.Tensor:
    return _bf16_input(rows, SITU_ROUTED_INPUT_WIDTH)


def _run_dequant_situ_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor | None,
    activation_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hasattr(torch.ops._C_ascend, "dequant_situ_quant")
    return torch.ops._C_ascend.dequant_situ_quant(
        x=x.npu(),
        weight_scale=None if weight_scale is None else weight_scale.npu(),
        activation_scale=None if activation_scale is None else activation_scale.npu(),
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        activate_left=True,
        quant_mode="dynamic",
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("tp_size", "input_width"),
    SITU_SHARED_TP_CASES,
    ids=("tp1", "tp2", "tp4", "tp8", "tp16"),
)
@pytest.mark.parametrize(("phase", "rows"), (("decode", 1), ("prefill", 65)), ids=("decode", "prefill"))
@torch.inference_mode()
def test_shared_dequant_situ_quant_single_op(
    tp_size: int,
    input_width: int,
    phase: str,
    rows: int,
):
    """Shared experts: INT32 accumulator plus FP32 dequant scales."""
    assert phase in {"decode", "prefill"}
    assert input_width == 12288 // tp_size
    x, weight_scale, activation_scale = _shared_inputs(rows, input_width)
    dequantized = x.float() * weight_scale[None, :] * activation_scale
    expected_y, expected_scale = _dynamic_int8_quant(_situ(dequantized))

    actual_y, actual_scale = _run_dequant_situ_quant(x, weight_scale, activation_scale)

    assert tuple(actual_y.shape) == (rows, input_width // 2)
    assert actual_y.dtype == torch.int8
    assert tuple(actual_scale.shape) == (rows,)
    assert actual_scale.dtype == torch.float32
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("phase", "rows"),
    (
        ("decode_empty", 0),
        ("decode_max_local", SITU_LOCAL_EXPERTS),
        ("prefill_max_local", 65 * SITU_LOCAL_EXPERTS),
    ),
    ids=("decode_empty_m0", "decode_max_m14", "prefill_max_m910"),
)
@torch.inference_mode()
def test_routed_bf16_dequant_situ_quant_single_op(phase: str, rows: int):
    """Routed experts: the same operator, BF16 input, no dequant stage."""
    assert phase in {"decode_empty", "decode_max_local", "prefill_max_local"}
    x = _routed_input(rows)
    expected_y, expected_scale = _dynamic_int8_quant(_situ(x))

    actual_y, actual_scale = _run_dequant_situ_quant(x, None, None)

    assert tuple(actual_y.shape) == (rows, SITU_ROUTED_INPUT_WIDTH // 2)
    assert actual_y.dtype == torch.int8
    assert tuple(actual_scale.shape) == (rows,)
    assert actual_scale.dtype == torch.float32
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)
