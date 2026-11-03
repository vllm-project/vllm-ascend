# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

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


def _bf16_input(rows: int, width: int) -> torch.Tensor:
    if rows == 0:
        return torch.empty((0, width), dtype=torch.bfloat16)
    values = torch.arange(rows * width, dtype=torch.int64)
    values = ((values * 37) % 4097).float() / 64.0 - 32.0
    return values.to(torch.bfloat16).reshape(rows, width)


def _routed_input(rows: int) -> torch.Tensor:
    return _bf16_input(rows, SITU_ROUTED_INPUT_WIDTH)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="SiTU MX requires Ascend A5",
)
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("tp_size", "input_width"),
    SITU_SHARED_TP_CASES,
    ids=("tp1", "tp2", "tp4", "tp8", "tp16"),
)
@pytest.mark.parametrize(("phase", "rows"), (("decode", 1), ("prefill", 65)), ids=("decode", "prefill"))
@torch.inference_mode()
def test_shared_situ_mx_quant_shapes_single_op(
    tp_size: int,
    input_width: int,
    phase: str,
    rows: int,
):
    """A5 shared-expert MXFP output and scale layout."""
    assert phase in {"decode", "prefill"}
    assert input_width == 12288 // tp_size
    x = _bf16_input(rows, input_width)
    y, mxscale = torch.ops._C_ascend.situ_mx_quant(
        x.npu(),
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        activate_left=True,
        dst_type=36,
    )

    output_width = input_width // 2
    assert tuple(y.shape) == (rows, output_width)
    assert y.dtype == torch.float8_e4m3fn
    assert tuple(mxscale.shape) == (rows, math.ceil(output_width / 64), 2)
    assert mxscale.dtype == torch.float8_e8m0fnu


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="SiTU MX requires Ascend A5",
)
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(("phase", "rows"), (("decode_max_local", 14), ("prefill_max_local", 910)))
@torch.inference_mode()
def test_routed_situ_mx_quant_shapes_single_op(phase: str, rows: int):
    """A5 routed-expert shape is TP-invariant."""
    assert phase in {"decode_max_local", "prefill_max_local"}
    x = _routed_input(rows)
    y, mxscale = torch.ops._C_ascend.situ_mx_quant(
        x.npu(),
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        activate_left=True,
        dst_type=36,
    )

    assert tuple(y.shape) == (rows, 3072)
    assert y.dtype == torch.float8_e4m3fn
    assert tuple(mxscale.shape) == (rows, 48, 2)
    assert mxscale.dtype == torch.float8_e8m0fnu
