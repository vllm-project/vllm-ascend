# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.step5.fused_add_rms_norm import (
    can_use_fused_add_rms_norm_cast,
    fused_add_rms_norm_cast,
)

pytestmark = pytest.mark.skipif(not (HAS_TRITON and is_950()), reason="Requires Ascend A5 with Triton")

EPS = 1e-6
STEP5_DEFAULT_HIDDEN_SIZE = 5120
BF16_TOLERANCE = 2**-6
MIN_WITHIN_TOLERANCE = 0.99
MAX_ABSOLUTE_ERROR = 1.0


@pytest.mark.parametrize(
    "tokens, hidden_size, input_dtype, zero_centered, has_residual",
    [
        (0, STEP5_DEFAULT_HIDDEN_SIZE, torch.bfloat16, True, True),
        (1, STEP5_DEFAULT_HIDDEN_SIZE, torch.bfloat16, True, True),
        (333, 6144, torch.float16, False, True),
        (4096, STEP5_DEFAULT_HIDDEN_SIZE, torch.float32, True, True),
        (0, STEP5_DEFAULT_HIDDEN_SIZE, torch.float32, True, False),
        (1, STEP5_DEFAULT_HIDDEN_SIZE, torch.float32, True, False),
        (333, STEP5_DEFAULT_HIDDEN_SIZE, torch.float32, False, False),
    ],
)
def test_fused_add_rms_norm_matches_decoder_path(tokens, hidden_size, input_dtype, zero_centered, has_residual):
    torch.manual_seed(0)
    hidden_states = torch.randn(tokens, hidden_size, device="npu", dtype=input_dtype)
    residual = torch.randn_like(hidden_states, dtype=torch.float32) if has_residual else None
    weight = torch.randn(hidden_size, device="npu", dtype=torch.float32) * 0.1
    expected_residual = hidden_states if residual is None else hidden_states.float() + residual
    assert can_use_fused_add_rms_norm_cast(hidden_states, residual, weight, fp32_residual=True)
    output, residual_out = fused_add_rms_norm_cast(hidden_states, residual, weight, EPS, zero_centered)

    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    assert output.dtype == torch.bfloat16
    assert output.shape == hidden_states.shape
    if residual is None:
        assert residual_out is hidden_states
    elif tokens:
        assert residual_out.data_ptr() != residual.data_ptr()
    if tokens == 0:
        return

    assert output.data_ptr() != hidden_states.data_ptr()

    if zero_centered:
        expected, _ = torch_npu.npu_rms_norm(expected_residual, 1.0 + weight, EPS)
    else:
        inv_rms = torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + EPS)
        expected = expected_residual * inv_rms * weight
    expected = expected.bfloat16().float()
    diff = (output.float() - expected).abs()
    assert (diff <= BF16_TOLERANCE * (1 + expected.abs())).float().mean().item() >= MIN_WITHIN_TOLERANCE
    assert diff.max().item() <= MAX_ABSOLUTE_ERROR


def test_fused_add_rms_norm_rejects_inputs_outside_fp32_path():
    hidden_states = torch.ones((2, 8), device="npu", dtype=torch.bfloat16)
    residual = torch.ones((2, 8), device="npu", dtype=torch.float32)
    weight = torch.ones(8, device="npu", dtype=torch.float32)

    assert not can_use_fused_add_rms_norm_cast(hidden_states, None, weight, fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual, weight, fp32_residual=False)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual, weight.bfloat16(), fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual[:1], weight, fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual.T.contiguous().T, weight, fp32_residual=True)
