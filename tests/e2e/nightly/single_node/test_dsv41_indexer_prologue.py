# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the optional QW provider with the unfused framework stages."""

from importlib import import_module
from importlib.util import find_spec

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer
from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")


def _provider():
    # The optional wheel is not a framework installation dependency. Once the
    # provider is present, preserve all transitive import errors as failures.
    if find_spec("ops") is None or find_spec("ops.indexer_prologue_qw_vllm") is None:
        pytest.skip("Requires the matching V4.1 QW operator wheel")
    return import_module("ops.indexer_prologue_qw_vllm").indexer_prologue_qw


def _inputs(tokens):
    generator = torch.Generator().manual_seed(21830)
    query = (torch.randn(tokens, 1280, generator=generator) * 0.5).to(torch.float8_e4m3fn).npu()
    weight = (torch.randn(4096, 1280, generator=generator) * 0.5).to(torch.float8_e4m3fn).npu()
    query_scale = torch.randint(124, 129, (tokens, 20, 2), generator=generator, dtype=torch.uint8).npu()
    weight_scale = torch.randint(124, 129, (4096, 20, 2), generator=generator, dtype=torch.uint8).npu()
    hidden = torch.randn(tokens, 5120, generator=generator).bfloat16().npu()
    weights = torch.randn(32, 5120, generator=generator).bfloat16().npu()
    angles = (torch.rand(tokens, 32, generator=generator) * (2 * torch.pi)).repeat_interleave(2, -1)
    cosine, sine = angles.cos().npu(), angles.sin().npu()
    return query, weight, query_scale, weight_scale, hidden, weights, cosine, sine


@pytest.mark.parametrize("tokens", [6, 8, 17, 65, 129, 320, 4096])
@torch.inference_mode()
def test_qw_matches_unfused_bf16_stages(tokens):
    """Intermediate BF16 rounding is part of the framework numerical contract."""
    operator = _provider()
    assert enable_custom_op()
    query, weight, query_scale, weight_scale, hidden, weights, cosine, sine = _inputs(tokens)
    scale = 128**-0.5 * 32**-0.5
    projected = torch_npu.npu_quant_matmul(
        query,
        torch_npu.npu_format_cast(weight.T.contiguous(), 29),
        weight_scale.transpose(0, 1).contiguous(),
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=query_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    ).view(tokens, 32, 128)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        projected.unsqueeze(1),
        cosine.view(tokens, 1, 1, 64),
        sine.view(tokens, 1, 1, 64),
        rotary_mode="interleave",
        partial_slice=[64, 128],
    )
    expected_q, expected_scale = quantize_mxfp4_indexer(projected)
    expected_weights = torch.nn.functional.linear(hidden, weights).float() * scale
    actual_q, actual_scale, actual_weights = operator(
        hidden,
        query,
        torch_npu.npu_format_cast(weight, 29),
        torch_npu.npu_format_cast(weights, 29),
        query_scale,
        weight_scale,
        sine,
        cosine,
        softmax_scale=scale,
    )
    torch.testing.assert_close(actual_q.cpu(), expected_q.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale.cpu().reshape(tokens, 32, 4), expected_scale.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual_weights.cpu(), expected_weights.cpu(), rtol=2e-2, atol=2e-2)
