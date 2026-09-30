#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Kimi K3 W4A8 numerical coverage for DequantSituQuant."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

K3_ROUTED_INPUT_WIDTH = 6144
K3_SHARED_TP_CASES = (
    (1, 12288),
    (2, 6144),
    (4, 3072),
    (8, 1536),
    (16, 768),
)
K3_SITU_BETA = 4.0
K3_SITU_LINEAR_BETA = 25.0
K3_LOCAL_EXPERTS = 14
K3_TOP_K = 16


def _kimi_k3_reference(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_index is None:
        row_expert = torch.zeros(x.shape[0], dtype=torch.long)
        weight_scale = weight_scale.reshape(1, -1)
        bias = None if bias is None else bias.reshape(1, -1)
    else:
        row_expert = torch.repeat_interleave(torch.arange(group_index.numel()), group_index)
        assert row_expert.numel() == x.shape[0]

    dequant = x.float() * weight_scale[row_expert] * activation_scale.reshape(-1, 1)
    if bias is not None:
        dequant = dequant + bias[row_expert]
    gate, up = dequant.chunk(2, dim=-1)
    gate = K3_SITU_BETA * torch.tanh(gate / K3_SITU_BETA) * torch.sigmoid(gate)
    up = K3_SITU_LINEAR_BETA * torch.tanh(up / K3_SITU_LINEAR_BETA)
    situ = gate * up
    scale = situ.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = torch.round(situ / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return y, scale


def _run_dequant_situ_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor | None,
    activation_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.dequant_situ_quant(
        x=x.npu(),
        weight_scale=None if weight_scale is None else weight_scale.npu(),
        activation_scale=None if activation_scale is None else activation_scale.npu(),
        bias=None if bias is None else bias.npu(),
        quant_scale=None,
        quant_offset=None,
        group_index=None if group_index is None else group_index.npu(),
        beta=K3_SITU_BETA,
        linear_beta=K3_SITU_LINEAR_BETA,
        activate_left=True,
        quant_mode="dynamic",
    )


def _kimi_k3_predequantized_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = x.float().chunk(2, dim=-1)
    gate = K3_SITU_BETA * torch.tanh(gate / K3_SITU_BETA) * torch.sigmoid(gate)
    up = K3_SITU_LINEAR_BETA * torch.tanh(up / K3_SITU_LINEAR_BETA)
    situ = gate * up
    scale = situ.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = torch.round(situ / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return y, scale


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "group_counts",
    (
        # One decode token expands to top-k=16 routed rows.  Zero-count
        # experts are expected on an EP rank and must not shift later scales.
        (2, 0, 1, 1, 0, 2, 1, 1, 1, 0, 2, 1, 2, 2),
        # A 65-token prefill expands to 65 * top-k rows.  Keep all 14 local
        # experts non-empty and uneven so every expert boundary is exercised.
        (75, 75, 75, 75, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74),
    ),
    ids=("decode_t1_topk16", "prefill_t65_topk16"),
)
@torch.inference_mode()
def test_kimi_k3_routed_multi_expert_dequant_situ_quant(group_counts: tuple[int, ...]):
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        pytest.skip("requires the DequantSituQuant custom operator")

    assert len(group_counts) == K3_LOCAL_EXPERTS
    assert sum(group_counts) in (K3_TOP_K, 65 * K3_TOP_K)
    group_index = torch.tensor(group_counts, dtype=torch.int64)
    rows = int(group_index.sum())
    x = (torch.arange(rows * K3_ROUTED_INPUT_WIDTH, dtype=torch.int64) * 37) % 40001 - 20000
    x = x.to(torch.int32).reshape(rows, K3_ROUTED_INPUT_WIDTH)

    phase = torch.linspace(-torch.pi, torch.pi, K3_ROUTED_INPUT_WIDTH, dtype=torch.float32)
    expert_offset = torch.arange(K3_LOCAL_EXPERTS, dtype=torch.float32).reshape(-1, 1)
    weight_scale = 0.0008 + 0.00005 * expert_offset + 0.0002 * phase.cos()
    bias = -0.75 + 0.125 * expert_offset + 0.35 * phase.sin()
    activation_scale = torch.linspace(0.020, 0.080, rows, dtype=torch.float32)

    expected_y, expected_scale = _kimi_k3_reference(x, weight_scale, activation_scale, bias, group_index)
    actual_y, actual_scale = _run_dequant_situ_quant(x, weight_scale, activation_scale, bias, group_index)

    assert tuple(actual_y.shape) == (rows, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("rows", (1, 16, 1040), ids=("decode_m1", "decode_topk16", "prefill_t65_topk16"))
@torch.inference_mode()
def test_kimi_k3_predequantized_bf16_situ_quant(rows: int):
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        pytest.skip("requires the DequantSituQuant custom operator")

    values = torch.linspace(-32.0, 32.0, rows * K3_ROUTED_INPUT_WIDTH, dtype=torch.float32)
    x = (values + 0.125 * torch.sin(values)).to(torch.bfloat16).reshape(rows, K3_ROUTED_INPUT_WIDTH)
    expected_y, expected_scale = _kimi_k3_predequantized_reference(x)
    actual_y, actual_scale = _run_dequant_situ_quant(x, None, None, None, None)

    assert tuple(actual_y.shape) == (rows, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("tp_size", "input_width"),
    K3_SHARED_TP_CASES,
    ids=("tp1", "tp2", "tp4", "tp8", "tp16"),
)
@pytest.mark.parametrize("rows", (1, 65), ids=("decode_m1", "prefill_m65"))
@torch.inference_mode()
def test_kimi_k3_shared_expert_without_bias_or_group_index(tp_size: int, input_width: int, rows: int):
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        pytest.skip("requires the DequantSituQuant custom operator")

    assert input_width == 12288 // tp_size
    x = (torch.arange(rows * input_width, dtype=torch.int64) * 19) % 30001 - 15000
    x = x.to(torch.int32).reshape(rows, input_width)
    weight_scale = torch.linspace(0.0007, 0.0017, input_width, dtype=torch.float32)
    activation_scale = torch.linspace(0.025, 0.075, rows, dtype=torch.float32).reshape(rows, 1)

    expected_y, expected_scale = _kimi_k3_reference(x, weight_scale, activation_scale, None, None)
    actual_y, actual_scale = _run_dequant_situ_quant(x, weight_scale, activation_scale, None, None)

    assert tuple(actual_y.shape) == (rows, input_width // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kimi_k3_zero_routed_rows_return_empty_outputs():
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        pytest.skip("requires the DequantSituQuant custom operator")

    group_index = torch.zeros(K3_LOCAL_EXPERTS, dtype=torch.int64)
    x = torch.empty((0, K3_ROUTED_INPUT_WIDTH), dtype=torch.int32)
    weight_scale = torch.ones((K3_LOCAL_EXPERTS, K3_ROUTED_INPUT_WIDTH), dtype=torch.float32)
    activation_scale = torch.empty((0,), dtype=torch.float32)
    bias = torch.zeros_like(weight_scale)
    actual_y, actual_scale = _run_dequant_situ_quant(x, weight_scale, activation_scale, bias, group_index)

    assert tuple(actual_y.shape) == (0, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (0,)
