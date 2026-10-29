# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""DequantSiTU shared, routed, group-index, and W4A8 regressions.

The group-index cases cover the BF16 path.

The BF16 (already-dequantized) routed path used to reject ``group_index``
and therefore scanned the whole worst-case-expanded MoE buffer row by row,
including the dead tail rows that no expert owns. With ``group_index``
accepted, the kernel iterates per-expert row groups and skips the dead tail.

These tests pin the equivalence contract:
  * passing counts whose sum equals the row length is bit-identical to
    passing ``None`` (full scan);
  * counts whose sum is smaller only touch the leading active rows;
  * counts whose sum exceeds the row length (padded tail-chunk buffers)
    clamp safely inside the kernel group loop;
  * outputs match a float32 reference within 1 int8 LSB.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,E402  (registers torch.ops._C_ascend)

enable_custom_op()

OP = torch.ops._C_ascend.dequant_situ_quant

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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSiTU regression targets Ascend A2/A3",
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSiTU regression targets Ascend A2/A3",
)
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


def _run(x, group_index=None):
    y, scale = OP(
        x=x,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_index,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        quant_mode="dynamic",
    )
    torch.npu.synchronize()
    return y, scale


def _counts(num_groups, total, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(num_groups, generator=g) + 0.2
    c = (w / w.sum() * total).to(torch.int64)
    c[0] += total - c.sum()
    assert int(c.sum()) == total
    return c.to(device)


def _golden(x_bf16):
    x = x_bf16.to(torch.float32)
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    g = torch.tanh(gate) * torch.sigmoid(gate)
    y = g * up
    absmax = y.abs().amax(dim=-1, keepdim=True)
    scale = absmax / 127.0
    inv = torch.where(scale > 0, 1.0 / scale, torch.ones_like(scale))
    y_int8 = (y * inv).round().clamp(-127, 127).to(torch.int8)
    # 1-D, matching the op's per-row scale output; a [rows, 1] return would
    # silently broadcast against the op's [rows] in the assertions below.
    return y_int8, scale.view(-1)


def _bf16_x(rows, cols, seed):
    torch.npu.manual_seed_all(seed)
    return (torch.randn(rows, cols, device="npu:0") * 2.0).to(torch.bfloat16)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
@pytest.mark.parametrize("rows,cols,groups", [(896, 6144, 14), (528, 6144, 14), (64, 128, 8)])
def test_group_counts_equal_rows_matches_full_scan(rows, cols, groups):
    """sum(counts) == rows: group path must be bit-identical to None."""
    x = _bf16_x(rows, cols, seed=1)
    counts = _counts(groups, rows, x.device)
    y_none, s_none = _run(x, None)
    y_grp, s_grp = _run(x, counts)
    assert torch.equal(y_none.int(), y_grp.int())
    assert torch.equal(s_none, s_grp)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
@pytest.mark.parametrize("rows,active,groups", [(896, 16, 14), (33664, 526, 14), (7296, 456, 14)])
def test_group_counts_smaller_only_touches_active_rows(rows, active, groups):
    """sum(counts) < rows: dead tail rows are skipped; active prefix matches
    the full-scan outputs (the dead rows are never consumed downstream)."""
    x = _bf16_x(rows, 6144, seed=2)
    counts = _counts(groups, active, x.device)
    y_none, s_none = _run(x, None)
    y_grp, s_grp = _run(x, counts)
    assert torch.equal(y_none[:active].int(), y_grp[:active].int())
    assert torch.equal(s_none[:active], s_grp[:active])


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
def test_group_counts_exceeding_rows_clamps():
    """sum(counts) > rows (padded tail-chunk buffer): must not crash and must
    still produce correct outputs for the rows that exist."""
    rows = 530
    x = _bf16_x(rows, 6144, seed=3)
    counts = torch.full((14,), 2400, dtype=torch.int64, device=x.device)
    counts[0] += rows - 2400 * 14  # sum == 33664 > rows == 530
    y_grp, s_grp = _run(x, counts)
    y_ref, s_ref = _golden(x)
    assert s_grp.shape == s_ref.shape
    assert (y_grp.int() - y_ref.int()).abs().max() <= 1
    assert ((s_grp - s_ref).abs() / s_ref.clamp_min(1e-6)).max() < 1e-5


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
def test_bf16_output_matches_float32_reference():
    x = _bf16_x(528, 6144, seed=4)
    y, scale = _run(x, None)
    y_ref, scale_ref = _golden(x)
    assert scale.shape == scale_ref.shape
    assert (y.int() - y_ref.int()).abs().max() <= 1
    assert ((scale - scale_ref).abs() / scale_ref.clamp_min(1e-6)).max() < 1e-5


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

# Kimi K3 W4A8 numerical coverage for DequantSituQuant.

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


def _run_dequant_situ_quant_w4a8(
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
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
    actual_y, actual_scale = _run_dequant_situ_quant_w4a8(x, weight_scale, activation_scale, bias, group_index)

    assert tuple(actual_y.shape) == (rows, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("rows", (1, 16, 1040), ids=("decode_m1", "decode_topk16", "prefill_t65_topk16"))
@torch.inference_mode()
def test_kimi_k3_predequantized_bf16_situ_quant(rows: int):
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        pytest.skip("requires the DequantSituQuant custom operator")

    values = torch.linspace(-32.0, 32.0, rows * K3_ROUTED_INPUT_WIDTH, dtype=torch.float32)
    x = (values + 0.125 * torch.sin(values)).to(torch.bfloat16).reshape(rows, K3_ROUTED_INPUT_WIDTH)
    expected_y, expected_scale = _kimi_k3_predequantized_reference(x)
    actual_y, actual_scale = _run_dequant_situ_quant_w4a8(x, None, None, None, None)

    assert tuple(actual_y.shape) == (rows, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
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
    actual_y, actual_scale = _run_dequant_situ_quant_w4a8(x, weight_scale, activation_scale, None, None)

    assert tuple(actual_y.shape) == (rows, input_width // 2)
    assert tuple(actual_scale.shape) == (rows,)
    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="DequantSituQuant is configured for Ascend A2/A3",
)
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
    actual_y, actual_scale = _run_dequant_situ_quant_w4a8(x, weight_scale, activation_scale, bias, group_index)

    assert tuple(actual_y.shape) == (0, K3_ROUTED_INPUT_WIDTH // 2)
    assert tuple(actual_scale.shape) == (0,)
