# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.fla.layernorm_guard import _layer_norm_fwd


def _layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    z: torch.Tensor | None,
    group_size: int,
    norm_before_gate: bool,
    is_rms_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    rows, hidden_size = x.shape
    num_groups = hidden_size // group_size
    x_grouped = x.float().reshape(rows, num_groups, group_size)
    z_grouped = None if z is None else z.float().reshape(rows, num_groups, group_size)

    if z_grouped is not None and not norm_before_gate:
        x_grouped = x_grouped * torch.nn.functional.silu(z_grouped)

    if is_rms_norm:
        mean = None
        variance = x_grouped.square().mean(dim=-1, keepdim=True)
        normalized = x_grouped * torch.rsqrt(variance + eps)
    else:
        mean = x_grouped.mean(dim=-1, keepdim=True)
        variance = (x_grouped - mean).square().mean(dim=-1, keepdim=True)
        normalized = (x_grouped - mean) * torch.rsqrt(variance + eps)

    output = normalized * weight.float().reshape(num_groups, group_size)
    if bias is not None:
        output = output + bias.float().reshape(num_groups, group_size)
    if z_grouped is not None and norm_before_gate:
        output = output * torch.nn.functional.silu(z_grouped)

    # The kernel stores statistics in group-major order: [group, row].
    mean_flat = None if mean is None else mean.squeeze(-1).transpose(0, 1).reshape(-1)
    rstd_flat = torch.rsqrt(variance + eps).squeeze(-1).transpose(0, 1).reshape(-1)
    return output.reshape(rows, hidden_size), mean_flat, rstd_flat


@pytest.mark.parametrize(
    ("group_size", "has_bias", "has_gate", "norm_before_gate", "is_rms_norm"),
    [
        (256, False, True, False, True),
        (64, True, True, True, True),
        (64, True, False, True, False),
        (256, True, True, False, False),
    ],
)
def test_layer_norm_fwd_kernel_accuracy(
    group_size: int,
    has_bias: bool,
    has_gate: bool,
    norm_before_gate: bool,
    is_rms_norm: bool,
):
    torch.manual_seed(2026)
    rows, hidden_size = 7, 256
    eps = 1e-6
    x_cpu = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    weight_cpu = torch.randn(hidden_size, dtype=torch.bfloat16)
    bias_cpu = torch.randn(hidden_size, dtype=torch.bfloat16) if has_bias else None
    z_cpu = torch.randn(rows, hidden_size, dtype=torch.bfloat16) if has_gate else None

    expected, expected_mean, expected_rstd = _layer_norm_reference(
        x_cpu,
        weight_cpu,
        bias_cpu,
        eps,
        z_cpu,
        group_size,
        norm_before_gate,
        is_rms_norm,
    )
    actual, actual_mean, actual_rstd = _layer_norm_fwd(
        x_cpu.npu(),
        weight_cpu.npu(),
        None if bias_cpu is None else bias_cpu.npu(),
        eps,
        z=None if z_cpu is None else z_cpu.npu(),
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    torch.testing.assert_close(actual.float().cpu(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_rstd.cpu(), expected_rstd, rtol=3e-3, atol=3e-3)
    if is_rms_norm:
        assert actual_mean is None
    else:
        assert actual_mean is not None
        assert expected_mean is not None
        torch.testing.assert_close(actual_mean.cpu(), expected_mean, rtol=3e-3, atol=3e-3)
