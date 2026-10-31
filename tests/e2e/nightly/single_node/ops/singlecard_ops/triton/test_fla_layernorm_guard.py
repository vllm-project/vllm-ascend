# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.fla.layernorm_guard import MAX_CORES, _layer_norm_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    is_rms_norm: bool,
):
    input_dtype = x.dtype
    x = x.float().cpu()
    weight = weight.float().cpu()
    bias = bias.float().cpu() if bias is not None else None
    z = z.float().cpu() if z is not None else None

    num_rows, hidden_size = x.shape
    group_size = group_size or hidden_size
    x = x.view(num_rows, -1, group_size)
    z = z.view_as(x) if z is not None else None

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    centered = x if is_rms_norm else x - x.mean(dim=-1, keepdim=True)
    output = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    output = output * weight.view(1, -1, group_size)
    if bias is not None:
        output = output + bias.view(1, -1, group_size)
    if z is not None and norm_before_gate:
        output = output * F.silu(z)

    return output.reshape(num_rows, hidden_size).to(input_dtype)


@pytest.mark.parametrize(
    (
        "shape",
        "group_size",
        "has_bias",
        "has_gate",
        "norm_before_gate",
        "is_rms_norm",
        "dtype",
    ),
    [
        pytest.param((1, 128), None, True, False, True, False, torch.float32, id="layer-norm-bias"),
        pytest.param((67, 192), 96, False, True, True, True, torch.bfloat16, id="group-rms-gate"),
        pytest.param(
            (MAX_CORES + 3, 16),
            None,
            True,
            False,
            True,
            False,
            torch.float16,
            id="rows-beyond-grid-limit",
        ),
    ],
)
@torch.inference_mode()
def test_layer_norm_fwd(
    shape,
    group_size,
    has_bias,
    has_gate,
    norm_before_gate,
    is_rms_norm,
    dtype,
):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    weight = torch.randn(shape[-1], dtype=dtype, device=DEVICE)
    bias = torch.randn(shape[-1], dtype=dtype, device=DEVICE) if has_bias else None
    z = torch.randn(shape, dtype=dtype, device=DEVICE) if has_gate else None
    eps = 1e-5

    actual, _, _ = _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    expected = layer_norm_ref(
        x,
        weight,
        bias,
        z,
        eps,
        group_size,
        norm_before_gate,
        is_rms_norm,
    )

    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=rtol, atol=atol)
