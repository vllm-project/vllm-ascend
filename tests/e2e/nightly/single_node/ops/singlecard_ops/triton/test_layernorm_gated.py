import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
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


def layer_norm_gated_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    z: torch.Tensor | None,
    group_size: int | None,
    norm_before_gate: bool,
    is_rms_norm: bool,
):
    input_dtype = x.dtype
    x = x.cpu().float()
    weight = weight.cpu().float()
    bias = bias.cpu().float() if bias is not None else None
    z = z.cpu().float() if z is not None else None

    num_rows, hidden_size = x.shape
    group_size = group_size or hidden_size
    num_groups = hidden_size // group_size
    x_grouped = x.view(num_rows, num_groups, group_size)
    z_grouped = z.view(num_rows, num_groups, group_size) if z is not None else None

    if z_grouped is not None and not norm_before_gate:
        x_grouped = x_grouped * F.silu(z_grouped)

    if is_rms_norm:
        mean = None
        x_centered = x_grouped
    else:
        mean = x_grouped.mean(dim=-1)
        x_centered = x_grouped - mean.unsqueeze(-1)

    rstd = torch.rsqrt(x_centered.square().mean(dim=-1) + eps)
    output = x_centered * rstd.unsqueeze(-1)
    output = output * weight.view(num_groups, group_size).unsqueeze(0)
    if bias is not None:
        output = output + bias.view(num_groups, group_size).unsqueeze(0)
    if z_grouped is not None and norm_before_gate:
        output = output * F.silu(z_grouped)

    # The kernel stores statistics group-major: [group, row].
    mean = mean.transpose(0, 1).contiguous().flatten() if mean is not None else None
    rstd = rstd.transpose(0, 1).contiguous().flatten()
    return output.reshape_as(x).to(input_dtype), mean, rstd


@pytest.mark.parametrize(
    (
        "shape",
        "group_size",
        "has_bias",
        "has_gate",
        "norm_before_gate",
        "is_rms_norm",
        "dtype",
        "use_out",
    ),
    [
        pytest.param((1, 128), None, True, False, True, False, torch.float32, False, id="layer-norm-bias"),
        pytest.param((67, 192), 96, False, True, True, True, torch.float16, True, id="group-rms-post-gate"),
        pytest.param((5, 256), 128, True, True, False, True, torch.bfloat16, False, id="group-rms-pre-gate"),
        pytest.param((7, 180), 60, True, True, False, False, torch.float16, False, id="group-layer-norm-pre-gate"),
    ],
)
@torch.inference_mode()
def test_layer_norm_fwd_npu_correctness(
    shape,
    group_size,
    has_bias,
    has_gate,
    norm_before_gate,
    is_rms_norm,
    dtype,
    use_out,
):
    torch.manual_seed(42)
    x = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE)
    weight = (torch.randn(shape[-1], dtype=torch.float32) * 0.2 + 1.0).to(dtype=dtype, device=DEVICE)
    bias = (torch.randn(shape[-1], dtype=torch.float32) * 0.1).to(dtype=dtype, device=DEVICE) if has_bias else None
    z = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE) if has_gate else None
    out = torch.empty_like(x) if use_out else None
    eps = 1e-5

    actual, actual_mean, actual_rstd = layer_norm_fwd_npu(
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
    expected, expected_mean, expected_rstd = layer_norm_gated_ref(
        x,
        weight,
        bias,
        eps,
        z,
        group_size,
        norm_before_gate,
        is_rms_norm,
    )

    if out is not None:
        assert actual.data_ptr() == out.data_ptr()
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_rstd.cpu(), expected_rstd, rtol=rtol, atol=atol)
    if is_rms_norm:
        assert actual_mean is None
    else:
        assert actual_mean is not None
        assert expected_mean is not None
        torch.testing.assert_close(actual_mean.cpu(), expected_mean, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_layer_norm_fwd_npu_rejects_invalid_group_size():
    x = torch.empty((2, 10), dtype=torch.float16, device=DEVICE)
    weight = torch.ones(10, dtype=torch.float16, device=DEVICE)

    with pytest.raises(AssertionError):
        layer_norm_fwd_npu(x, weight, None, 1e-5, group_size=6)
