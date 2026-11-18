import pytest
import torch
import torch.nn.functional as F

import vllm_ascend.ops.triton.layernorm_gated as layernorm_gated
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

DEVICE = "npu"
pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
TOLERANCES = {
    torch.float16: (2**-9, 2**-9, 0.1),
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float32: (2**-10, 2**-16, 0.01),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("Requires Ascend A2/A3")
    init_device_properties_triton()


def assert_reference_accuracy(actual, expected):
    """Apply both contribution-guide bounds using the output tensor's dtype."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    rtol, atol, max_error = TOLERANCES[actual.dtype]
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    assert torch.isfinite(actual).all(), "Non-finite actual output"
    assert torch.isfinite(expected).all(), "Non-finite reference output"
    error = (actual - expected).abs()
    within_tolerance = error <= atol + rtol * expected.abs()
    assert within_tolerance.sum().item() >= 0.99 * error.numel()
    assert error.max().item() <= max_error


def layer_norm_gated_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    z: torch.Tensor | None,
    group_size: int | None,
    norm_before_gate: bool,
    is_rms_norm: bool,
    activation: str = "swish",
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

    gate = None
    if z_grouped is not None:
        gate = torch.sigmoid(z_grouped) if activation == "sigmoid" else F.silu(z_grouped)
    if gate is not None and not norm_before_gate:
        x_grouped = x_grouped * gate

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
    if gate is not None and norm_before_gate:
        output = output * gate

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
        pytest.param((65, 192), None, True, True, True, True, torch.float32, False, id="single-group-base16"),
        pytest.param((65, 512), None, False, True, False, True, torch.float16, False, id="base16-width-limit"),
        pytest.param((7, 180), 60, True, True, False, False, torch.float16, False, id="group-layer-norm-pre-gate"),
    ],
)
@pytest.mark.parametrize("activation", ["swish", "silu", "sigmoid"])
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
    activation,
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
        activation=activation,
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
        activation,
    )

    if out is not None:
        assert actual.data_ptr() == out.data_ptr()
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert_reference_accuracy(actual, expected)
    assert_reference_accuracy(actual_rstd, expected_rstd)
    if is_rms_norm:
        assert actual_mean is None
    else:
        assert actual_mean is not None
        assert expected_mean is not None
        assert_reference_accuracy(actual_mean, expected_mean)


class _KernelLaunchRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.grids = []
        self.launch_kwargs = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        launch = self.kernel[grid]

        def record_launch(*args, **kwargs):
            self.launch_kwargs.append(kwargs)
            return launch(*args, **kwargs)

        return record_launch


@pytest.mark.parametrize(
    ("rows_kind", "kernel_name", "dtype", "is_rms_norm", "has_bias", "has_gate", "norm_before_gate"),
    [
        pytest.param(
            "before",
            "_layer_norm_fwd_1pass_kernel_npu",
            torch.bfloat16,
            True,
            False,
            True,
            True,
            id="before-quarter-wave-rmsnorm-post-gate",
        ),
        pytest.param(
            "at",
            "_layer_norm_fwd_persistent_hoist_kernel_npu",
            torch.float16,
            False,
            True,
            True,
            False,
            id="quarter-wave-boundary-layernorm-pre-gate-fp16",
        ),
        pytest.param(
            "above",
            "_layer_norm_fwd_persistent_hoist_kernel_npu",
            torch.bfloat16,
            True,
            False,
            True,
            True,
            id="above-quarter-wave-rmsnorm-post-gate",
        ),
        pytest.param(
            "model",
            "_layer_norm_fwd_persistent_hoist_kernel_npu",
            torch.float32,
            True,
            False,
            True,
            True,
            id="gdn-flattened-m20449-n128-fp32",
        ),
        pytest.param(
            "large",
            "_layer_norm_fwd_persistent_hoist_kernel_npu",
            torch.bfloat16,
            True,
            False,
            True,
            True,
            id="hoist-rmsnorm-post-gate-large-m",
        ),
    ],
)
@pytest.mark.parametrize("activation", ["swish", "silu", "sigmoid"])
@torch.inference_mode()
def test_layer_norm_fwd_npu_hoist_routes(
    rows_kind,
    kernel_name,
    dtype,
    is_rms_norm,
    has_bias,
    has_gate,
    norm_before_gate,
    activation,
    monkeypatch,
):
    vector_cores = get_vectorcore_num()
    threshold = (vector_cores + 3) // 4 * 32 - 31
    rows = {
        "before": max(1, threshold - 1),
        "at": threshold,
        "above": threshold + 1,
        "large": 65536,
        "model": 20449,
    }[rows_kind]
    shape = (rows, 128)

    original_kernel = getattr(layernorm_gated, kernel_name)
    recorder = _KernelLaunchRecorder(original_kernel)
    monkeypatch.setattr(layernorm_gated, kernel_name, recorder)

    generator = torch.Generator(device="cpu").manual_seed(0x1F3A5C8)
    x = torch.randn(shape, generator=generator, dtype=dtype).to(DEVICE)
    weight = torch.randn((128,), generator=generator, dtype=dtype).to(DEVICE)
    bias = torch.randn((128,), generator=generator, dtype=dtype).to(DEVICE) if has_bias else None
    z = torch.randn(shape, generator=generator, dtype=dtype).to(DEVICE) if has_gate else None
    eps = 1e-6

    actual, actual_mean, actual_rstd = layer_norm_fwd_npu(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=128,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    expected, expected_mean, expected_rstd = layer_norm_gated_ref(
        x, weight, bias, eps, z, 128, norm_before_gate, is_rms_norm, activation
    )

    assert recorder.launch_kwargs[-1]["ACTIVATION"] == (1 if activation == "sigmoid" else 0)
    uses_hoist = 4 * ((rows + 31) // 32) >= vector_cores
    if not uses_hoist:
        assert recorder.grids == [((rows + 15) // 16, 1)]
        assert recorder.launch_kwargs[-1]["BLOCK_M"] == 16
        assert recorder.launch_kwargs[-1]["BLOCK_N"] == 128
    else:
        assert recorder.grids == [(min(vector_cores, (rows + 31) // 32),)]
        assert recorder.launch_kwargs[-1]["BLOCK_M"] == 32
        assert recorder.launch_kwargs[-1]["BLOCK_N"] == 128
    assert_reference_accuracy(actual, expected)
    assert_reference_accuracy(actual_rstd, expected_rstd)
    if is_rms_norm:
        assert actual_mean is None
        assert expected_mean is None
    else:
        assert_reference_accuracy(actual_mean, expected_mean)


@torch.inference_mode()
def test_layer_norm_fwd_npu_rejects_invalid_group_size():
    x = torch.empty((2, 10), dtype=torch.float16, device=DEVICE)
    weight = torch.ones(10, dtype=torch.float16, device=DEVICE)

    with pytest.raises(AssertionError):
        layer_norm_fwd_npu(x, weight, None, 1e-5, group_size=6)
