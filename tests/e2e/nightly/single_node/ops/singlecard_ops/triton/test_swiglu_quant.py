import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
INT8_MAX = 127
TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}
SCALE_TOLERANCES = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (2e-2, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def _swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.cpu().float()
    gate, up = x_fp32.chunk(2, dim=-1)
    return F.silu(gate) * up


def _dynamic_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=-1) / INT8_MAX
    quantized = torch.round(x / scale.unsqueeze(-1)).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


def _assert_quantized_result(x, actual, actual_scale):
    swiglu_ref = _swiglu_reference(x)
    expected, expected_scale = _dynamic_quant_reference(swiglu_ref)
    scale_rtol, scale_atol = SCALE_TOLERANCES[x.dtype]

    assert actual.shape == swiglu_ref.shape
    assert actual.dtype == torch.int8
    assert actual_scale.shape == (x.shape[0],)
    assert actual_scale.dtype == torch.float32
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=scale_rtol, atol=scale_atol)
    # The kernel casts through the input dtype before converting to int8.
    # Permit one quantization bin of error relative to the FP32 reference.
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=0, atol=1)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("shape", "dtype", "group_counts", "group_list_dtype"),
    [
        pytest.param((1, 4096), torch.float16, [1], torch.int32, id="single-token-fp16"),
        pytest.param(
            (65, 14336),
            torch.bfloat16,
            [0, 17, 1, 0, 31, 16],
            torch.int64,
            id="multi-core-bf16-empty-experts",
        ),
    ],
)
@pytest.mark.parametrize("group_list_type", [0, 1], ids=["cumsum", "count"])
@torch.inference_mode()
def test_swiglu_quant_correctness(shape, dtype, group_counts, group_list_dtype, group_list_type):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    group_list = torch.tensor(group_counts, dtype=group_list_dtype, device=DEVICE)
    if group_list_type == 0:
        group_list = group_list.cumsum(dim=0)

    actual, actual_scale = swiglu_quant(x, group_list, group_list_type=group_list_type)
    _assert_quantized_result(x, actual, actual_scale)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("shape", "group_counts"),
    [
        pytest.param((2, 256), [1, 1] + [0] * 6, id="moe-small"),
        pytest.param((384, 4096), [6] * 64, id="moe-large"),
        pytest.param((198, 4096), [25] * 6 + [24] * 2, id="moe-irregular-tokens"),
        pytest.param((2, 2048), [1, 1] + [0] * 62, id="moe-empty-experts"),
    ],
)
@torch.inference_mode()
def test_swiglu_quant_moe_routed_shapes(shape, group_counts):
    # Extracted from the W8A8 AllGather cases: GMM1 produces
    # [tokens * top_k, 2 * intermediate_size], with INT64 expert counts.
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    group_list = torch.tensor(group_counts, dtype=torch.int64, device=DEVICE)

    actual, actual_scale = swiglu_quant(x, group_list, group_list_type=1)
    _assert_quantized_result(x, actual, actual_scale)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_swiglu_quant_without_quantization(dtype):
    torch.manual_seed(42)
    x = torch.randn((9, 4096), dtype=dtype, device=DEVICE)
    group_list = torch.tensor([2, 0, 7], dtype=torch.int64, device=DEVICE)

    actual, _ = swiglu_quant(x, group_list, group_list_type=1, need_quant=False)
    expected = _swiglu_reference(x)
    rtol, atol = TOLERANCES[dtype]

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_swiglu_quant_rejects_invalid_group_list_type():
    x = torch.empty((1, 4096), dtype=torch.float16, device=DEVICE)
    group_list = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="group_list_type must be 0 or 1"):
        swiglu_quant(x, group_list, group_list_type=2)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_swiglu_quant_rejects_invalid_group_list_dtype():
    x = torch.empty((1, 4096), dtype=torch.float16, device=DEVICE)
    group_list = torch.tensor([1], dtype=torch.float32, device=DEVICE)

    with pytest.raises(ValueError, match="group_list dtype must be torch.int32 or torch.int64"):
        swiglu_quant(x, group_list, group_list_type=1)
