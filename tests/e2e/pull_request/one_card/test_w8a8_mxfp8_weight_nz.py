import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


def _is_a5_runtime() -> bool:
    try:
        if not torch.npu.is_available():
            return False
        return device_type_from_runtime_soc(torch.npu.get_soc_version()) == AscendDeviceType.A5
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _is_a5_runtime(), reason="W8A8 MXFP8 requires Ascend 950")

_PARAM_BUFFER_NAMES = {
    "w13_weight": "_mxfp8_w13_weight_buf",
    "w2_weight": "_mxfp8_w2_weight_buf",
    "w13_weight_scale": "_mxfp8_w13_scale_buf",
    "w2_weight_scale": "_mxfp8_w2_scale_buf",
}


def _fp8_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 255, shape, dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)


def _make_layer() -> torch.nn.Module:
    num_experts = 2
    hidden_size = 128
    intermediate_size = 128
    group_size = 32
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        _fp8_tensor((num_experts, 2 * intermediate_size, hidden_size)), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        _fp8_tensor((num_experts, hidden_size, intermediate_size)), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            255,
            (num_experts, 2 * intermediate_size, hidden_size // group_size),
            dtype=torch.uint8,
            device="npu",
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            255,
            (num_experts, hidden_size, intermediate_size // group_size),
            dtype=torch.uint8,
            device="npu",
        ),
        requires_grad=False,
    )
    return layer


def _make_method(*, use_eplb: bool) -> AscendW8A8MXFP8DynamicFusedMoEMethod:
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    method.use_eplb = use_eplb
    return method


def _as_nd(tensor: torch.Tensor) -> torch.Tensor:
    if int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ:
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def _byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual = _as_nd(actual)
    if actual.dtype == torch.float8_e4m3fn:
        actual = actual.view(torch.uint8)
        expected = expected.view(torch.uint8)
    return torch.equal(actual.cpu(), expected.cpu())


def _transformed_values(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {
        "w13_weight": layer.w13_weight.data.transpose(1, 2).contiguous().cpu(),
        "w2_weight": layer.w2_weight.data.transpose(1, 2).contiguous().cpu(),
    }
    for scale_name in ("w13_weight_scale", "w2_weight_scale"):
        scale = getattr(layer, scale_name).data
        g_num, n_size, k_size = scale.shape
        values[scale_name] = scale.reshape(g_num, n_size, k_size // 2, 2).transpose(1, 2).contiguous().cpu()
    return values


def test_non_eplb_rl_reload_preserves_nz_execution_buffers():
    layer = _make_layer()
    method = _make_method(use_eplb=False)
    original_shapes = {name: tuple(getattr(layer, name).shape) for name in _PARAM_BUFFER_NAMES}

    method.process_weights_after_loading(layer)
    execution_ptrs = {
        name: getattr(layer, buffer_name).data_ptr() for name, buffer_name in _PARAM_BUFFER_NAMES.items()
    }
    assert int(torch_npu.get_npu_format(layer.w13_weight)) == ACL_FORMAT_FRACTAL_NZ
    assert int(torch_npu.get_npu_format(layer.w2_weight)) == ACL_FORMAT_FRACTAL_NZ
    assert int(torch_npu.get_npu_format(layer.w13_weight_scale)) != ACL_FORMAT_FRACTAL_NZ
    assert int(torch_npu.get_npu_format(layer.w2_weight_scale)) != ACL_FORMAT_FRACTAL_NZ

    for _ in range(3):
        method.restore_weights_for_rl_loading(layer)
        for name, shape in original_shapes.items():
            assert tuple(getattr(layer, name).shape) == shape
        assert int(torch_npu.get_npu_format(layer.w13_weight)) == ACL_FORMAT_FRACTAL_ND
        assert int(torch_npu.get_npu_format(layer.w2_weight)) == ACL_FORMAT_FRACTAL_ND

        layer.w13_weight.data.copy_(_fp8_tensor(original_shapes["w13_weight"]))
        layer.w2_weight.data.copy_(_fp8_tensor(original_shapes["w2_weight"]))
        layer.w13_weight_scale.data.copy_(
            torch.randint(0, 255, original_shapes["w13_weight_scale"], dtype=torch.uint8, device="npu")
        )
        layer.w2_weight_scale.data.copy_(
            torch.randint(0, 255, original_shapes["w2_weight_scale"], dtype=torch.uint8, device="npu")
        )
        expected = _transformed_values(layer)

        method.process_weights_after_loading(layer)
        torch.npu.synchronize()
        for name, expected_value in expected.items():
            assert getattr(layer, name).data_ptr() == execution_ptrs[name]
            assert _byte_equal(getattr(layer, name), expected_value)
        assert int(torch_npu.get_npu_format(layer.w13_weight)) == ACL_FORMAT_FRACTAL_NZ
        assert int(torch_npu.get_npu_format(layer.w2_weight)) == ACL_FORMAT_FRACTAL_NZ

    method.process_weights_after_loading(layer)
    assert {name: getattr(layer, name).data_ptr() for name in _PARAM_BUFFER_NAMES} == execution_ptrs


def test_eplb_keeps_nd_transpose_views():
    layer = _make_layer()
    method = _make_method(use_eplb=True)
    original_shapes = {name: tuple(getattr(layer, name).shape) for name in _PARAM_BUFFER_NAMES}
    original_storage_ptrs = {
        name: getattr(layer, name).untyped_storage().data_ptr() for name in _PARAM_BUFFER_NAMES
    }

    method.process_weights_after_loading(layer)

    for name, buffer_name in _PARAM_BUFFER_NAMES.items():
        tensor = getattr(layer, name)
        assert not tensor.is_contiguous()
        assert int(torch_npu.get_npu_format(tensor)) != ACL_FORMAT_FRACTAL_NZ
        assert tensor.untyped_storage().data_ptr() == original_storage_ptrs[name]
        assert not hasattr(layer, buffer_name)

    for source, view in zip(
        (layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale),
        method.get_eplb_weight_views(layer),
    ):
        assert view.is_contiguous()
        assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert view.view(original_shapes["w13_weight"][0], -1).data_ptr() == source.data_ptr()

    method.restore_weights_for_rl_loading(layer)
    for name, shape in original_shapes.items():
        tensor = getattr(layer, name)
        assert tuple(tensor.shape) == shape
        assert int(torch_npu.get_npu_format(tensor)) != ACL_FORMAT_FRACTAL_NZ
        assert tensor.untyped_storage().data_ptr() == original_storage_ptrs[name]
