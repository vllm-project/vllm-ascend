# SPDX-License-Identifier: Apache-2.0
"""ND runtime-address regressions; CPU fixtures do not validate NPU kernels."""

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from tests.ut.quantization.test_keep_param_objects import (
    assert_kept,
    config,  # noqa: F401
    get_ascend_config,
    module,
    register,
    snapshot,
)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


def build(case, value, vc, monkeypatch, device=None):
    device = torch.device("cpu") if device is None else device
    get_ascend_config().eplb_config.dynamic_eplb = case == "w4-moe-list"
    get_ascend_config().enable_fused_mc2 = 1 if case == "w8-moe" else 0
    if case == "w8-moe" and device.type == "cpu":
        # CPU allocation regression; real NZ layout is tested on NPU.
        monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, *args, **kwargs: tensor.clone())
    layer = torch.nn.Module()
    layer.activation = "silu"
    if case in ("w4-moe", "w4-moe-list", "w4-moe-compressed", "w8-moe", "mx-moe-odd"):
        path, classname = {
            "w4-moe": ("w4a8.w4a8", "AscendW4A8DynamicFusedMoEMethod"),
            "w4-moe-list": ("w4a8.w4a8", "AscendW4A8DynamicFusedMoEMethod"),
            "w4-moe-compressed": ("w4a8.w4a8", "AscendW4A8DynamicFusedMoEMethod"),
            "w8-moe": ("w8a8.w8a8_dynamic", "AscendW8A8DynamicFusedMoEMethod"),
            "mx-moe-odd": ("w4a4.w4a4_mxfp4", "AscendW4A4MXFP4DynamicFusedMoEMethod"),
        }[case]
        scheme = getattr(module(path, monkeypatch), classname)()
        if case == "w4-moe-compressed":
            scheme.quant_method = COMPRESSED_TENSORS_METHOD
        size = 96 if case == "mx-moe-odd" else 64
        values = scheme.get_weight(3, size, size, torch.bfloat16)
        values.update(scheme.get_dynamic_quant_param(3, size, size, torch.bfloat16))
        layer.quant_method = AscendFusedMoEMethod(scheme, None)
    elif case.startswith("flat"):
        mx = case == "flat-mx"
        mod = module("w4a4.w4a4_mxfp4_flatquant" if mx else "w4a4.w4a4_flatquant", monkeypatch)
        scheme = (
            mod.AscendW4A4MXFP4FlatQuantDynamicLinearMethod if mx else mod.AscendW4A4FlatQuantDynamicLinearMethod
        )()
        values = {
            "weight": torch.empty(16, 32 if mx else 64, dtype=torch.uint8 if mx else torch.int8),
            "weight_scale": torch.empty((16, 2) if mx else (16, 1), dtype=torch.uint8 if mx else torch.bfloat16),
            "weight_offset": torch.zeros(16, 1, dtype=torch.bfloat16),
            "left_trans": torch.eye(8, dtype=torch.bfloat16),
            "right_trans": torch.eye(8, dtype=torch.bfloat16),
            "clip_ratio": torch.tensor([0.75]),
        }
        layer.quant_method = AscendLinearMethod(scheme)
    elif case == "w4-linear":
        scheme = module("w4a8.w4a8", monkeypatch).AscendW4A8DynamicLinearMethod()
        values = {
            "weight": torch.empty(32, 64, dtype=torch.int8),
            "weight_scale": torch.empty(64, 1),
            "weight_offset": torch.zeros(64, 1),
            "scale_bias": torch.empty(64, 16),
        }
        layer.quant_method = AscendLinearMethod(scheme)
    else:
        scheme = module("w4a4.w4a4_mxfp4", monkeypatch).AscendW4A4MXFP4DynamicLinearMethod()
        size = 96
        if case == "mx-linear-tp-pad":
            layer = RowParallelLinear.__new__(RowParallelLinear)
            torch.nn.Module.__init__(layer)
            layer.tp_rank = 1
            layer.input_size_per_partition = size = 48
            # Exercise PWAL's real isinstance/TP-padding branch, not the model's TP setup.
            layer.update_param_tp_status = lambda: None
        values = scheme.get_weight(size, 16, torch.bfloat16)
        values.update(scheme.get_pergroup_param(size, 16, torch.bfloat16))
        layer.quant_method = AscendLinearMethod(scheme)
    for name, tensor in list(values.items()):
        data = torch.arange(tensor.numel()).reshape(tensor.shape) % 5 + value
        if name == "clip_ratio":
            data = tensor
        elif "offset" in name:
            data.zero_()
        elif "scale" in name and tensor.dtype == torch.uint8:
            data = 125 + data % 3
        values[name] = data.to(tensor.dtype)
        register(layer, name, values[name].clone())
    before = snapshot(layer)
    process_weights_after_loading(layer, vc.model_config, device)
    return layer, values, before


def runtime(layer):
    result = {name: p for name, p in layer.named_parameters() if not getattr(p, "retired", False)}
    for name in ("w13_weight_scale_fp32", "fused_w1_scale", "fused_w2_scale", "weight_scale_fp32"):
        tensor = getattr(layer, name, None)
        if tensor is not None:
            result[name] = tensor
    return result


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


@pytest.mark.parametrize(
    "case",
    [
        "w4-linear",
        "w4-moe",
        "w4-moe-compressed",
        "w8-moe",
        "mx-linear-odd",
        "mx-linear-tp-pad",
        "mx-moe-odd",
        "flat-int",
        "flat-mx",
    ],
)
def test_nd_tensor_runtime_a_b_a(config, monkeypatch, case):  # noqa: F811
    layer, _, before = build(case, 1, config, monkeypatch)
    held = {name: tensor.detach() for name, tensor in runtime(layer).items()}
    pointers = {name: tensor.data_ptr() for name, tensor in held.items()}
    initial = {name: raw(tensor).clone() for name, tensor in held.items()}
    for value in (3, 1):
        fresh, values, _ = build(case, value, config, monkeypatch)
        start_reload(layer)
        for name, tensor in values.items():
            parameter = layer.get_parameter(name)
            parameter.weight_loader(parameter, tensor)
        finish_reload(layer, config.model_config, torch.device("cpu"))
        assert_kept(layer, before)
        current, expected = runtime(layer), runtime(fresh)
        assert current.keys() == expected.keys() == held.keys()
        moved = [name for name, tensor in current.items() if tensor.data_ptr() != pointers[name]]
        assert not moved, f"kernel storage moved: {moved}"
        for name, tensor in current.items():
            assert torch.equal(raw(tensor), raw(expected[name])), name
            assert torch.equal(raw(held[name]), raw(expected[name])), f"held reference is stale: {name}"
        if value == 3:
            assert any(not torch.equal(raw(t), initial[n]) for n, t in current.items())
        else:
            assert all(torch.equal(raw(t), initial[n]) for n, t in current.items())


@pytest.mark.parametrize("case", ["flat-int", "flat-mx"])
def test_changed_flatquant_clip_refused_and_full_retry_recovers(config, monkeypatch, case):  # noqa: F811
    layer, values, _ = build(case, 1, config, monkeypatch)
    start_reload(layer)
    for name, tensor in values.items():
        parameter = layer.get_parameter(name)
        parameter.weight_loader(parameter, torch.tensor([0.5]) if name == "clip_ratio" else tensor)
    with pytest.raises(ValueError, match="cannot change FlatQuant clip_ratio"):
        finish_reload(layer, config.model_config, torch.device("cpu"))
    assert layer.aclnn_clip_ratio == 0.75
    # Failed direct finish has no rollback. A new complete transaction is required.
    start_reload(layer)
    for name, tensor in values.items():
        parameter = layer.get_parameter(name)
        parameter.weight_loader(parameter, tensor)
    finish_reload(layer, config.model_config, torch.device("cpu"))
    fresh, _, _ = build(case, 1, config, monkeypatch)
    assert all(torch.equal(raw(t), raw(runtime(fresh)[n])) for n, t in runtime(layer).items())


@pytest.mark.parametrize("case", ["flat-int", "flat-mx"])
def test_flatquant_non_direct_pwal_accepts_changed_clip(config, monkeypatch, case):  # noqa: F811
    layer, values, _ = build(case, 1, config, monkeypatch)
    # Restore checkpoint layout before exercising the existing non-direct PWAL.
    start_reload(layer)
    for name, tensor in values.items():
        parameter = layer.get_parameter(name)
        parameter.weight_loader(parameter, torch.tensor([0.5]) if name == "clip_ratio" else tensor)
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    layer.quant_method.quant_method.process_weights_after_loading(layer)
    assert layer.aclnn_clip_ratio == 0.5
