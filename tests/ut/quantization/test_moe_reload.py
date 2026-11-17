# SPDX-License-Identifier: Apache-2.0
"""CPU direct-reload transactions; format casts and device kernels are not validated here."""

import pytest
import torch
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from tests.ut.quantization.test_keep_param_objects import (
    MOE_CASES,
    assert_kept,
    config,  # noqa: F401
    get_ascend_config,
    module,
    register,
    snapshot,
)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.reload import refill_expert_list


def runtime_lists(layer):
    return {n: v for n, v in vars(layer).items() if n.endswith("_list") and isinstance(v, list)}


def bytes_of(tensor):
    return tensor.contiguous().view(torch.uint8)


def checkpoint(scheme, value):
    tensors = scheme.get_weight(3, 64, 64, torch.bfloat16)
    tensors.update(scheme.get_dynamic_quant_param(3, 64, 64, torch.bfloat16))
    for name, tensor in tensors.items():
        # Different experts and different rows; changing B must be observable.
        data = torch.arange(tensor.numel()).reshape(tensor.shape) % 5 + value
        if "offset" in name:
            data.zero_()
        elif "scale" in name and tensor.dtype == torch.uint8:
            data = 125 + data % 3  # valid E8M0 encodings
        tensors[name] = data.to(tensor.dtype)
    return tensors


def cold(scheme, values, vc):
    layer = torch.nn.Module()
    layer.activation = "silu"
    layer.quant_method = AscendFusedMoEMethod(scheme, None)
    for name, tensor in values.items():
        register(layer, name, tensor.clone())
    before = snapshot(layer)
    process_weights_after_loading(layer, vc.model_config, torch.device("cpu"))
    assert_kept(layer, before)
    return layer, before


@pytest.mark.parametrize(
    "path,classname,mega,fused_mc2",
    [(*case, mc2) for case in MOE_CASES for mc2 in (0, 1)]
    + [(*("w4a8.w4a8_mxfp4", "AscendW4A8MXFPDynamicFusedMoEMethod", True), mc2) for mc2 in (0, 1)]
    + [
        # W8A8 MegaMoE requires the fused MC2 scales.
        ("w8a8.w8a8_dynamic", "AscendW8A8DynamicFusedMoEMethod", True, 1),
    ],
)
def test_expert_lists_full_a_b_a(config, monkeypatch, path, classname, mega, fused_mc2):  # noqa: F811
    mod = module(path, monkeypatch, mega)
    monkeypatch.setattr(get_ascend_config().eplb_config, "dynamic_eplb", not mega)
    monkeypatch.setattr(get_ascend_config(), "enable_fused_mc2", fused_mc2)
    scheme = getattr(mod, classname)()
    a, b = checkpoint(scheme, 1), checkpoint(scheme, 3)
    layer, before = cold(scheme, a, config)
    held = runtime_lists(layer)
    assert held
    pointers = {n: [t.data_ptr() for t in ts] for n, ts in held.items()}
    initial = {n: [bytes_of(t).clone() for t in ts] for n, ts in held.items()}
    # Re-entering the adapter without a new transaction must not transpose twice.
    layer.quant_method.process_weights_after_loading(layer)
    assert all(torch.equal(bytes_of(t), initial[n][i]) for n, ts in held.items() for i, t in enumerate(ts))
    for values in [b, a]:
        fresh, _ = cold(getattr(mod, classname)(), values, config)
        expected = runtime_lists(fresh)
        start_reload(layer)
        for name, tensor in values.items():
            param = layer.get_parameter(name)
            param.weight_loader(param, tensor)
        previous = {n: [bytes_of(t).clone() for t in ts] for n, ts in held.items()}
        finish_reload(layer, config.model_config, torch.device("cpu"))
        assert_kept(layer, before)
        for name, tensors in held.items():
            assert getattr(layer, name) is tensors, f"list replaced: {name}"
            assert [t.data_ptr() for t in tensors] == pointers[name], name
            for actual, reference in zip(tensors, expected[name]):
                assert torch.equal(bytes_of(actual), bytes_of(reference)), name
        assert any(not torch.equal(bytes_of(t), previous[n][i]) for n, ts in held.items() for i, t in enumerate(ts)), (
            "finish skipped runtime processing"
        )
        if values is b:
            assert any(
                not torch.equal(bytes_of(t), initial[n][i]) for n, ts in held.items() for i, t in enumerate(ts)
            ), "B did not reach runtime experts"
        else:
            assert all(torch.equal(bytes_of(t), initial[n][i]) for n, ts in held.items() for i, t in enumerate(ts))
    with pytest.raises(RuntimeError, match="finish without a begin"):
        finish_reload(layer, config.model_config, torch.device("cpu"))


@pytest.mark.parametrize("change", ["count", "shape", "dtype", "stride"])
def test_expert_layout_change_fails_before_copy(monkeypatch, change):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    layer = torch.nn.Module()
    held = [torch.ones(2, 2), torch.ones(2, 2)]
    layer.weights = held
    new = [torch.full((2, 2), 2.0), torch.full((2, 2), 3.0)]
    if change == "count":
        new.pop()
    elif change == "shape":
        new[1] = torch.zeros(3, 2)
    elif change == "dtype":
        new[1] = new[1].double()
    else:
        new[1] = new[1].t()
    with pytest.raises(ValueError, match="runtime layout changed"):
        refill_expert_list(layer, "weights", new)
    assert layer.weights is held
    assert all(torch.equal(t, torch.ones(2, 2)) for t in held)


def test_e8m0_refill_preserves_all_encodings_and_addresses(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    layer = torch.nn.Module()
    old = torch.zeros(256, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    layer.scales = [old]
    address = old.data_ptr()
    expected = torch.arange(256, dtype=torch.uint8)
    refill_expert_list(layer, "scales", [expected.view(torch.float8_e8m0fnu)])
    assert layer.scales[0] is old and old.data_ptr() == address
    assert torch.equal(old.view(torch.uint8), expected)
