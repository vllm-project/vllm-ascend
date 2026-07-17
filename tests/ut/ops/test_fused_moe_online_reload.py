#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod


def _build_method(monkeypatch, release_branch=False):
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    method._maybe_pad_weight = lambda weight: weight

    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=False),
    )
    monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", lambda weight: weight)
    monkeypatch.setattr(fused_moe_module, "vllm_version_is", lambda version: release_branch)
    return method


def _build_layer():
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 3, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 3), requires_grad=False)
    layer.register_buffer("e_score_correction_bias", torch.zeros(2), persistent=True)
    return layer


def _copy_loader(param, loaded_weight):
    with torch.no_grad():
        param.copy_(loaded_weight)


def test_unquantized_moe_declares_native_reload_support():
    assert AscendUnquantizedFusedMoEMethod.requires_native_layerwise_reload is True


@pytest.mark.parametrize("release_branch", [False, True], ids=["main", "release"])
def test_native_reload_preserves_runtime_storage_and_numerics(monkeypatch, release_branch):
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    method = _build_method(monkeypatch, release_branch=release_branch)
    layer = _build_layer()
    object.__setattr__(layer, "quant_method", method)
    layer.w13_weight.weight_loader = _copy_loader
    layer.w2_weight.weight_loader = _copy_loader

    record_metadata_for_reloading(layer)
    method.process_weights_after_loading(layer)

    # Simulate wake_up's checkpoint-layout views before native reload starts.
    processed_storage = (
        layer.w13_weight.untyped_storage().data_ptr(),
        layer.w2_weight.untyped_storage().data_ptr(),
    )
    layer.w13_weight = method.make_weight_loading_view(layer.w13_weight)
    layer.w2_weight = method.make_weight_loading_view(layer.w2_weight)
    assert method.prepare_for_native_layerwise_reload(layer) is True
    assert (
        layer.w13_weight.untyped_storage().data_ptr(),
        layer.w2_weight.untyped_storage().data_ptr(),
    ) == processed_storage

    runtime_w13 = layer.w13_weight
    runtime_w2 = layer.w2_weight
    runtime_correction_bias = layer.e_score_correction_bias
    runtime_pointers = (
        runtime_w13.data_ptr(),
        runtime_w13.untyped_storage().data_ptr(),
        runtime_w2.data_ptr(),
        runtime_w2.untyped_storage().data_ptr(),
        runtime_correction_bias.data_ptr(),
        runtime_correction_bias.untyped_storage().data_ptr(),
    )
    model_input = torch.tensor([[1.0, -0.25, 0.5]], dtype=runtime_w13.dtype)

    for seed in (17, 29):
        generator = torch.Generator().manual_seed(seed)
        checkpoint_w13 = torch.randn((2, 3, 4), generator=generator)
        checkpoint_w2 = torch.randn((2, 4, 3), generator=generator)
        checkpoint_correction_bias = torch.randn((2,), generator=generator)

        initialize_layerwise_reload(layer)
        with torch.no_grad():
            layer.e_score_correction_bias.copy_(checkpoint_correction_bias)
        layer.w13_weight.weight_loader(layer.w13_weight, checkpoint_w13)
        layer.w2_weight.weight_loader(layer.w2_weight, checkpoint_w2)
        finalize_layerwise_reload(layer, SimpleNamespace(dtype=torch.float32))

        assert layer.w13_weight is runtime_w13
        assert layer.w2_weight is runtime_w2
        assert layer.e_score_correction_bias is runtime_correction_bias
        assert (
            runtime_w13.data_ptr(),
            runtime_w13.untyped_storage().data_ptr(),
            runtime_w2.data_ptr(),
            runtime_w2.untyped_storage().data_ptr(),
            runtime_correction_bias.data_ptr(),
            runtime_correction_bias.untyped_storage().data_ptr(),
        ) == runtime_pointers

        torch.testing.assert_close(runtime_w13, checkpoint_w13.transpose(1, 2))
        torch.testing.assert_close(runtime_w2, checkpoint_w2.transpose(1, 2))
        torch.testing.assert_close(runtime_correction_bias, checkpoint_correction_bias)
        runtime_output = model_input @ runtime_w13[0].T @ runtime_w2[0].T
        checkpoint_reference = model_input @ checkpoint_w13[0] @ checkpoint_w2[0]
        torch.testing.assert_close(runtime_output, checkpoint_reference)


def test_loading_view_round_trip_is_zero_copy_and_preserves_loader(monkeypatch):
    method = _build_method(monkeypatch)
    layer = _build_layer()
    layer.w13_weight.weight_loader = _copy_loader
    layer.w2_weight.weight_loader = _copy_loader
    method.process_weights_after_loading(layer)
    layer.w13_weight.unrelated_mutable_metadata = []
    layer.w2_weight.unrelated_mutable_metadata = []

    runtime_shape_stride = {
        "w13_weight": (layer.w13_weight.shape, layer.w13_weight.stride()),
        "w2_weight": (layer.w2_weight.shape, layer.w2_weight.stride()),
    }
    runtime_storage = {name: getattr(layer, name).untyped_storage().data_ptr() for name in ("w13_weight", "w2_weight")}

    for name in ("w13_weight", "w2_weight"):
        setattr(layer, name, method.make_weight_loading_view(getattr(layer, name)))
        assert getattr(getattr(layer, name), "_ascend_moe_checkpoint_layout_view", False) is True
        assert callable(getattr(layer, name).weight_loader)
        assert not hasattr(getattr(layer, name), "unrelated_mutable_metadata")

    assert method.prepare_for_native_layerwise_reload(layer) is True
    assert method.prepare_for_native_layerwise_reload(layer) is False

    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        assert (param.shape, param.stride()) == runtime_shape_stride[name]
        assert param.untyped_storage().data_ptr() == runtime_storage[name]
        assert not hasattr(param, "_ascend_moe_checkpoint_layout_view")
        assert callable(param.weight_loader)
        assert not hasattr(param, "unrelated_mutable_metadata")


@pytest.mark.parametrize("release_branch", [False, True], ids=["main", "release"])
def test_direct_reload_remains_usable_for_two_updates(monkeypatch, release_branch):
    method = _build_method(monkeypatch, release_branch=release_branch)
    layer = _build_layer()
    layer.w13_weight.weight_loader = _copy_loader
    layer.w2_weight.weight_loader = _copy_loader
    method.process_weights_after_loading(layer)
    runtime_storage = (
        layer.w13_weight.untyped_storage().data_ptr(),
        layer.w2_weight.untyped_storage().data_ptr(),
    )

    for seed in (41, 53):
        generator = torch.Generator().manual_seed(seed)
        checkpoint_w13 = torch.randn((2, 3, 4), generator=generator)
        checkpoint_w2 = torch.randn((2, 4, 3), generator=generator)

        layer.w13_weight = method.make_weight_loading_view(layer.w13_weight)
        layer.w2_weight = method.make_weight_loading_view(layer.w2_weight)
        layer.w13_weight.weight_loader(layer.w13_weight, checkpoint_w13)
        layer.w2_weight.weight_loader(layer.w2_weight, checkpoint_w2)
        method.process_weights_after_loading(layer)

        torch.testing.assert_close(layer.w13_weight, checkpoint_w13.transpose(1, 2))
        torch.testing.assert_close(layer.w2_weight, checkpoint_w2.transpose(1, 2))
        assert callable(layer.w13_weight.weight_loader)
        assert callable(layer.w2_weight.weight_loader)
        assert not hasattr(layer.w13_weight, "_ascend_moe_checkpoint_layout_view")
        assert not hasattr(layer.w2_weight, "_ascend_moe_checkpoint_layout_view")
        assert (
            layer.w13_weight.untyped_storage().data_ptr(),
            layer.w2_weight.untyped_storage().data_ptr(),
        ) == runtime_storage


def test_native_reload_rejects_mixed_fused_moe_layouts(monkeypatch):
    method = _build_method(monkeypatch)
    layer = _build_layer()
    method.process_weights_after_loading(layer)
    layer.w13_weight = method.make_weight_loading_view(layer.w13_weight)

    with pytest.raises(RuntimeError, match="mixed runtime and checkpoint-layout"):
        method.prepare_for_native_layerwise_reload(layer)


def test_dynamic_eplb_split_weights_reject_online_reload(monkeypatch):
    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    method = _build_method(monkeypatch)
    method.dynamic_eplb = True
    layer = _build_layer()
    object.__setattr__(layer, "quant_method", method)
    layer.w13_weight.weight_loader = _copy_loader
    layer.w2_weight.weight_loader = _copy_loader
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1),
    )
    monkeypatch.setattr(fused_moe_module.torch_npu, "npu_format_cast", lambda weight, _: weight)
    monkeypatch.setattr(fused_moe_module.torch.npu, "empty_cache", lambda: None)

    record_metadata_for_reloading(layer)
    method.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight_list")
    assert hasattr(layer, "w2_weight_list")

    initialize_layerwise_reload(layer)
    layer.w13_weight.weight_loader(layer.w13_weight, torch.randn(2, 3, 4))
    with pytest.raises(RuntimeError, match="dynamic EPLB"):
        layer.w2_weight.weight_loader(layer.w2_weight, torch.randn(2, 4, 3))
