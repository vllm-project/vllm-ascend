# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    AscendUnquantizedFusedMoEMethod,
    EplbExpertTensorList,
)


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_loader_view_preserves_storage_and_restores_layout(monkeypatch, fail):
    monkeypatch.setattr("torch_npu.get_npu_format", lambda parameter: 2, raising=False)
    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.zeros(2, 3, 8))
    layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 3))
    addresses = [p.data_ptr() for p in layer.parameters()]
    try:
        with torch.no_grad(), method.checkpoint_weight_loader_view(layer):
            assert layer.w13_weight.shape == (2, 8, 3)
            assert layer.w2_weight.shape == (2, 3, 4)
            layer.w13_weight[1, 7, 2] = 9
            layer.w2_weight[0, 2, 3] = 5
            if fail:
                raise RuntimeError("loader failed")
    except RuntimeError:
        assert fail
    assert layer.w13_weight.shape == (2, 3, 8)
    assert layer.w2_weight.shape == (2, 4, 3)
    assert layer.w13_weight[1, 2, 7] == 9
    assert layer.w2_weight[0, 3, 2] == 5
    assert [p.data_ptr() for p in layer.parameters()] == addresses


@pytest.mark.parametrize("nz", [False, True])
def test_checkpoint_loader_view_rejects_unsupported_expert_storage(monkeypatch, nz):
    monkeypatch.setattr("torch_npu.get_npu_format", lambda parameter: 29, raising=False)
    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    layer = SimpleNamespace()
    if nz:
        layer.w13_weight = layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 4))
    with pytest.raises(NotImplementedError), method.checkpoint_weight_loader_view(layer):
        pytest.fail("unsupported layout entered")


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_preserves_independent_expert_tensors():
    expert_tensors = [torch.randn(3, 4), torch.randn(3, 4)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert len(views) == 1
    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].shape == torch.Size([2, 3, 4])
    assert all(actual is expected for actual, expected in zip(views[0], expert_tensors))

    buffer = torch.empty_like(views[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert buffer.shape == views[0].shape
    assert all(tensor.storage_offset() == 0 for tensor in buffer)


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_wrong_expert_tensor_list_length():
    with pytest.raises(ValueError, match="must contain local_num_experts"):
        list(_routed_experts([[torch.randn(3, 4)]]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_ascend_expert_map_follows_model_runner(use_v2_model_runner):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    object.__setattr__(routed_experts, "_use_v2_model_runner", use_v2_model_runner)
    # Both v0.28.0 and main read quant_method in RoutedExperts.expert_map.
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    routed_experts.ascend_expert_map = legacy_map
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    expected = upstream_map if use_v2_model_runner else legacy_map
    assert routed_experts.ascend_expert_map is expected


def test_update_expert_map_preserves_upstream_and_legacy_contracts(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    parent_update_calls = []

    def parent_update(instance):
        parent_update_calls.append(instance)

    monkeypatch.setattr(type(routed_experts).__mro__[1], "update_expert_map", parent_update)
    expert_map_manager = SimpleNamespace(_expert_map=None)
    object.__setattr__(routed_experts, "expert_map_manager", expert_map_manager)

    routed_experts.update_expert_map()

    assert parent_update_calls == [routed_experts]

    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    routed_experts.update_expert_map(legacy_map)

    assert routed_experts.ascend_expert_map is legacy_map
    assert expert_map_manager._expert_map is legacy_map
