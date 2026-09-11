# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    EplbExpertTensorList,
    pad_static_expert_capacity,
)
from vllm_ascend.utils import vllm_version_is


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
    if not vllm_version_is("0.28.0"):
        # main (cdc4824a21): RoutedExperts.expert_map reads quant_method.moe_kernel
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


@pytest.mark.parametrize("ep_size", [1, 4, 6, 12])
def test_static_expert_padding_preserves_checkpoint_and_dispatch_placement(ep_size):
    logical_count = 128
    physical_count = ((logical_count + ep_size - 1) // ep_size) * ep_size
    placement = []
    for rank in range(ep_size):
        parallel = SimpleNamespace(
            use_ep=ep_size > 1,
            enable_eplb=False,
            ep_size=ep_size,
            ep_rank=rank,
            needs_round_robin_routing_tables=False,
        )
        manager = ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=8,
            global_num_experts=logical_count,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        config = SimpleNamespace(
            num_experts=logical_count,
            num_logical_experts=logical_count,
            num_local_experts=manager.local_num_experts,
            moe_parallel_config=parallel,
            ep_size=ep_size,
            ep_rank=rank,
        )
        padding = pad_static_expert_capacity(config, manager)

        assert padding == physical_count - logical_count
        assert config.num_logical_experts == logical_count
        assert config.num_experts == physical_count
        assert config.num_local_experts == physical_count // ep_size
        _, dispatch_map, log2phy, redundancy = init_eplb_config(
            SimpleNamespace(dynamic_eplb=False, expert_map_path=None, num_redundant_experts=0),
            0,
            config,
        )
        assert redundancy == 0  # Dummy slots are not EPLB replicas.
        assert log2phy is None
        if ep_size == 1:
            assert manager.expert_map is dispatch_map is None
            continue
        torch.testing.assert_close(dispatch_map, manager.expert_map)
        # A checkpoint expert must land in the same local slot used by dispatch.
        local_count = config.num_local_experts
        expected = torch.full((physical_count,), -1, dtype=torch.int32)
        expected[rank * local_count : (rank + 1) * local_count] = torch.arange(local_count, dtype=torch.int32)
        torch.testing.assert_close(manager.expert_map, expected)
        placement.append(manager.expert_map[:logical_count] >= 0)
    if placement:
        assert torch.stack(placement).sum(dim=0).tolist() == [1] * logical_count


def test_static_padding_leaves_eplb_capacity_unchanged():
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True, enable_eplb=True))
    # EPLB owns its capacity and placement; static padding must not touch either.
    assert pad_static_expert_capacity(config, None) == 0


def test_padded_profile_routes_only_to_logical_experts(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.routed_experts.get_ascend_config",
        lambda: SimpleNamespace(enable_force_eplb=False),
    )
    weights = torch.ones(16, 8)
    layer = SimpleNamespace(
        router=SimpleNamespace(_select_experts=lambda **kwargs: (weights, torch.zeros(16, 8, dtype=torch.int32))),
        log2phy=None,
        n_shared_experts=0,
        moe_config=SimpleNamespace(num_experts=132, num_logical_experts=128),
        global_redundant_expert_num=0,
    )
    _, ids = AscendRoutedExperts._select_experts(layer, torch.ones(16, 4), torch.ones(16, 128), True)
    assert ids.shape == (16, 8)
    assert torch.all((ids >= 0) & (ids < 128))
