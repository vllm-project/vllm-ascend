# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.router import fused_topk_router as module
from vllm_ascend.patch.platform import patch_fused_moe


@pytest.fixture(autouse=True)
def hardware_profile():
    with patch.object(module, "get_current_hardware_profile") as profile:
        profile.return_value.supports.return_value = True
        yield profile.return_value


def make_router(**kwargs):
    router = module.AscendFusedTopKRouter(
        top_k=2, global_num_experts=2048, **({"scoring_func": "sqrtsoftplus"} | kwargs)
    )
    device = torch.device("npu:0")
    logits = SimpleNamespace(device=device, dtype=torch.float32, shape=(3, 2048))
    table = SimpleNamespace(device=device, dtype=torch.int32, shape=(1024, 2048), ndim=2, is_contiguous=lambda: True)
    router.eplb_state = SimpleNamespace(
        expert_replica_routing_table=table, allgather_expert_replica_routing_table=table
    )
    router._validate_eplb_state = Mock()
    router._apply_eplb_mapping = MethodType(patch_fused_moe._ascend_apply_eplb_mapping, router)
    return router, logits, table


def routing_table_used(router, logits):
    result = (torch.ones(3, 2), torch.ones(3, 2, dtype=torch.int32), None)
    with (
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", return_value=result, create=True) as op,
        patch.object(module.torch_npu, "npu_moe_gating_top_k", return_value=result, create=True),
    ):
        router._compute_routing(logits, logits, None)
    return op.call_args.kwargs["expert_replica_routing_table"] if op.called else None


@pytest.mark.parametrize("dtype", [None, torch.int32, torch.int64])
def test_fusion_forwards_table_and_dtype_without_second_map(dtype):
    router, logits, table = make_router()
    assert "_select_experts" not in module.AscendFusedTopKRouter.__dict__
    standalone_map = Mock(side_effect=AssertionError("Physical IDs must not be mapped again"))
    op = Mock(return_value=(torch.ones(3, 2), torch.full((3, 2), 2050, dtype=torch.int32), None))
    with (
        patch.object(module, "use_allgather_eplb", return_value=False),
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", op, create=True),
        patch.object(torch.ops.vllm, "ascend_eplb_map_to_physical", standalone_map, create=True),
    ):
        weights, ids = router._select_experts(logits, logits, dtype)
    assert op.call_args.kwargs["expert_replica_routing_table"] is table
    assert ids.dtype == (dtype or torch.int32)
    assert torch.all(ids == 2050)
    assert weights.dtype == torch.float32
    standalone_map.assert_not_called()
    assert router._use_gating_top_k_with_map
    router._validate_eplb_state.assert_called_once()


def test_capture_configured_before_routing_uses_standalone_map():
    router, logits, table = make_router()
    logical_ids = torch.tensor([[1, 2]] * 3, dtype=torch.int32)
    events = []

    def topk(**kwargs):
        ids = logical_ids + 10 if kwargs["expert_replica_routing_table"] is not None else logical_ids
        return torch.ones(3, 2), ids, None

    def map_ids(ids, routing_table):
        assert routing_table is table
        events.append(("map", ids))
        return ids + 10

    with (
        patch.object(module, "use_allgather_eplb", return_value=False),
        patch.object(patch_fused_moe, "use_allgather_eplb", return_value=False),
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", side_effect=topk, create=True),
        patch.object(torch.ops.vllm, "ascend_eplb_map_to_physical", side_effect=map_ids, create=True),
    ):
        router.set_capture_fn(lambda ids: events.append(("capture", ids)))
        _, fallback_ids = router._select_experts(logits, logits)
    assert [name for name, _ in events] == ["capture", "map"]
    assert all(value is logical_ids for _, value in events)
    torch.testing.assert_close(fallback_ids, logical_ids + 10)
    assert not router._use_gating_top_k_with_map


def test_failed_routing_preserves_fixed_choice_for_retry():
    router, logits, _ = make_router()
    with (
        patch.object(module, "use_allgather_eplb", return_value=False),
        patch.object(
            torch.ops._C_ascend, "moe_gating_top_k_hash", side_effect=RuntimeError("kernel failed"), create=True
        ),
        pytest.raises(RuntimeError, match="kernel failed"),
    ):
        router._select_experts(logits, logits)
    assert router._use_gating_top_k_with_map
    with (
        patch.object(module, "use_allgather_eplb", return_value=False),
        patch.object(
            torch.ops._C_ascend,
            "moe_gating_top_k_hash",
            return_value=(torch.ones(3, 2), torch.full((3, 2), 2050, dtype=torch.int32), None),
            create=True,
        ),
    ):
        _, ids = router._select_experts(logits, logits)
    assert torch.all(ids == 2050)
    assert router._use_gating_top_k_with_map


@pytest.mark.parametrize("kwargs", [{"scoring_func": "softmax"}, {"scoring_func": "sigmoid"}, {"num_expert_group": 2}])
def test_unsupported_configuration_falls_back(kwargs):
    router, logits, _ = make_router(**kwargs)
    assert routing_table_used(router, logits) is None


@pytest.mark.parametrize("allgather", [False, True])
def test_missing_table_raises_without_disabling_fusion(allgather):
    router, logits, _ = make_router()
    name = "allgather_expert_replica_routing_table" if allgather else "expert_replica_routing_table"
    setattr(router.eplb_state, name, None)
    with (
        patch.object(module, "use_allgather_eplb", return_value=allgather),
        pytest.raises(RuntimeError, match="initialized expert replica routing table"),
    ):
        routing_table_used(router, logits)
    assert router._use_gating_top_k_with_map


@pytest.mark.parametrize("allgather", [False, True])
def test_table_selection_observes_replacement(allgather):
    router, logits, table = make_router()
    name = "allgather_expert_replica_routing_table" if allgather else "expert_replica_routing_table"
    replacement = SimpleNamespace(**vars(table))
    with patch.object(module, "use_allgather_eplb", return_value=allgather):
        assert routing_table_used(router, logits) is table
        setattr(router.eplb_state, name, replacement)
        assert routing_table_used(router, logits) is replacement


def test_unsupported_hardware_falls_back(hardware_profile):
    router, logits, _ = make_router()
    hardware_profile.supports.return_value = False
    assert routing_table_used(router, logits) is None


def test_eplb_disabled_keeps_standalone_routing():
    router, logits, _ = make_router()
    router.eplb_state = None
    assert routing_table_used(router, logits) is None
    assert not router._use_gating_top_k_with_map


def test_parent_fallback_does_not_mark_ids_as_mapped():
    router, logits, _ = make_router()
    with (
        patch.object(router, "is_fused_supported", return_value=False),
        patch.object(module.AscendGroupedTopKRouter, "_compute_routing", return_value=(None, None)) as fallback,
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", create=True) as op,
    ):
        router._compute_routing(logits, logits, None)
    fallback.assert_called_once()
    op.assert_not_called()
    assert not router._use_gating_top_k_with_map


@pytest.mark.parametrize("vision_supported", [False, True])
def test_vision_branch_selects_mapping_once(hardware_profile, vision_supported):
    router, logits, table = make_router(bias_vl=torch.ones(2048))
    hardware_profile.supports.side_effect = lambda capability: (
        capability != module.HardwareCapability.MOE_GATING_TOP_K_HASH_VISION or vision_supported
    )
    logical_ids = torch.ones(3, 2, dtype=torch.int32)
    context = SimpleNamespace(
        moe_comm_type=module.MoECommType.ALLGATHER,
        moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(all_gather_input_ids=lambda ids: ids)),
    )
    with (
        patch.object(module, "_EXTRA_CTX", context),
        patch.object(module, "use_allgather_eplb", return_value=False),
        patch.object(patch_fused_moe, "use_allgather_eplb", return_value=False),
        patch.object(
            module, "select_deepseek_v4_vision_experts", return_value=(torch.ones(3, 2), logical_ids)
        ) as fallback,
        patch.object(
            torch.ops._C_ascend,
            "moe_gating_top_k_hash",
            return_value=(torch.ones(3, 2), logical_ids + 10, None),
            create=True,
        ) as op,
        patch.object(
            torch.ops.vllm,
            "ascend_eplb_map_to_physical",
            return_value=logical_ids + 10,
            create=True,
        ) as map_op,
    ):
        _, ids = router._select_experts(logits, logits, input_ids=torch.arange(3))
    torch.testing.assert_close(ids, logical_ids + 10)
    if vision_supported:
        assert op.call_args.kwargs["expert_replica_routing_table"] is table
        fallback.assert_not_called()
        map_op.assert_not_called()
    else:
        fallback.assert_called_once()
        op.assert_not_called()
        map_op.assert_called_once_with(logical_ids, table)
    assert router._use_gating_top_k_with_map is vision_supported


@pytest.mark.parametrize("enabled", [False, True])
def test_fusion_choice_is_initialized_once(enabled):
    router, logits, _ = make_router()
    with (
        patch.object(router, "_can_fuse_gating_map", return_value=enabled) as can_fuse,
        patch.object(module, "use_allgather_eplb", return_value=False),
    ):
        routing_table_used(router, logits)
        routing_table_used(router, logits)
    can_fuse.assert_called_once_with(logits)
    assert router._use_gating_top_k_with_map is enabled
