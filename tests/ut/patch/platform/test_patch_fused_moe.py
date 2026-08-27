# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.distributed.eplb.state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.patch.platform import patch_fused_moe


class _Router:
    def __init__(self) -> None:
        self.eplb_state = None

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        return topk_ids

    def _validate_eplb_state(self) -> None:
        return None


class _CustomRoutedExperts(RoutedExperts):
    def weight_loader(self):
        return "custom-loader"


def test_factory_adapts_only_the_returned_router():
    router = _Router()
    untouched_router = _Router()
    runner = SimpleNamespace(router=router)
    original_factory = MagicMock(return_value=runner)
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            expert_map_path=None,
            num_redundant_experts=0,
        )
    )

    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=ascend_config),
        patch.object(
            patch_fused_moe,
            "get_current_vllm_config",
            return_value=SimpleNamespace(use_v2_model_runner=True),
        ),
    ):
        result = patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
            router=router,
            enable_eplb=True,
        )

    assert result is runner
    assert isinstance(router.eplb_state, AscendEplbLayerState)
    assert getattr(router._apply_eplb_mapping, "__func__", None) is patch_fused_moe._ascend_apply_eplb_mapping
    assert getattr(untouched_router._apply_eplb_mapping, "__func__", None) is _Router._apply_eplb_mapping


def test_factory_keeps_v1_eplb_on_the_legacy_routing_path():
    router = _Router()
    runner = SimpleNamespace(router=router)
    original_factory = MagicMock(return_value=runner)
    router_factory = MagicMock(return_value=router)
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            dynamic_eplb=True,
            expert_map_path=None,
            num_redundant_experts=2,
        )
    )

    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "create_ascend_fused_moe_router", router_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=ascend_config),
        patch.object(
            patch_fused_moe,
            "get_current_vllm_config",
            return_value=SimpleNamespace(use_v2_model_runner=False),
        ),
    ):
        result = patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
        )

    assert result is runner
    assert router_factory.call_args.kwargs["eplb_state"] is None
    assert original_factory.call_args.kwargs["enable_eplb"] is True
    assert original_factory.call_args.kwargs["num_redundant_experts"] == 2
    assert router.eplb_state is None
    assert getattr(router._apply_eplb_mapping, "__func__", None) is _Router._apply_eplb_mapping


def test_adapted_router_uses_ascend_mapping_operation():
    router = _Router()
    patch_fused_moe._adapt_eplb_router(router, enable_eplb=True)
    assert isinstance(router.eplb_state, AscendEplbLayerState)
    router.eplb_state.expert_replica_routing_table = torch.tensor([[0, 1]], dtype=torch.int32)
    router.eplb_state.expert_load_view = torch.zeros(2, dtype=torch.int32)
    router.eplb_state.should_record_tensor = torch.tensor(True)
    router.eplb_state.num_unpadded_tokens_tensors = [torch.tensor(1, dtype=torch.int32)]
    topk_ids = torch.tensor([[1]], dtype=torch.int32)
    physical_ids = torch.tensor([[0]], dtype=torch.int32)

    with (
        patch.object(patch_fused_moe, "dbo_current_ubatch_id", return_value=0),
        patch.object(
            patch_fused_moe.torch.ops.vllm,
            "ascend_eplb_map_to_physical_and_record",
            return_value=physical_ids,
        ) as mapping_op,
    ):
        result = router._apply_eplb_mapping(topk_ids)

    assert result is physical_ids
    mapping_op.assert_called_once_with(
        topk_ids,
        router.eplb_state.expert_replica_routing_table,
        router.eplb_state.expert_load_view,
        router.eplb_state.should_record_tensor,
        router.eplb_state.num_unpadded_tokens_tensors[0],
    )


def test_factory_shares_upstream_hash_table_with_legacy_ascend_routing():
    hash_indices_table = torch.tensor([[1, 3]], dtype=torch.int32)
    router = _Router()
    runner = SimpleNamespace(router=router)
    original_factory = MagicMock(return_value=runner)
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            expert_map_path=None,
            num_redundant_experts=0,
        )
    )

    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=ascend_config),
    ):
        patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
            router=router,
            hash_indices_table=hash_indices_table,
        )

    kwargs = original_factory.call_args.kwargs
    assert kwargs["hash_indices_table"] is hash_indices_table
    assert kwargs["routed_experts_args"]["tid2eid"] is hash_indices_table


def test_factory_composes_custom_routed_experts_with_ascend_contract():
    router = _Router()
    runner = SimpleNamespace(router=router)
    original_factory = MagicMock(return_value=runner)
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            expert_map_path=None,
            num_redundant_experts=0,
        )
    )

    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=ascend_config),
    ):
        result = patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
            router=router,
            n_shared_experts=2,
            routed_experts_cls=_CustomRoutedExperts,
        )

    assert result is runner
    kwargs = original_factory.call_args.kwargs
    adapted_cls = kwargs["routed_experts_cls"]
    assert adapted_cls.__mro__[:4] == (
        adapted_cls,
        _CustomRoutedExperts,
        AscendRoutedExperts,
        RoutedExperts,
    )
    assert adapted_cls.__init__ is AscendRoutedExperts.__init__
    assert adapted_cls.weight_loader is _CustomRoutedExperts.weight_loader
    assert kwargs["routed_experts_args"]["n_shared_experts"] == 2
    assert patch_fused_moe._adapt_routed_experts_cls(_CustomRoutedExperts) is adapted_cls
    assert patch_fused_moe._adapt_routed_experts_cls(AscendRoutedExperts) is AscendRoutedExperts
