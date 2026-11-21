# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
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

    with patch.object(
        patch_fused_moe.torch.ops.vllm,
        "ascend_eplb_map_to_physical",
        return_value=physical_ids,
    ) as mapping_op:
        result = router._apply_eplb_mapping(topk_ids)

    assert result is physical_ids
    mapping_op.assert_called_once_with(
        topk_ids,
        router.eplb_state.expert_replica_routing_table,
    )


def test_factory_passes_hash_table_only_through_upstream_path():
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
    assert "tid2eid" not in kwargs["routed_experts_args"]


def test_factory_keeps_vision_bias_in_ascend_router_only():
    bias_vl = torch.arange(8, dtype=torch.float32)
    router = _Router()
    runner = SimpleNamespace(router=router)
    original_factory = MagicMock(return_value=runner)
    router_factory = MagicMock(return_value=router)
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            expert_map_path=None,
            num_redundant_experts=0,
        )
    )

    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "create_ascend_fused_moe_router", router_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=ascend_config),
    ):
        patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
            bias_vl=bias_vl,
            image_sentinel_lo=129257,
        )

    router_kwargs = router_factory.call_args.kwargs
    assert router_kwargs["bias_vl"] is bias_vl
    assert router_kwargs["image_sentinel_lo"] == 129257
    factory_kwargs = original_factory.call_args.kwargs
    assert "bias_vl" not in factory_kwargs
    assert "image_sentinel_lo" not in factory_kwargs


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


@pytest.mark.parametrize("scale_output", [False, True])
@pytest.mark.parametrize("guard_shared", [False, True])
def test_factory_maps_legacy_scaling_to_the_active_native_contract(scale_output, guard_shared):
    router = _Router()
    router_factory = MagicMock(return_value=router)
    original_factory = MagicMock(return_value=SimpleNamespace(router=router))
    config = SimpleNamespace(
        eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None, num_redundant_experts=0)
    )
    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=config),
        patch.object(
            patch_fused_moe, "get_current_vllm_config", return_value=SimpleNamespace(use_v2_model_runner=True)
        ),
        patch.object(patch_fused_moe, "create_ascend_fused_moe_router", router_factory),
    ):
        patch_fused_moe._ascend_FusedMoE(
            num_experts=8,
            top_k=2,
            routed_scaling_factor=2.0,
            apply_routed_scale_to_output=scale_output,
            divide_shared_output_in_fp16=guard_shared,
        )
    kwargs = original_factory.call_args.kwargs
    if patch_fused_moe.vllm_version_is("0.33.0"):
        assert kwargs["apply_routed_scale_to_output"] is scale_output
        assert "routed_scale_mode" not in kwargs
        assert "divide_shared_output_in_fp16" not in kwargs
        scales_output = scale_output
    else:
        mode = kwargs["routed_scale_mode"]
        assert mode.value == (
            "output_fp16_guard" if scale_output and guard_shared else "output" if scale_output else "router"
        )
        assert "apply_routed_scale_to_output" not in kwargs
        assert "divide_shared_output_in_fp16" not in kwargs
        scales_output = mode.scales_output
    assert router_factory.call_args.kwargs["routed_scaling_factor"] == (1.0 if scales_output else 2.0)


@pytest.mark.parametrize("mode_name", ["ROUTER", "OUTPUT", "OUTPUT_FP16_GUARD"])
def test_factory_preserves_native_scaling_modes(mode_name):
    router = _Router()
    router_factory = MagicMock(return_value=router)
    original_factory = MagicMock(return_value=SimpleNamespace(router=router))
    config = SimpleNamespace(
        eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None, num_redundant_experts=0)
    )
    main = not patch_fused_moe.vllm_version_is("0.33.0")
    scaling: dict[str, Any] = {}
    if main:
        from vllm.model_executor.layers.fused_moe.config import RoutedScaleMode

        scaling["routed_scale_mode"] = RoutedScaleMode[mode_name]
    else:
        scaling["apply_routed_scale_to_output"] = mode_name != "ROUTER"
    with (
        patch.object(patch_fused_moe, "_original_FusedMoE", original_factory),
        patch.object(patch_fused_moe, "get_ascend_config", return_value=config),
        patch.object(
            patch_fused_moe, "get_current_vllm_config", return_value=SimpleNamespace(use_v2_model_runner=True)
        ),
        patch.object(patch_fused_moe, "create_ascend_fused_moe_router", router_factory),
    ):
        patch_fused_moe._ascend_FusedMoE(num_experts=8, top_k=2, routed_scaling_factor=2.0, **scaling)
    kwargs = original_factory.call_args.kwargs
    if main:
        assert kwargs["routed_scale_mode"] is scaling["routed_scale_mode"]
        assert "apply_routed_scale_to_output" not in kwargs
    else:
        assert kwargs["apply_routed_scale_to_output"] is scaling["apply_routed_scale_to_output"]
        assert "routed_scale_mode" not in kwargs
    assert router_factory.call_args.kwargs["routed_scaling_factor"] == (2.0 if mode_name == "ROUTER" else 1.0)
