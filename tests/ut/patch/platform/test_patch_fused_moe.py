# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)

from vllm_ascend.patch.platform import patch_fused_moe


class TestAscendRouterPatch(unittest.TestCase):
    def test_fused_topk_uses_ascend_selection_before_upstream_eplb_mapping(
        self,
    ):
        logical_ids = torch.tensor([[1, 3]], dtype=torch.int32)
        physical_ids = torch.tensor([[4, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
        select_experts = MagicMock(return_value=(weights, logical_ids))
        router = FusedTopKRouter(
            top_k=2,
            global_num_experts=8,
            scoring_func="softmax",
            renormalize=True,
        )
        apply_eplb_mapping = MagicMock(return_value=physical_ids)

        with (
            patch.object(
                patch_fused_moe,
                "_ascend_select_experts",
                select_experts,
            ),
            patch.object(
                router,
                "_apply_eplb_mapping",
                apply_eplb_mapping,
            ),
        ):
            result_weights, result_ids = router.select_experts(
                hidden_states=torch.randn(1, 16),
                router_logits=torch.randn(1, 8),
            )

        self.assertIs(result_weights, weights)
        self.assertIs(result_ids, physical_ids)
        select_experts.assert_called_once()
        self.assertFalse(select_experts.call_args.kwargs["use_grouped_topk"])
        apply_eplb_mapping.assert_called_once_with(logical_ids)

    def test_grouped_topk_uses_ascend_selection_and_applies_scaling_once(
        self,
    ):
        logical_ids = torch.tensor([[1, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.4, 0.1]], dtype=torch.float32)
        select_experts = MagicMock(return_value=(weights, logical_ids))
        router = GroupedTopKRouter(
            top_k=2,
            global_num_experts=8,
            num_expert_group=2,
            topk_group=1,
            routed_scaling_factor=2.0,
        )

        with patch.object(
            patch_fused_moe,
            "_ascend_select_experts",
            select_experts,
        ):
            result_weights, result_ids = router._compute_routing(
                hidden_states=torch.randn(1, 16),
                router_logits=torch.randn(1, 8),
                indices_type=torch.int32,
            )

        torch.testing.assert_close(result_weights, weights * 2.0)
        self.assertIs(result_ids, logical_ids)
        kwargs = select_experts.call_args.kwargs
        self.assertTrue(kwargs["use_grouped_topk"])
        self.assertEqual(kwargs["num_expert_group"], 2)
        self.assertEqual(kwargs["topk_group"], 1)
        self.assertEqual(kwargs["routed_scaling_factor"], 1.0)

    def test_fused_topk_bias_preserves_upstream_shared_expert_slots(
        self,
    ):
        logical_ids = torch.tensor([[1, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.4, 0.1]], dtype=torch.float32)
        select_experts = MagicMock(return_value=(weights, logical_ids))
        router = FusedTopKBiasRouter(
            top_k=2,
            global_num_experts=8,
            e_score_correction_bias=torch.zeros(8),
            routed_scaling_factor=2.0,
            num_fused_shared_experts=1,
            shared_expert_weight=0.5,
        )

        with patch.object(
            patch_fused_moe,
            "_ascend_select_experts",
            select_experts,
        ):
            result_weights, result_ids = router._compute_routing(
                hidden_states=torch.randn(1, 16),
                router_logits=torch.randn(1, 8),
                indices_type=torch.int32,
            )

        torch.testing.assert_close(
            result_weights,
            torch.tensor([[0.8, 0.2, 0.5]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            result_ids,
            torch.tensor([[1, 3, 8]], dtype=torch.int32),
        )
        self.assertEqual(
            select_experts.call_args.kwargs["routed_scaling_factor"],
            1.0,
        )

    def test_router_compute_patch_is_idempotent(self):
        patched_method = FusedTopKRouter._compute_routing

        patch_fused_moe._patch_router_compute()

        self.assertIs(FusedTopKRouter._compute_routing, patched_method)
