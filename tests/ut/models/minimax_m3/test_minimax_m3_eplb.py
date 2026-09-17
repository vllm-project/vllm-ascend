# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.models.interfaces import MixtureOfExperts

from vllm_ascend.models.minimax_m3 import minimax_m3 as minimax_module
from vllm_ascend.models.minimax_m3 import minimax_m3_vl as vl_module


class _FakeGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.out_dtype = torch.float32

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        self.call_count += 1
        return hidden_states + 10, None


class _FakeExperts(nn.Module):
    def __init__(self, *, is_internal_router: bool = False) -> None:
        super().__init__()
        self.is_internal_router = is_internal_router
        self.moe_config = SimpleNamespace(global_redundant_expert_num=0)
        self.global_redundant_expert_num = 0
        self.update_expert_map = Mock()
        self.last_hidden_states: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.last_hidden_states = hidden_states
        self.last_router_logits = router_logits
        return hidden_states * 2


class _FakeDecoderLayer(nn.Module):
    def __init__(self, moe: nn.Module | None = None) -> None:
        super().__init__()
        self.is_layer_sparse = moe is not None
        if moe is not None:
            self.block_sparse_moe = moe


def _make_moe_metadata() -> minimax_module.MiniMaxM3MoE:
    moe = minimax_module.MiniMaxM3MoE.__new__(minimax_module.MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.experts = _FakeExperts()
    moe.n_logical_experts = 8
    moe.n_physical_experts = 10
    moe.n_local_physical_experts = 5
    moe.n_routed_experts = 8
    moe.n_shared_experts = 1
    moe.n_redundant_experts = 2
    return moe


class TestMiniMaxM3EPLB(unittest.TestCase):
    def test_models_implement_mixture_of_experts_interface(self) -> None:
        self.assertIn(
            MixtureOfExperts,
            minimax_module.MiniMaxM3SparseForCausalLM.__mro__,
        )
        self.assertIn(
            MixtureOfExperts,
            vl_module.MiniMaxM3SparseForConditionalGeneration.__mro__,
        )

    def test_moe_initializes_eplb_metadata_and_factory(self) -> None:
        config = PretrainedConfig(
            hidden_size=64,
            intermediate_size=32,
            num_local_experts=8,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            n_shared_experts=1,
            use_routing_bias=False,
            routed_scaling_factor=1.0,
            swiglu_limit=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        parallel_config = SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=2),
        )
        ep_group = SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 2),
            rank_in_group=1,
        )
        gate = _FakeGate()
        experts = _FakeExperts()

        with (
            patch.object(
                minimax_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                minimax_module,
                "get_ep_group",
                return_value=ep_group,
            ),
            patch.object(
                minimax_module,
                "MiniMaxM3MLP",
                return_value=nn.Identity(),
            ),
            patch.object(
                minimax_module,
                "GateLinear",
                return_value=gate,
            ),
            patch.object(
                minimax_module,
                "FusedMoEFactory",
                return_value=experts,
            ) as fused_moe,
        ):
            moe = minimax_module.MiniMaxM3MoE(
                config=config,
                parallel_config=parallel_config,
                prefix="model.layers.0.block_sparse_moe",
            )

        self.assertEqual(moe.n_logical_experts, 8)
        self.assertEqual(moe.n_physical_experts, 10)
        self.assertEqual(moe.n_local_physical_experts, 5)
        self.assertEqual(moe.physical_expert_start, 5)
        self.assertEqual(moe.physical_expert_end, 10)
        self.assertEqual(experts.global_redundant_expert_num, 2)
        self.assertEqual(
            experts.moe_config.global_redundant_expert_num,
            2,
        )
        kwargs = fused_moe.call_args.kwargs
        self.assertIs(kwargs["gate"], gate)
        self.assertTrue(kwargs["enable_eplb"])
        self.assertEqual(kwargs["num_redundant_experts"], 2)

    def test_moe_forward_selects_router_input(self) -> None:
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        for is_internal_router in (False, True):
            with self.subTest(is_internal_router=is_internal_router):
                moe = minimax_module.MiniMaxM3MoE.__new__(minimax_module.MiniMaxM3MoE)
                nn.Module.__init__(moe)
                gate = _FakeGate()
                experts = _FakeExperts(is_internal_router=is_internal_router)
                moe.gate = gate
                moe.experts = experts

                output = moe(hidden_states)

                torch.testing.assert_close(output, hidden_states * 2)
                torch.testing.assert_close(
                    experts.last_hidden_states,
                    hidden_states,
                )
                if is_internal_router:
                    self.assertEqual(gate.call_count, 0)
                    torch.testing.assert_close(
                        experts.last_router_logits,
                        hidden_states,
                    )
                else:
                    self.assertEqual(gate.call_count, 1)
                    torch.testing.assert_close(
                        experts.last_router_logits,
                        hidden_states + 10,
                    )

    def test_text_and_vl_models_update_expert_metadata(self) -> None:
        moe = _make_moe_metadata()
        sparse_layer = _FakeDecoderLayer(moe)
        dense_layer = _FakeDecoderLayer()

        language_model = minimax_module.MiniMaxM3SparseForCausalLM.__new__(minimax_module.MiniMaxM3SparseForCausalLM)
        nn.Module.__init__(language_model)
        language_model.model = nn.Module()
        language_model.model.layers = nn.ModuleList([dense_layer, sparse_layer])

        with patch.object(
            minimax_module,
            "MiniMaxM3DecoderLayer",
            _FakeDecoderLayer,
        ):
            language_model._set_moe_parameters()

        self.assertEqual(language_model.num_moe_layers, 1)
        self.assertEqual(language_model.moe_layers, [moe.experts])
        self.assertEqual(language_model.moe_mlp_layers, [moe])
        self.assertEqual(language_model.num_logical_experts, 8)
        self.assertEqual(language_model.num_redundant_experts, 2)

        vl_model = vl_module.MiniMaxM3SparseForConditionalGeneration.__new__(
            vl_module.MiniMaxM3SparseForConditionalGeneration
        )
        nn.Module.__init__(vl_model)
        vl_model.language_model = language_model
        vl_model._sync_moe_parameters()
        vl_model.update_physical_experts_metadata(12, 5)

        self.assertIs(vl_model.moe_layers, language_model.moe_layers)
        self.assertEqual(vl_model.num_physical_experts, 12)
        self.assertEqual(vl_model.num_redundant_experts, 4)
        self.assertEqual(moe.n_physical_experts, 12)
        self.assertEqual(moe.n_redundant_experts, 4)
        moe.experts.update_expert_map.assert_called_once_with()
        self.assertEqual(
            moe.experts.global_redundant_expert_num,
            4,
        )
        self.assertEqual(
            moe.experts.moe_config.global_redundant_expert_num,
            4,
        )

    def test_model_without_sparse_layers_has_zero_expert_metadata(
        self,
    ) -> None:
        language_model = minimax_module.MiniMaxM3SparseForCausalLM.__new__(minimax_module.MiniMaxM3SparseForCausalLM)
        nn.Module.__init__(language_model)
        language_model.model = nn.Module()
        language_model.model.layers = nn.ModuleList([_FakeDecoderLayer()])

        with patch.object(
            minimax_module,
            "MiniMaxM3DecoderLayer",
            _FakeDecoderLayer,
        ):
            language_model._set_moe_parameters()

        self.assertEqual(language_model.num_moe_layers, 0)
        self.assertEqual(language_model.num_logical_experts, 0)
        self.assertEqual(language_model.num_physical_experts, 0)
        self.assertEqual(language_model.num_redundant_experts, 0)


if __name__ == "__main__":
    unittest.main()
