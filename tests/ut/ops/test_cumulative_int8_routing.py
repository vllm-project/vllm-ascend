# SPDX-License-Identifier: Apache-2.0
"""The producer representation must match the opted-in INT8 consumer."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import PropertyMock, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import token_dispatcher as dispatchers
from vllm_ascend.ops.fused_moe.dataclass import token_dispatcher as dispatch_data
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list
from vllm_ascend.quantization.quant_type import QuantType


class TestCumulativeInt8Routing(TestCase):
    def test_consumer_preference_preserves_other_mlp_paths(self):
        cases = [
            (QuantType.W8A8, MoEActivation.SILU, False, True, True),
            (QuantType.W8A8, "silu", False, True, True),
            (QuantType.W8A8, MoEActivation.SILU, False, False, False),
            (QuantType.W8A8, MoEActivation.SILU, True, True, False),
            (QuantType.W8A8, MoEActivation.SWIGLUOAI_UNINTERLEAVE, False, True, False),
            (QuantType.W4A8, MoEActivation.SILU, False, True, False),
            (QuantType.W8A8MXFP, MoEActivation.SILU, False, True, False),
            (QuantType.NONE, MoEActivation.SILU, False, True, False),
        ]
        for quant_type, activation, eplb, fusion, expected in cases:
            with self.subTest(quant=quant_type, activation=activation, eplb=eplb, fusion=fusion):
                source = SimpleNamespace(
                    hidden_states=torch.zeros(1, 128),
                    topk_weights=torch.ones(1, 2),
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                    routing=MoeRouterInput(
                        expert_map=None,
                        global_redundant_expert_num=0,
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(quant_type=quant_type),
                    activation=activation,
                    dynamic_eplb=eplb,
                )
                with patch.object(dispatch_data, "enable_fusion_gmmswigluquant", return_value=fusion):
                    result = dispatch_data.build_token_dispatch_input(fused_experts_input=source)
                self.assertEqual(result.prefer_cumulative_group_list, expected)
                self.assertIs(result.hidden_states, source.hidden_states)
                self.assertIs(result.quant, source.quant)

    def test_routing_format_and_semantic_fallbacks(self):
        cases = ("eligible", "legacy", "mapped", "ep", "redundant", "lora", "other_quant", "a3", "a5", "310p")
        for case in cases:
            with self.subTest(case=case):
                states = torch.randn(3, 128)
                ids = torch.tensor([[0, 2], [2, 3], [0, 3]], dtype=torch.int32)
                weights = torch.full((3, 2), 0.5)
                prefix = case == "eligible"
                groups = torch.tensor([2, 2, 4, 6] if prefix else [2, 0, 2, 2], dtype=torch.int64)
                dispatcher = dispatchers.TokenDispatcherWithAllGather(top_k=2, num_experts=4, num_local_experts=4)
                if case == "lora":
                    dispatcher.lora_context = SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False))
                source = dispatch_data.MoETokenDispatchInput(
                    hidden_states=states,
                    topk_weights=weights,
                    topk_ids=ids,
                    routing=MoeRouterInput(
                        expert_map=torch.arange(4) if case == "mapped" else None,
                        global_redundant_expert_num=int(case == "redundant"),
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(quant_type=QuantType.W4A8 if case == "other_quant" else QuantType.W8A8),
                    prefer_cumulative_group_list=case != "legacy",
                )
                device = {"a3": AscendDeviceType.A3, "a5": AscendDeviceType.A5, "310p": AscendDeviceType._310P}.get(
                    case, AscendDeviceType.A2
                )
                with (
                    patch.object(
                        dispatchers.TokenDispatcherWithAllGather,
                        "ep_size",
                        new_callable=PropertyMock,
                        return_value=2 if case == "ep" else 1,
                    ),
                    patch.object(dispatchers, "get_ep_group", return_value=SimpleNamespace(rank_in_group=0)),
                    patch.object(
                        dispatchers, "get_current_hardware_profile", return_value=get_hardware_profile(device)
                    ),
                    patch.object(dispatchers, "validate_quant_moe_lora_activation_input"),
                    patch.object(
                        dispatchers.DeviceOperator,
                        "npu_moe_init_routing",
                        return_value=(states, torch.arange(6), groups, None),
                    ) as route,
                ):
                    result = dispatcher.token_dispatch(source)
                self.assertEqual(route.call_args.kwargs["expert_tokens_num_type"], 0 if prefix else 1)
                self.assertEqual(result.group_list_type, 0 if prefix else 1)
                torch.testing.assert_close(result.group_list, groups)
                torch.testing.assert_close(result.combine_metadata.topk_weights, weights)

    def test_cumulative_group_list_identity(self):
        groups = torch.tensor([2, 2, 4, 6], dtype=torch.int64)
        with patch.object(torch.Tensor, "cumsum", side_effect=AssertionError("redundant conversion")):
            self.assertIs(cumsum_group_list(groups, 0, 0), groups)
