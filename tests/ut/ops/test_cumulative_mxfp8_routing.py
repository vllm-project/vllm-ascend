# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import PropertyMock, patch

import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    EXPERT_TOKEN_NUMS_TYPE_COUNT,
    EXPERT_TOKEN_NUMS_TYPE_CUMSUM,
    TokenDispatcherWithAllGather,
)
from vllm_ascend.quantization.quant_type import QuantType


class TestCumulativeMxfp8Routing(TestCase):
    def test_format_selection_and_consumer_contract(self):
        for case in ("mxfp8", "mapped", "distributed", "redundant", "lora", "bf16", "mxfp4"):
            with self.subTest(case=case):
                quant_type = {"bf16": QuantType.NONE, "mxfp4": QuantType.W4A4MXFP}.get(case, QuantType.W8A8MXFP)
                dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=4, num_local_experts=4)
                if case == "lora":
                    dispatcher.lora_context = SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False))
                states = torch.randn(3, 128, dtype=torch.bfloat16)
                ids = torch.tensor([[0, 2], [2, 3], [0, 3]], dtype=torch.int32)
                weights = torch.full((3, 2), 0.5)
                expected_type = EXPERT_TOKEN_NUMS_TYPE_CUMSUM if case == "mxfp8" else EXPERT_TOKEN_NUMS_TYPE_COUNT
                groups = torch.tensor([2, 2, 4, 6] if case == "mxfp8" else [2, 0, 2, 2], dtype=torch.int64)
                dispatch_input = MoETokenDispatchInput(
                    hidden_states=states,
                    topk_weights=weights,
                    topk_ids=ids,
                    routing=MoeRouterInput(
                        expert_map=torch.arange(4) if case == "mapped" else None,
                        global_redundant_expert_num=int(case == "redundant"),
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(
                        quant_type=quant_type,
                        mxfp=MoEMxfpParams(act_quant_type=torch.float8_e4m3fn),
                    ),
                )
                with (
                    patch.object(
                        TokenDispatcherWithAllGather,
                        "ep_size",
                        new_callable=PropertyMock,
                        return_value=2 if case == "distributed" else 1,
                    ),
                    patch(
                        "vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group",
                        return_value=SimpleNamespace(rank_in_group=0),
                    ),
                    patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_dynamic_mx_quant_scale_alg", return_value=0),
                    patch("vllm_ascend.ops.fused_moe.token_dispatcher.validate_quant_moe_lora_activation_input"),
                    patch(
                        "vllm_ascend.ops.fused_moe.token_dispatcher.DeviceOperator.npu_moe_init_routing",
                        return_value=(states, torch.arange(6), groups, None),
                    ) as route,
                ):
                    result = dispatcher.token_dispatch(dispatch_input)
                self.assertEqual(route.call_args.kwargs["expert_tokens_num_type"], expected_type)
                self.assertEqual(result.group_list_type, expected_type)
                torch.testing.assert_close(result.group_list, groups)
                torch.testing.assert_close(result.combine_metadata.topk_weights, weights)
                if case == "mxfp8":
                    with patch.object(torch.Tensor, "cumsum", side_effect=AssertionError("unnecessary cumsum")):
                        self.assertIs(
                            cumsum_group_list(result.group_list, result.group_list_type, 0), result.group_list
                        )
                else:
                    torch.testing.assert_close(
                        cumsum_group_list(result.group_list, result.group_list_type, 0), torch.tensor([2, 2, 4, 6])
                    )
