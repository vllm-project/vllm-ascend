from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicFusedMoEMethod,
    _force_load_balance_topk_ids,
)


class TestAscendW4A8MXFP4ForcedLoadBalance(TestBase):
    def test_matches_local_expert_loads_across_ep_ranks(self):
        topk_ids = torch.zeros((10, 2), dtype=torch.int32)

        balanced_ids = _force_load_balance_topk_ids(
            topk_ids,
            num_experts=8,
            ep_size=4,
            capturing=False,
        )

        per_expert_load = torch.bincount(balanced_ids.flatten().to(torch.int64), minlength=8).reshape(4, 2)
        self.assertTrue(torch.equal(per_expert_load, torch.tensor([[3, 2], [3, 2], [3, 2], [3, 2]])))
        self.assertTrue(
            all(torch.unique(token_experts).numel() == token_experts.numel() for token_experts in balanced_ids)
        )
        self.assertEqual(balanced_ids.dtype, topk_ids.dtype)
        self.assertEqual(balanced_ids.device, topk_ids.device)

    def test_rejects_graph_capture(self):
        with self.assertRaisesRegex(RuntimeError, "only outside graph capture"):
            _force_load_balance_topk_ids(
                torch.zeros((4, 2), dtype=torch.int64),
                num_experts=4,
                ep_size=2,
                capturing=True,
            )

    def test_requires_exact_rank_divisibility(self):
        with self.assertRaisesRegex(ValueError, "Routed rows .* divisible by EP size"):
            _force_load_balance_topk_ids(
                torch.zeros((9, 2), dtype=torch.int64),
                num_experts=8,
                ep_size=4,
                capturing=False,
            )

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.select_experts")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_forward_context")
    def test_apply_reuses_existing_force_balance_flag(self, mock_forward_context, mock_select_experts):
        method = AscendW4A8MXFPDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDynamicFusedMoEMethod)
        method.use_weight_packed = False
        method.dynamic_eplb = False
        method.ep_group = SimpleNamespace(world_size=4)

        topk_weights = torch.full((10, 2), 0.5, dtype=torch.float32)
        mock_select_experts.return_value = (topk_weights, torch.zeros((10, 2), dtype=torch.int64))
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.ones((10, 16), dtype=torch.bfloat16)
        mock_forward_context.return_value = SimpleNamespace(capturing=False, moe_comm_method=moe_comm_method)
        layer = SimpleNamespace(
            n_shared_experts=0,
            w13_weight=torch.empty((2, 1, 1)),
            w2_weight=torch.empty((2, 1, 1)),
            w13_weight_scale=torch.empty((2, 1, 1)),
            w2_weight_scale=torch.empty((2, 1, 1)),
            swiglu_limit=0.0,
        )

        method.apply(
            layer=layer,
            x=torch.randn((10, 16), dtype=torch.bfloat16),
            router_logits=torch.randn((10, 8), dtype=torch.bfloat16),
            top_k=2,
            renormalize=True,
            num_experts=8,
            enable_force_load_balance=True,
        )

        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        per_expert_load = torch.bincount(fused_input.topk_ids.flatten(), minlength=8).reshape(4, 2)
        self.assertTrue(torch.equal(per_expert_load, torch.tensor([[3, 2], [3, 2], [3, 2], [3, 2]])))
