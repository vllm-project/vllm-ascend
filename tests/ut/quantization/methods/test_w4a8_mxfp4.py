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
            dp_size=1,
            dp_rank=0,
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
                dp_size=1,
                dp_rank=0,
                capturing=True,
            )

    def test_requires_exact_global_rank_divisibility(self):
        with self.assertRaisesRegex(ValueError, "Global routed rows .* divisible by EP size"):
            _force_load_balance_topk_ids(
                torch.zeros((9, 2), dtype=torch.int64),
                num_experts=8,
                ep_size=4,
                dp_size=1,
                dp_rank=0,
                capturing=False,
            )

    def test_uses_dp_rank_offsets_for_exact_ep32_balance(self):
        per_dp_ids = [
            _force_load_balance_topk_ids(
                torch.zeros((8, 2), dtype=torch.int64),
                num_experts=896,
                ep_size=32,
                dp_size=4,
                dp_rank=dp_rank,
                capturing=False,
            )
            for dp_rank in range(4)
        ]

        global_load = torch.bincount(torch.cat(per_dp_ids).flatten(), minlength=896).reshape(32, 28)
        expected = torch.zeros((32, 28), dtype=torch.int64)
        expected[:, :2] = 1
        self.assertTrue(torch.equal(global_load, expected))

    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.select_experts")
    @patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_forward_context")
    def test_apply_reuses_existing_force_balance_flag(self, mock_forward_context, mock_select_experts):
        method = AscendW4A8MXFPDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDynamicFusedMoEMethod)
        method.use_weight_packed = False
        method.dynamic_eplb = False
        method.ep_group = SimpleNamespace(world_size=4)
        method.dp_group = SimpleNamespace(world_size=1, rank_in_group=0)

        topk_weights = torch.full((10, 2), 0.5, dtype=torch.float32)
        mc2_mask = torch.tensor([True, False, True, False, True, False, True, False, True, False])
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
            mc2_mask=mc2_mask,
        )

        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        per_expert_load = torch.bincount(fused_input.topk_ids.flatten(), minlength=8).reshape(4, 2)
        self.assertTrue(torch.equal(per_expert_load, torch.tensor([[3, 2], [3, 2], [3, 2], [3, 2]])))
        self.assertTrue(torch.equal(fused_input.routing.mc2_mask, torch.ones_like(mc2_mask)))
