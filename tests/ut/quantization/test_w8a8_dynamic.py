from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w8a8_dynamic import (
    fused_experts, fused_experts_with_all2all, fused_experts_with_allgather,
    fused_experts_with_mc2)


class TestAscendW8A8FusedMoEMethod(TestBase):

    def setUp(self):
        self.hidden_size = 128
        self.num_tokens = 128
        self.top_k = 8
        self.placeholder = torch.randn(self.num_tokens,
                                       self.hidden_size,
                                       dtype=torch.bfloat16)

    @patch("torch_npu.npu_grouped_matmul_finalize_routing")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch.zeros")
    @patch("torch_npu.npu_moe_init_routing_v2")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ep_group")
    def test_fused_experts_with_allgather(
        self,
        mock_get_ep_group,
        mock_get_rank,
        mock_get_world_size,
        mock_dynamic_quant,
        mock_moe_init_routing_v2,
        mock_zeros,
        mock_grouped_matmul_swiglu_quant,
        mock_grouped_matmul_finalize_routing,
    ):
        placeholder_int8 = torch.randint(0,
                                         100,
                                         (self.num_tokens, self.hidden_size),
                                         dtype=torch.int8)
        placeholder_ones = torch.ones(self.num_tokens, dtype=torch.int32)

        expert_map = MagicMock()
        ep_group = MagicMock()
        ep_group.device_group = "ep_group"

        mock_get_ep_group.return_value = ep_group
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1
        mock_dynamic_quant.return_value = (
            placeholder_int8,
            placeholder_ones,
        )
        mock_moe_init_routing_v2.return_value = (
            placeholder_int8,
            placeholder_ones,
            placeholder_ones,
            self.placeholder,
        )
        mock_zeros.return_value = torch.zeros(
            (self.num_tokens, self.hidden_size), dtype=torch.bfloat16)
        mock_grouped_matmul_swiglu_quant.return_value = (
            placeholder_int8,
            self.placeholder,
            self.placeholder,
        )
        mock_grouped_matmul_finalize_routing.return_value = self.placeholder

        result = fused_experts_with_allgather(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w1_scale=self.placeholder,
            w2=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=self.top_k,
            expert_map=expert_map,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (128, 128))

    @patch("torch_npu.npu_moe_distribute_combine_v2", create=True)
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch_npu.npu_moe_distribute_dispatch_v2", create=True)
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ascend_soc_version")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_mc2_group")
    def test_fused_experts_with_mc2(
        self,
        mock_get_mc2_group,
        mock_get_ascend_soc_version,
        mock_dispatch_v2,
        mock_grouped_matmul_swiglu_quant,
        mock_grouped_matmul,
        mock_combine_v2,
    ):
        placeholder_int8 = torch.randint(0,
                                         100,
                                         (self.num_tokens, self.hidden_size),
                                         dtype=torch.int8)
        placeholder_ones = torch.ones(self.num_tokens, dtype=torch.int32)

        expert_map = MagicMock()
        ep_group = MagicMock()
        ep_group.rank_in_group = 0
        ep_group.world_size = 1
        mock_get_mc2_group.return_value = ep_group
        mock_get_ascend_soc_version.return_value = MagicMock()
        mock_dispatch_v2.return_value = (
            self.placeholder,
            self.placeholder,
            self.placeholder,
            placeholder_ones,
            placeholder_ones,
        )
        mock_grouped_matmul_swiglu_quant.return_value = (
            placeholder_int8,
            self.placeholder,
            self.placeholder,
        )
        mock_grouped_matmul.return_value = self.placeholder
        mock_combine_v2.return_value = self.placeholder

        result = fused_experts_with_mc2(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w1_scale=self.placeholder,
            w2=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=self.top_k,
            expert_map=expert_map,
            moe_all_to_all_group_name="group",
            log2phy=None,
            global_redundant_expert_num=256,
            mc2_mask=self.placeholder,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (128, 128))

    @patch("torch.distributed.all_to_all_single")
    @patch("torch_npu.npu_moe_re_routing")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_moe_finalize_routing")
    @patch("torch_npu.npu_moe_init_routing")
    def test_fused_experts_with_all2all(
            self, mock_moe_init_routing, mock_moe_finalize_routing,
            mock_dynamic_quant, mock_grouped_matmul_swiglu_quant,
            mock_grouped_matmul, mock_moe_re_routing, mock_all_to_all_single):
        expert_map = MagicMock()
        ep_group = MagicMock()
        placeholder_int8 = torch.randint(0,
                                         100,
                                         (self.num_tokens, self.hidden_size),
                                         dtype=torch.int8)
        placeholder_ones = torch.ones(self.num_tokens, dtype=torch.int32)
        mock_all_to_all_single.side_effect = lambda output, input, *args, **kwargs: output.copy_(
            input)
        mock_moe_init_routing.return_value = (
            placeholder_int8,
            placeholder_ones,
            placeholder_ones,
        )
        mock_moe_re_routing.return_value = (placeholder_int8, self.placeholder,
                                            torch.randint(0,
                                                          100,
                                                          (self.num_tokens, ),
                                                          dtype=torch.int32),
                                            self.placeholder)
        mock_grouped_matmul.return_value = self.placeholder
        mock_grouped_matmul_swiglu_quant.return_value = (
            placeholder_int8,
            self.placeholder,
            self.placeholder,
        )
        mock_dynamic_quant.return_value = (
            placeholder_int8,
            torch.randn(self.num_tokens),
        )
        mock_moe_finalize_routing.return_value = self.placeholder

        result = fused_experts_with_all2all(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w1_scale=self.placeholder,
            w2=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=self.top_k,
            expert_map=expert_map,
            ep_group=ep_group,
            log2phy=None,
            global_redundant_expert_num=256,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (128, 128))

    @patch("torch_npu.npu_moe_finalize_routing")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_moe_compute_expert_tokens")
    @patch("torch_npu.npu_moe_init_routing")
    def test_fused_experts(
        self,
        mock_moe_init_routing,
        mock_moe_compute_expert_tokens,
        mock_dynamic_quant,
        mock_grouped_matmul_swiglu_quant,
        mock_grouped_matmul,
        mock_moe_finalize_routing,
    ):
        placeholder_int8 = torch.randint(0,
                                         100,
                                         (self.num_tokens, self.hidden_size),
                                         dtype=torch.int8)
        placeholder_ones = torch.ones(self.num_tokens, dtype=torch.int32)

        mock_moe_init_routing.return_value = (
            placeholder_int8,
            placeholder_ones,
            placeholder_ones,
        )
        mock_moe_compute_expert_tokens.return_value = placeholder_ones
        mock_dynamic_quant.return_value = (
            placeholder_int8,
            torch.randn(self.num_tokens),
        )
        mock_grouped_matmul_swiglu_quant.return_value = (
            placeholder_int8,
            self.placeholder,
            self.placeholder,
        )
        mock_grouped_matmul_swiglu_quant.return_value = (
            placeholder_int8,
            self.placeholder,
            self.placeholder,
        )
        mock_moe_finalize_routing.return_value = self.placeholder

        result = fused_experts(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w1_scale=self.placeholder,
            w2=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=self.top_k,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (128, 128))
