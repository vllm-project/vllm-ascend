from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.torchair.quantization.torchair_w8a8_dynamic import \
    torchair_fused_experts_with_all2all


class TestAscendW8A8FusedMoEMethod(TestBase):

    def setUp(self):
        self.hidden_size = 128
        self.num_tokens = 128
        self.placeholder = torch.randn(self.num_tokens,
                                       self.hidden_size,
                                       dtype=torch.bfloat16)

    @patch("torch.distributed.all_to_all_single")
    @patch("torch_npu.npu_moe_re_routing")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_moe_finalize_routing")
    @patch("torch_npu.npu_moe_init_routing")
    def test_torchair_fused_experts_with_all2all(
            self, mock_moe_init_routing, mock_moe_finalize_routing,
            mock_dynamic_quant, mock_swiglu, mock_grouped_matmul,
            mock_moe_re_routing, mock_all_to_all_single):

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
        mock_swiglu.return_value = self.placeholder
        mock_dynamic_quant.return_value = (
            placeholder_int8,
            torch.randn(self.num_tokens),
        )
        mock_moe_finalize_routing.return_value = self.placeholder

        result = torchair_fused_experts_with_all2all(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w1_scale=self.placeholder,
            w2=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=8,
            expert_map=expert_map,
            ep_group=ep_group,
            log2phy=None,
            global_redundant_expert_num=256,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (128, 128))
