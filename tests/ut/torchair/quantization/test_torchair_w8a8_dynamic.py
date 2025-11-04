from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.torchair.quantization.torchair_w8a8_dynamic import (
    torchair_fused_experts_with_all2all, torchair_fused_experts_with_mc2)
from vllm_ascend.utils import AscendSocVersion


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
    @patch("torch_npu.npu_moe_init_routing_quant")
    def test_torchair_fused_experts_with_all2all(
            self, mock_npu_moe_init_routing_quant, mock_moe_finalize_routing,
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
        mock_npu_moe_init_routing_quant.return_value = (
            placeholder_int8, placeholder_ones, placeholder_ones,
            torch.bincount(placeholder_ones, minlength=len(expert_map)),
            torch.randn(self.num_tokens))
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

    @patch.dict('os.environ', {
        'HCCL_INTRA_ROCE_ENABLE': '0',
        'HCCL_INTRA_PCIE_ENABLE': '1'
    })
    @patch(
        "vllm_ascend.torchair.quantization.torchair_w8a8_dynamic.get_ascend_soc_version"
    )
    @patch(
        'vllm_ascend.torchair.quantization.torchair_w8a8_dynamic.get_mc2_group'
    )
    @patch('torch_npu.npu_moe_distribute_combine_v2')
    @patch('torch_npu.npu_moe_distribute_dispatch_v2')
    @patch(
        'vllm_ascend.torchair.quantization.torchair_w8a8_dynamic.torchair_apply_mlp_decode'
    )
    def test_torchair_fused_experts_with_mc2_a2_optimization(
            self, mock_mlp_decode, mock_dispatch, mock_combine, mock_get_group,
            mock_ascend_soc_version):
        """Test expert_scales is passed in A2 SOC version with mc2 optimization"""
        # Setup mocks
        mock_ascend_soc_version.return_value = AscendSocVersion.A2

        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 4
        mock_get_group.return_value = mock_group

        mock_combine.return_value = self.placeholder

        mock_dispatch.return_value = (torch.randn(32, 1024), torch.randn(1),
                                      torch.randint(0, 32, (32, )),
                                      torch.randint(1, 5, (8, )),
                                      torch.randint(1, 5, (4, )), None,
                                      torch.randn(32))
        mock_mlp_decode.return_value = self.placeholder

        result = torchair_fused_experts_with_mc2(
            hidden_states=self.placeholder,
            w1=self.placeholder,
            w2=self.placeholder,
            w1_scale=self.placeholder,
            w2_scale=self.placeholder,
            topk_weights=self.placeholder,
            topk_ids=self.placeholder,
            top_k=2,
            mc2_mask=self.placeholder)

        # Check that expert_scales was passed to dispatch
        call_args = mock_dispatch.call_args[1]
        self.assertIn('expert_scales', call_args)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, self.placeholder.shape)
