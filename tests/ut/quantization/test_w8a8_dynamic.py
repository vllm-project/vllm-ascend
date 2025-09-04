from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod


class TestAscendW8A8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ascend_config")
    @patch("vllm_ascend.quantization.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_get_ep_group, mock_get_ascend_config,
              mock_get_mc2_group, mock_get_rank):
        mock_ep_group = Mock()
        mock_get_ep_group.return_value = mock_ep_group
        mock_ascend_config = Mock()
        mock_ascend_config.torchair_graph_config = Mock(enabled=False)
        mock_get_ascend_config.return_value = mock_ascend_config
        mock_mc2_group = Mock(device_group=0)
        mock_get_mc2_group.return_value = mock_mc2_group
        mock_rank = Mock()
        mock_get_rank.return_value = mock_rank

        self.quant_method = AscendW8A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(self.num_experts,
                                                  self.intermediate_size,
                                                  self.hidden_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(
            param_dict["w13_weight"].shape,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size))

    def test_get_dynamic_quant_param(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size,
            torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].shape,
                         (self.num_experts, 2 * self.intermediate_size, 1))
