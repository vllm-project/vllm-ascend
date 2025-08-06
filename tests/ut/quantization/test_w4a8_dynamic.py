from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a8_dynamic import (
    AscendW4A8DynamicFusedMoEMethod, AscendW4A8DynamicLinearMethod)


class TestAscendW4A8DynamicLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendW4A8DynamicLinearMethod()
        self.method.group_size = 8

    def test_get_weight(self):
        weight = self.method.get_weight(8, 32, torch.bfloat16)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (32, 8))

    def test_get_pergroup_param(self):
        params = self.method.get_pergroup_param(8, 32, torch.bfloat16)
        self.assertEqual(params["weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_scale"].shape, (32, 1))
        self.assertEqual(params["weight_offset"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_offset"].shape, (32, 1))
        self.assertEqual(params["weight_scale_second"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_scale_second"].shape, (32, 1))
        self.assertEqual(params["weight_offset_second"].dtype, torch.bfloat16)
        self.assertEqual(params["weight_offset_second"].shape, (32, 1))


class TestAscendW4A8DynamicFusedMoEMethod(TestBase):

    @patch('vllm_ascend.quantization.w4a8_dynamic.get_ep_group')
    @patch("vllm_ascend.ascend_config.get_ascend_config")
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_mc2_group')
    @patch('torch.distributed.get_rank', return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, mock_get_ascend_config,
              mock_get_ep_group):
        mock_ascend_config = Mock()
        mock_ascend_config.torchair_graph_config = Mock(enabled=False)
        mock_get_ascend_config.return_value = mock_ascend_config
        self.quant_method = AscendW4A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(8, 4, 14, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape, (8, 8, 14))

    @patch('vllm_ascend.quantization.w4a8_dynamic.get_current_vllm_config')
    def test_get_dynamic_quant_param(self, mock_get_current_vllm_config):
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(
            quant_description={"group_size": 2})
        mock_get_current_vllm_config.return_value = mock_vllm_config
        param_dict = self.quant_method.get_dynamic_quant_param(
            8, 4, 14, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (8, 8, 1))
        self.assertEqual(param_dict["w13_weight_scale_second"].dtype,
                         torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale_second"].shape,
                         (8, 8, 7))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (8, 14, 1))
        self.assertEqual(param_dict["w2_weight_scale_second"].dtype,
                         torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_scale_second"].shape,
                         (8, 14, 2))

    @patch('torch_npu.npu_quantize')
    @patch('torch.Tensor.npu')
    def test_process_weights_after_loading(self, mock_npu, mock_npu_quantize):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.zeros((8, 8, 14),
                                                          dtype=torch.int8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.zeros((8, 14, 4),
                                                         dtype=torch.int8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            (8, 8, 1), dtype=torch.bfloat16),
                                                    requires_grad=False)
        layer.w13_weight_offset = torch.nn.Parameter(torch.zeros(
            (8, 8, 1), dtype=torch.bfloat16),
                                                     requires_grad=False)
        layer.w13_weight_scale_second = torch.nn.Parameter(torch.ones(
            (8, 8, 7), dtype=torch.bfloat16),
                                                           requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            (8, 14, 1), dtype=torch.bfloat16),
                                                   requires_grad=False)
        layer.w2_weight_offset = torch.nn.Parameter(torch.zeros(
            (8, 14, 1), dtype=torch.bfloat16),
                                                    requires_grad=False)
        layer.w2_weight_scale_second = torch.nn.Parameter(torch.ones(
            (8, 14, 2), dtype=torch.bfloat16),
                                                          requires_grad=False)

        mock_npu.return_value = torch.Tensor()
        mock_npu_quantize.return_value = torch.Tensor()
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertEqual(layer.w13_scale_bias.data.shape, (8, 8))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertTrue(hasattr(layer, "w2_scale_bias"))
        self.assertEqual(layer.w2_scale_bias.data.shape, (8, 14))
        self.assertEqual(layer.w2_scale_bias.data.dtype, torch.float32)
