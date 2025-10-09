from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a8_dynamic import (
    AscendW4A8DynamicFusedMoEMethod, AscendW4A8DynamicLinearMethod)


class TestAscendW4A8DynamicLinearMethod(TestBase):

    def setUp(self):
        with patch(
                'vllm_ascend.quantization.w4a8_dynamic.get_current_vllm_config'
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(
                quant_description={"group_size": 256})
            mock_vllm_config.scheduler_config = Mock(
                max_num_batched_tokens=2048,
                max_model_len=2048,
                enable_chunked_prefill=False)
            mock_get_current_vllm_config.return_value = mock_vllm_config
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
    experts = 8
    input_size = 16
    output_size = 56
    group_size = 2

    @patch('vllm_ascend.quantization.w4a8_dynamic.get_ascend_config')
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_current_vllm_config')
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_ep_group')
    @patch('vllm_ascend.quantization.w4a8_dynamic.get_mc2_group')
    @patch('torch.distributed.get_rank', return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, mock_get_ep_group,
              get_current_vllm_config, mock_get_ascend_config):
        # Mock ascend config
        mock_ascend_config = Mock()
        mock_ascend_config.dynamic_eplb = False
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={
            "group_size": self.group_size,
            "version": "0.0.0"
        })
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        mock_vllm_config.scheduler_config = Mock(max_num_batched_tokens=2048,
                                                 max_model_len=2048,
                                                 enable_chunked_prefill=False)
        get_current_vllm_config.return_value = mock_vllm_config
        self.quant_method = AscendW4A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        # old quant version w4a8 weight
        param_dict = self.quant_method.get_weight(self.experts,
                                                  self.input_size,
                                                  self.output_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape,
                         (self.experts, 2 * self.input_size, self.output_size))
        # new quant version weight
        self.quant_method.new_quant_version = True
        param_dict = self.quant_method.get_weight(self.experts,
                                                  self.input_size,
                                                  self.output_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape,
                         (self.experts, self.input_size, self.output_size))

    def test_get_dynamic_quant_param(self):
        # old quant version weight
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape,
                         (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w13_weight_scale_second"].dtype,
                         torch.float32)
        self.assertEqual(param_dict["w13_weight_scale_second"].shape,
                         (self.experts, 2 * self.input_size,
                          self.output_size // self.group_size))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape,
                         (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w2_weight_scale_second"].dtype,
                         torch.float32)
        self.assertEqual(param_dict["w2_weight_scale_second"].shape,
                         (self.experts, self.output_size,
                          self.input_size // self.group_size))
        # new quant version weight
        self.quant_method.new_quant_version = True
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w2_scale_bias"].dtype, torch.float32)
        self.assertEqual(
            param_dict["w2_scale_bias"].shape,
            (self.experts, self.output_size, 16 // self.quant_method.tp_size))
        # per-channel weight
        self.quant_method.is_per_channel_weight = True
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)
        pergroup_param = [
            "w13_weight_scale_second", "w13_weight_offset_second",
            "w2_weight_scale_second", "w2_weight_offset_second"
        ]
        is_contains = any(key in param_dict for key in pergroup_param)
        self.assertFalse(is_contains)

    def build_layer(self,
                    is_new_quant_version=True,
                    is_per_channel_weight=False):
        layer = torch.nn.Module()
        if is_new_quant_version:
            layer.w13_weight = torch.nn.Parameter(torch.zeros(
                (self.experts, self.input_size, self.output_size),
                dtype=torch.int8),
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(torch.zeros(
                (self.experts, self.output_size // 2, self.input_size),
                dtype=torch.int8),
                                                 requires_grad=False)
            w13_scale_bias = torch.zeros(
                (self.experts, 2 * self.input_size, 1), dtype=torch.float32)
            layer.w13_scale_bias = torch.nn.Parameter(w13_scale_bias,
                                                      requires_grad=False)
            w2_scale_bias = torch.zeros((self.experts, self.output_size,
                                         16 // self.quant_method.tp_size),
                                        dtype=torch.float32)
            layer.w2_scale_bias = torch.nn.Parameter(w2_scale_bias,
                                                     requires_grad=False)
        else:
            layer.w13_weight = torch.nn.Parameter(torch.zeros(
                (self.experts, 2 * self.input_size, self.output_size),
                dtype=torch.int8),
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(torch.zeros(
                (self.experts, self.output_size, self.input_size),
                dtype=torch.int8),
                                                 requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            (self.experts, 2 * self.input_size, 1), dtype=torch.float32),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            (self.experts, self.output_size, 1), dtype=torch.float32),
                                                   requires_grad=False)
        if not is_per_channel_weight:
            layer.w13_weight_scale_second = torch.nn.Parameter(
                torch.ones((self.experts, 2 * self.input_size,
                            self.output_size // self.group_size),
                           dtype=torch.float32),
                requires_grad=False)
            layer.w13_weight_offset_second = torch.nn.Parameter(
                torch.empty_like(layer.w13_weight_scale_second.data),
                requires_grad=False)
            layer.w2_weight_scale_second = torch.nn.Parameter(
                torch.ones((self.experts, self.output_size,
                            self.input_size // self.group_size),
                           dtype=torch.float32),
                requires_grad=False)
            layer.w2_weight_offset_second = torch.nn.Parameter(
                torch.empty_like(layer.w2_weight_scale_second.data),
                requires_grad=False)
        return layer

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu.npu_quantize')
    @patch('torch.Tensor.npu')
    def test_process_weights_after_loading(self, mock_npu, mock_npu_quantize,
                                           mock_npu_format_cast):
        mock_npu.return_value = torch.Tensor()
        mock_npu_quantize.return_value = torch.Tensor()

        def func_by_args(weight, num_format):
            return weight

        mock_npu_format_cast.side_effect = func_by_args
        # old quant version weight
        layer = self.build_layer(is_new_quant_version=False)
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertEqual(layer.w13_scale_bias.data.shape,
                         (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertTrue(hasattr(layer, "w2_scale_bias"))
        self.assertEqual(layer.w2_scale_bias.data.shape,
                         (self.experts, self.output_size))
        self.assertEqual(layer.w2_scale_bias.data.dtype, torch.float32)
        # new quant version weight
        self.quant_method.new_quant_version = True
        new_layer = self.build_layer(is_new_quant_version=True)
        self.quant_method.process_weights_after_loading(new_layer)
        self.assertEqual(new_layer.w13_scale_bias.data.shape,
                         (self.experts, 2 * self.input_size))
        self.assertEqual(new_layer.w2_scale_bias.data.shape,
                         (self.experts, self.output_size))
        self.assertFalse(hasattr(new_layer, "w13_weight_scale_second"))
        # per-channel weight
        self.quant_method.is_per_channel_weight = True
        per_channel_layer = self.build_layer(is_new_quant_version=True,
                                             is_per_channel_weight=True)
        self.quant_method.process_weights_after_loading(per_channel_layer)
        self.assertEqual(new_layer.w13_scale_bias.data.shape,
                         (self.experts, 2 * self.input_size))
