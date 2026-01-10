from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a16 import (AscendW4A16FusedMoEMethod,
                                            get_quant_description,
                                            pack_to_int32, unpack_from_int32)


class TestGetQuantDescription(TestBase):
    group_size = 32
    symmetric = True

    def setUp(self):
        self.mock_vllm_config = Mock()
        self.mock_weight_quant = Mock(
            group_size=self.group_size,
            symmetric=self.symmetric,
        )

    def test_get_quant_description_with_ascend_quant_method(self):
        quant_description = dict(
            quant_method="ascend",
            group_size=self.group_size,
            symmetric=self.symmetric,
        )
        self.mock_vllm_config.quant_config = Mock(
            quant_description=quant_description)

        result = get_quant_description(self.mock_vllm_config,
                                       self.mock_weight_quant)

        self.assertEqual(result, quant_description)

    def test_get_quant_description_with_compressed_tensors_quant_method(self):
        quant_description = dict(quant_method="compressed-tensors")
        self.mock_vllm_config.quant_config = Mock(
            quant_description=quant_description)

        result = get_quant_description(self.mock_vllm_config,
                                       self.mock_weight_quant)

        self.assertEqual(result, self.mock_weight_quant.__dict__)

    def test_get_quant_description_with_not_implemented_quant_method(self):
        quant_description = dict(quant_method="not-implemented")
        self.mock_vllm_config.quant_config = Mock(
            quant_description=quant_description)

        with self.assertRaises(NotImplementedError):
            get_quant_description(self.mock_vllm_config,
                                  self.mock_weight_quant)


class TestUnpackFromInt32(TestBase):

    def test_unpack_from_int32_packed_dim_1(self):
        weight = torch.tensor([[305419896, -1420531520]], dtype=torch.int32)
        shape = torch.Size([1, 8])
        num_bits = 4

        result = unpack_from_int32(weight, shape, num_bits, packed_dim=1)

        self.assertEqual(result.dtype, torch.int8)
        self.assertEqual(result.shape, shape)

    def test_unpack_from_int32_packed_dim_0(self):
        weight = torch.tensor([[305419896], [-1420531520]], dtype=torch.int32)
        shape = torch.Size([8, 1])
        num_bits = 4

        result = unpack_from_int32(weight, shape, num_bits, packed_dim=0)

        self.assertEqual(result.dtype, torch.int8)
        self.assertEqual(result.shape, shape)

    def test_unpack_from_int32_assertions(self):
        with self.assertRaises(AssertionError):
            weight = torch.tensor([[1, 2]], dtype=torch.int64)
            unpack_from_int32(weight, torch.Size([8, 1]), 4)

        with self.assertRaises(AssertionError):
            weight = torch.tensor([[1, 2]], dtype=torch.int32)
            unpack_from_int32(weight, torch.Size([8, 1]), 16)


class TestPackToInt32(TestBase):

    @patch(
        "vllm_ascend.quantization.w4a16.torch_npu.npu_convert_weight_to_int4pack"
    )
    def test_pack_to_int32_int8(self, mock_npu_convert_weight_to_int4pack):
        weight = torch.zeros((2, 8, 8), dtype=torch.int8)
        result = pack_to_int32(weight)

        mock_npu_convert_weight_to_int4pack.assert_not_called()
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.shape, torch.Size([2, 8, 2]))

    @patch(
        "vllm_ascend.quantization.w4a16.torch_npu.npu_convert_weight_to_int4pack"
    )
    def test_pack_to_int32_int32(self, mock_npu_convert_weight_to_int4pack):

        def mock_convert_weight(weight):
            return weight[..., :weight.shape[-1] // 8]

        mock_npu_convert_weight_to_int4pack.side_effect = mock_convert_weight
        weight = torch.zeros((2, 8, 8), dtype=torch.int32)
        result = pack_to_int32(weight)

        mock_npu_convert_weight_to_int4pack.assert_called_once()
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.shape, torch.Size([2, 8, 1]))

    def test_pack_to_int32_assertion_dim(self):
        with self.assertRaises(AssertionError):
            weight = torch.zeros((8, 8), dtype=torch.int8)
            pack_to_int32(weight)

    def test_pack_to_int32_assertion_dtype(self):
        with self.assertRaises(AssertionError):
            weight = torch.zeros((2, 8, 8), dtype=torch.float32)
            pack_to_int32(weight)

    def test_pack_to_int32_assertion_divisible(self):
        with self.assertRaises(AssertionError):
            weight = torch.zeros((2, 8, 7), dtype=torch.int32)
            pack_to_int32(weight)

        with self.assertRaises(AssertionError):
            weight = torch.zeros((2, 8, 7), dtype=torch.int8)
            pack_to_int32(weight)


class TestAscendW4A16FusedMoEMethod(TestBase):
    experts = 8
    input_size = 32
    output_size = 128
    group_size = 32

    @patch("vllm_ascend.quantization.w4a16.get_ascend_config")
    @patch("vllm_ascend.quantization.w4a16.get_current_vllm_config")
    def setUp(self, mock_get_current_vllm_config, mock_get_ascend_config):
        mock_ascend_config = Mock()
        mock_ascend_config.dynamic_eplb = False
        mock_ascend_config.expert_map_record_path = None
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(
            quant_description={
                "quant_method": "ascend",
                "group_size": self.group_size,
                "symmetric": True
            })
        mock_get_current_vllm_config.return_value = mock_vllm_config

        self.quant_method = AscendW4A16FusedMoEMethod()

    def test_init(self):
        self.assertTrue(self.quant_method.transpose_weight)
        self.assertEqual(self.quant_method.num_bits, 4)
        self.assertEqual(self.quant_method.pack_factor, 8)
        self.assertEqual(self.quant_method.group_size, self.group_size)
        self.assertEqual(self.quant_method.symmetric, True)
        self.assertFalse(self.quant_method.dynamic_eplb)

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(self.experts,
                                                  self.input_size,
                                                  self.output_size,
                                                  torch.bfloat16)

        self.assertEqual(param_dict["w13_weight_packed"].dtype, torch.int32)
        expected_w13_shape = (self.experts, 2 * self.input_size,
                              self.output_size //
                              self.quant_method.pack_factor)
        self.assertEqual(param_dict["w13_weight_packed"].shape,
                         expected_w13_shape)

        self.assertEqual(param_dict["w2_weight_packed"].dtype, torch.int32)
        expected_w2_shape = (self.experts, self.output_size,
                             self.input_size // self.quant_method.pack_factor)
        self.assertEqual(param_dict["w2_weight_packed"].shape,
                         expected_w2_shape)

    def test_get_dynamic_quant_param(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16)

        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        expected_w13_scale_shape = (self.experts, 2 * self.input_size,
                                    self.output_size // self.group_size)
        self.assertEqual(param_dict["w13_weight_scale"].shape,
                         expected_w13_scale_shape)

        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.bfloat16)
        expected_w2_scale_shape = (self.experts, self.output_size,
                                   self.input_size // self.group_size)
        self.assertEqual(param_dict["w2_weight_scale"].shape,
                         expected_w2_scale_shape)

        self.assertEqual(param_dict["w13_weight_shape"].dtype, torch.int32)
        self.assertEqual(param_dict["w13_weight_shape"].shape,
                         (self.experts, 2))

        self.assertEqual(param_dict["w2_weight_shape"].dtype, torch.int32)
        self.assertEqual(param_dict["w2_weight_shape"].shape,
                         (self.experts, 2))

        self.assertEqual(param_dict["w13_weight_offset"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_offset"].shape, (1, ))

        self.assertEqual(param_dict["w2_weight_offset"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_offset"].shape, (1, ))

    def build_layer(self):
        """Build a mock layer for testing"""
        layer = torch.nn.Module()

        w13_shape = (self.experts, 2 * self.input_size,
                     self.output_size // self.quant_method.pack_factor)
        w2_shape = (self.experts, self.output_size,
                    self.input_size // self.quant_method.pack_factor)

        layer.w13_weight_packed = torch.nn.Parameter(torch.randint(
            -100, 100, w13_shape, dtype=torch.int32),
                                                     requires_grad=False)
        layer.w2_weight_packed = torch.nn.Parameter(torch.randint(
            -100, 100, w2_shape, dtype=torch.int32),
                                                    requires_grad=False)

        w13_scale_shape = (self.experts, 2 * self.input_size,
                           self.output_size // self.group_size)
        w2_scale_shape = (self.experts, self.output_size,
                          self.input_size // self.group_size)

        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            w13_scale_shape, dtype=torch.bfloat16),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            w2_scale_shape, dtype=torch.bfloat16),
                                                   requires_grad=False)

        layer.w13_weight_offset = torch.nn.Parameter(torch.zeros(
            1, dtype=torch.bfloat16),
                                                     requires_grad=False)
        layer.w2_weight_offset = torch.nn.Parameter(torch.zeros(
            1, dtype=torch.bfloat16),
                                                    requires_grad=False)

        layer.w13_weight_shape = torch.nn.Parameter(torch.tensor(
            [[2 * self.input_size, self.output_size]] * self.experts,
            dtype=torch.int32),
                                                    requires_grad=False)
        layer.w2_weight_shape = torch.nn.Parameter(torch.tensor(
            [[self.output_size, self.input_size]] * self.experts,
            dtype=torch.int32),
                                                   requires_grad=False)

        return layer

    @patch(
        "vllm_ascend.quantization.w4a16.torch_npu.npu_convert_weight_to_int4pack"
    )
    def test_process_weights_after_loading_with_transpose(
            self, mock_npu_convert_weight_to_int4pack):

        def mock_convert_weight(weight):
            new_shape = list(weight.shape)
            new_shape[-1] = new_shape[-1] // 8
            return torch.zeros(new_shape, dtype=torch.int32)

        mock_npu_convert_weight_to_int4pack.side_effect = mock_convert_weight

        layer = self.build_layer()
        self.quant_method.transpose_weight = True

        self.quant_method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight_packed.data.shape,
                         torch.Size([8, 128, 8]))
        self.assertEqual(layer.w2_weight_packed.data.shape,
                         torch.Size([8, 32, 16]))

        self.assertEqual(layer.w13_weight_scale.data.shape,
                         torch.Size([8, 4, 64]))
        self.assertEqual(layer.w2_weight_scale.data.shape,
                         torch.Size([8, 1, 128]))

        self.assertTrue(layer.w13_weight_scale.data.is_contiguous())
        self.assertTrue(layer.w2_weight_scale.data.is_contiguous())

    def test_process_weights_after_loading_without_transpose(self):
        layer = self.build_layer()
        self.quant_method.transpose_weight = False

        original_w13_data = layer.w13_weight_packed.data.clone()
        original_w2_data = layer.w2_weight_packed.data.clone()

        self.quant_method.process_weights_after_loading(layer)

        self.assertTrue(
            torch.equal(layer.w13_weight_packed.data, original_w13_data))
        self.assertTrue(
            torch.equal(layer.w2_weight_packed.data, original_w2_data))
