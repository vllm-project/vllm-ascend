#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AWQConfig
from vllm_ascend.quantization.methods.w4a16_awq import (AscendW4A16AWQFusedMoEMethod,
                                                         AscendW4A16AWQLinearMethod,
                                                         _unpack_qzero_from_int32,
                                                         _unpack_weight_from_int32)


class TestAWQConfig(TestBase):
    """Test AWQConfig class."""

    def test_awq_config_init(self):
        """Test AWQConfig initialization with valid parameters."""
        config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            modules_to_not_convert=["lm_head"],
        )

        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        self.assertEqual(config.pack_factor, 8)

    def test_awq_config_invalid_weight_bits(self):
        """Test AWQConfig raises error for non-4-bit weight quantization."""
        with self.assertRaises(ValueError) as context:
            AWQConfig(weight_bits=8, group_size=128, zero_point=True)

        self.assertIn("only 4-bit weight quantization is supported", str(context.exception))

    def test_awq_config_from_config(self):
        """Test AWQConfig from_config method."""
        config_dict = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": ["lm_head"],
        }

        config = AWQConfig.from_config(config_dict)

        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)


class TestAscendW4A16AWQLinearMethod(TestBase):
    """Test AscendW4A16AWQLinearMethod class."""

    def setUp(self):
        super().setUp()
        self.quant_config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
        )
        self.quant_method = AscendW4A16AWQLinearMethod(self.quant_config)

    def test_init(self):
        """Test AscendW4A16AWQLinearMethod initialization."""
        self.assertEqual(self.quant_method.pack_factor, 8)
        self.assertEqual(self.quant_method.group_size, 128)

    def test_process_weights_after_loading(self):
        """Test process_weights_after_loading converts weights correctly."""
        layer = torch.nn.Module()
        hidden_size = 512
        out_features = 1024
        pack_factor = 8
        group_size = 128

        # Original vLLM AWQ format weights
        num_groups = hidden_size // group_size
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 100, (hidden_size, out_features // pack_factor), dtype=torch.int32),
            requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.randint(0, 100, (num_groups, out_features // pack_factor), dtype=torch.int32),
            requires_grad=False
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((num_groups, out_features), dtype=torch.bfloat16),
            requires_grad=False
        )

        # Process weights
        self.quant_method.process_weights_after_loading(layer)

        # Verify qweight shape is unchanged and contiguous
        self.assertEqual(layer.qweight.shape, (hidden_size, out_features // pack_factor))
        self.assertTrue(layer.qweight.data.is_contiguous())

        # Verify qzeros is unpacked from (num_groups, out//pack) to (num_groups, out), bfloat16
        self.assertEqual(layer.qzeros.shape, (num_groups, out_features))
        self.assertEqual(layer.qzeros.dtype, torch.bfloat16)
        self.assertTrue(layer.qzeros.data.is_contiguous())

        # Verify parameters require no gradient
        self.assertFalse(layer.qweight.requires_grad)
        self.assertFalse(layer.scales.requires_grad)
        self.assertFalse(layer.qzeros.requires_grad)

    def _build_layer(self, hidden_size: int, out_features: int) -> torch.nn.Module:
        """Build a post-process_weights_after_loading mock linear layer."""
        group_size = self.quant_method.group_size
        pack_factor = self.quant_method.pack_factor
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 100, (hidden_size, out_features // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((hidden_size // group_size, out_features), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros((hidden_size // group_size, out_features), dtype=torch.bfloat16),
            requires_grad=False,
        )
        return layer

    @patch("vllm_ascend.quantization.methods.w4a16_awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply(self, mock_npu_matmul):
        """Test apply method calls npu_weight_quant_batchmatmul."""
        batch_size = 2
        seq_len = 8
        hidden_size = 512
        out_features = 1024

        mock_output = torch.randn(batch_size, seq_len, out_features, dtype=torch.float32)
        mock_npu_matmul.return_value = mock_output

        layer = self._build_layer(hidden_size, out_features)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

        result = self.quant_method.apply(layer, x)

        mock_npu_matmul.assert_called_once()
        self.assertEqual(result.shape, (batch_size, seq_len, out_features))

    @patch("vllm_ascend.quantization.methods.w4a16_awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_with_bias(self, mock_npu_matmul):
        """Test apply method handles bias correctly."""
        batch_size = 1
        seq_len = 1
        hidden_size = 256
        out_features = 512

        mock_output = torch.randn(batch_size, seq_len, out_features, dtype=torch.float32)
        mock_npu_matmul.return_value = mock_output

        layer = self._build_layer(hidden_size, out_features)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        bias = torch.randn(out_features, dtype=torch.bfloat16)

        # Call apply with bias
        result = self.quant_method.apply(layer, x, bias)

        # Verify result is returned and bias is converted to float
        self.assertIsNotNone(result)
        call_kwargs = mock_npu_matmul.call_args.kwargs
        self.assertEqual(call_kwargs["bias"].dtype, torch.float32)


class TestAscendW4A16AWQFusedMoEMethod(TestBase):
    """Test AscendW4A16AWQFusedMoEMethod class."""

    def setUp(self):
        super().setUp()
        self.quant_config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
        )
        self.quant_method = AscendW4A16AWQFusedMoEMethod(self.quant_config)

    def test_init(self):
        """Test AscendW4A16AWQFusedMoEMethod initialization."""
        self.assertEqual(self.quant_method.pack_factor, 8)
        self.assertEqual(self.quant_method.group_size, 128)

    def test_get_weight(self):
        """Test get_weight returns correctly shaped weight tensors."""
        num_experts = 4
        intermediate = 512
        hidden = 256

        result = self.quant_method.get_weight(num_experts, intermediate, hidden, torch.bfloat16)

        self.assertIn("w13_qweight", result)
        self.assertIn("w2_qweight", result)
        self.assertEqual(result["w13_qweight"].shape, (num_experts, hidden, 2 * intermediate // 8))
        self.assertEqual(result["w2_qweight"].shape, (num_experts, intermediate, hidden // 8))
        self.assertEqual(result["w13_qweight"].dtype, torch.int32)
        self.assertEqual(result["w2_qweight"].dtype, torch.int32)

    def test_get_dynamic_quant_param(self):
        """Test get_dynamic_quant_param returns correctly shaped scale/zero tensors."""
        num_experts = 4
        intermediate = 512
        hidden = 256
        group_size = 128

        result = self.quant_method.get_dynamic_quant_param(num_experts, intermediate, hidden, torch.bfloat16)

        num_groups_w13 = hidden // group_size
        num_groups_w2 = intermediate // group_size

        self.assertEqual(result["w13_scales"].shape, (num_experts, num_groups_w13, intermediate * 2))
        self.assertEqual(result["w2_scales"].shape, (num_experts, num_groups_w2, hidden))
        self.assertEqual(result["w13_qzeros"].shape, (num_experts, num_groups_w13, 2 * intermediate // 8))
        self.assertEqual(result["w2_qzeros"].shape, (num_experts, num_groups_w2, hidden // 8))
        self.assertEqual(result["w13_qzeros"].dtype, torch.int32)
        self.assertEqual(result["w2_qzeros"].dtype, torch.int32)


class TestUnpackQzeroFromInt32(TestBase):
    """Test unpack_qzero_from_int32 function for AWQ zero-points."""

    def test_unpack_qzero_from_int32_linear_layer(self):
        """Test unpacking zero-points for linear layer."""
        weight = torch.tensor([[305419896, -1420531520]], dtype=torch.int32)
        param_dtype = torch.bfloat16

        result = _unpack_qzero_from_int32(weight, param_dtype, pack_factor=8, is_moe_layer=False)

        # (1, 2) packed → (1, 16) unpacked (2 elements × 8 nibbles each)
        self.assertEqual(result.shape, (1, 16))
        self.assertEqual(result.dtype, param_dtype)
        self.assertTrue(result.is_contiguous())

    def test_unpack_qzero_from_int32_moe_layer(self):
        """Test unpacking zero-points for MoE layer."""
        weight = torch.tensor([[[305419896, -1420531520]]], dtype=torch.int32)
        param_dtype = torch.bfloat16

        result = _unpack_qzero_from_int32(weight, param_dtype, pack_factor=8, is_moe_layer=True)

        # (1, 1, 2) packed → (1, 1, 16) unpacked (2 elements × 8 nibbles each)
        self.assertEqual(result.shape, (1, 1, 16))
        self.assertEqual(result.dtype, param_dtype)
        self.assertTrue(result.is_contiguous())

    def test_unpack_qzero_from_int32_unsigned_to_signed(self):
        """Test unsigned int4 [0,15] to signed int4 [-8,7] conversion."""
        weight = torch.tensor([[0, 1, 7, 8, 9, 10, 15, 0]], dtype=torch.int32)
        param_dtype = torch.bfloat16

        result = _unpack_qzero_from_int32(weight, param_dtype, pack_factor=8, is_moe_layer=False)

        # Each int32 element unpacks to 8 nibbles; element k's lowest nibble lands at index k*8.
        self.assertEqual(result[0, 0].item(), 8)    # element 0: 0 -> -(0-8) = 8
        self.assertEqual(result[0, 8].item(), 7)    # element 1: 1 -> -(1-8) = 7
        self.assertEqual(result[0, 24].item(), 0)   # element 3: 8 -> -(8-8) = 0 (zero point)
        self.assertEqual(result[0, 48].item(), -7)  # element 6: 15 -> -(15-8) = -7


class TestUnpackWeightFromInt32(TestBase):
    """Test unpack_weight_from_int32 function for AWQ weights."""

    def test_unpack_weight_from_int32_basic(self):
        """Test unpacking weights with XOR transformation."""
        weight = torch.tensor([[305419896, -1420531520]], dtype=torch.int32)

        result = _unpack_weight_from_int32(weight, pack_factor=8)

        # Output shape is unchanged — repacking stays within the same int32 layout
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(result.is_contiguous())

    def test_unpack_weight_from_int32_xor_transformation(self):
        """Test XOR 0x88888888 transformation is applied."""
        weight = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)

        result = _unpack_weight_from_int32(weight, pack_factor=8)

        # All-zero input → repack loop produces all-zero weight_tmp → XOR with
        # 0x88888888 makes every int32 element 0x88888888 = -2004318072 (signed int32).
        self.assertEqual(result[0, 0].item(), -2004318072)  # 0x88888888 as int32

    def test_unpack_weight_from_int32_contiguous(self):
        """Test output is contiguous."""
        weight = torch.randint(0, 100, (16, 8), dtype=torch.int32)

        result = _unpack_weight_from_int32(weight, pack_factor=8)

        self.assertTrue(result.is_contiguous())


if __name__ == "__main__":
    import unittest
    unittest.main()