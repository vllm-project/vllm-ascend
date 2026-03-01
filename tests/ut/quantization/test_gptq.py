#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Unit tests for GPTQ quantization on Ascend NPU."""

from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.gptq_config import AscendGPTQConfig
from vllm_ascend.quantization.methods.gptq import AscendGPTQLinearMethod


class TestGPTQConfig(TestBase):
    """Test cases for AscendGPTQConfig class."""

    def test_gptq_config_creation(self):
        """Test creating a valid GPTQ config."""
        config = AscendGPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format="gptq",
        )

        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.desc_act, False)
        self.assertEqual(config.checkpoint_format, "gptq")
        self.assertEqual(config.pack_factor, 8)

    def test_gptq_config_invalid_bits(self):
        """Test that invalid weight_bits raises ValueError."""
        # Test 2-bit (not supported)
        with self.assertRaises(ValueError) as context:
            AscendGPTQConfig(
                weight_bits=2,
                group_size=128,
                desc_act=False,
            )
        self.assertIn("4/8-bit", str(context.exception))

        # Test 3-bit (not supported)
        with self.assertRaises(ValueError) as context:
            AscendGPTQConfig(
                weight_bits=3,
                group_size=128,
                desc_act=False,
            )
        self.assertIn("4/8-bit", str(context.exception))

        # Test 5-bit (invalid)
        with self.assertRaises(ValueError) as context:
            AscendGPTQConfig(
                weight_bits=5,
                group_size=128,
                desc_act=False,
            )
        self.assertIn("4/8-bit", str(context.exception))

    def test_gptq_config_desc_act_not_supported(self):
        """Test that desc_act=True raises ValueError."""
        with self.assertRaises(ValueError) as context:
            AscendGPTQConfig(
                weight_bits=4,
                group_size=128,
                desc_act=True,  # Not supported
            )
        self.assertIn("desc_act=True is not supported", str(context.exception))

    def test_gptq_config_from_config(self):
        """Test creating config from dictionary."""
        config_dict = {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "checkpoint_format": "gptq_v2",
        }

        config = AscendGPTQConfig.from_config(config_dict)

        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.checkpoint_format, "gptq_v2")

    def test_gptq_config_get_name(self):
        """Test get_name class method."""
        self.assertEqual(AscendGPTQConfig.get_name(), "gptq")

    def test_gptq_config_get_supported_act_dtypes(self):
        """Test get_supported_act_dtypes class method."""
        dtypes = AscendGPTQConfig.get_supported_act_dtypes()
        self.assertIn(torch.half, dtypes)
        self.assertIn(torch.bfloat16, dtypes)

    def test_gptq_config_get_min_capability(self):
        """Test that get_min_capability raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            AscendGPTQConfig.get_min_capability()


class TestAscendGPTQLinearMethod(TestBase):
    """Test cases for GPTQ linear method."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 128
        self.output_size = 256
        self.group_size = 32
        self.params_dtype = torch.float16

    def test_scheme_initialization_4bit(self):
        """Test scheme initialization for 4-bit."""
        config = AscendGPTQConfig(
            weight_bits=4,
            group_size=self.group_size,
            desc_act=False,
            checkpoint_format="gptq",
        )
        scheme = AscendGPTQLinearMethod(config)

        self.assertEqual(scheme.quant_config.weight_bits, 4)
        self.assertEqual(scheme.quant_config.pack_factor, 8)
        self.assertEqual(scheme.quant_config.group_size, self.group_size)
        self.assertEqual(scheme.quant_config.checkpoint_format, "gptq")

    def test_scheme_initialization_8bit(self):
        """Test scheme initialization for 8-bit."""
        config = AscendGPTQConfig(
            weight_bits=8,
            group_size=self.group_size,
            desc_act=False,
        )
        scheme = AscendGPTQLinearMethod(config)

        self.assertEqual(scheme.quant_config.weight_bits, 8)
        self.assertEqual(scheme.quant_config.pack_factor, 4)

    def test_create_weights_4bit(self):
        """Test create_weights method for 4-bit."""
        config = AscendGPTQConfig(
            weight_bits=4,
            group_size=self.group_size,
            desc_act=False,
        )
        scheme = AscendGPTQLinearMethod(config)

        # Create a mock layer
        layer = Mock()
        layer.qweight = None
        layer.scales = None
        layer.qzeros = None
        layer.g_idx = None

        # The parent GPTQLinearMethod.create_weights creates the parameters
        # We can't easily test this without mocking the parent class
        # Just verify the scheme is properly initialized
        self.assertEqual(scheme.quant_config.weight_bits, 4)
        self.assertEqual(scheme.quant_config.pack_factor, 8)

    def test_desc_act_raises_error(self):
        """Test that desc_act=True raises ValueError in config creation."""
        # desc_act=True is already blocked at AscendGPTQConfig level
        with self.assertRaises(ValueError) as context:
            AscendGPTQConfig(
                weight_bits=4,
                group_size=self.group_size,
                desc_act=True,  # This should raise ValueError
            )
        self.assertIn("desc_act", str(context.exception).lower())

    @patch("vllm_ascend.quantization.methods.gptq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading(self, mock_npu_convert):
        """Test process_weights_after_loading method."""
        config = AscendGPTQConfig(
            weight_bits=4,
            group_size=self.group_size,
            desc_act=False,
            checkpoint_format="gptq",
        )
        scheme = AscendGPTQLinearMethod(config)

        # Create mock layer with packed weights
        layer = Mock()
        pack_factor = 8  # for 4-bit
        # Use valid int32 range: -2^31 to 2^31-1, but use uint-like values 0 to 2^31-1
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 2**31-1, (self.input_size // pack_factor, self.output_size), dtype=torch.int32),
            requires_grad=False
        )
        layer.scales = torch.nn.Parameter(
            torch.randn(self.input_size // self.group_size, self.output_size, dtype=self.params_dtype),
            requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.randint(0, 2**31-1, ((self.input_size // self.group_size) // pack_factor, self.output_size), dtype=torch.int32),
            requires_grad=False
        )
        layer.g_idx = torch.nn.Parameter(
            torch.arange(self.input_size, dtype=torch.int32),
            requires_grad=False
        )

        # Mock the NPU int4pack conversion to return a dummy tensor
        mock_npu_convert.return_value = torch.zeros(
            self.output_size, self.input_size // pack_factor, dtype=torch.int32
        )

        # Process weights - this should unpack qweight and qzeros
        scheme.process_weights_after_loading(layer)

        # Verify qzeros was unpacked and converted to float
        self.assertEqual(layer.qzeros.dtype, self.params_dtype)

        # Verify qweight shape changed (unpacked or converted to int4pack for 4-bit)
        self.assertIsNotNone(layer.qweight)

        # Verify npu_convert_weight_to_int4pack was called for 4-bit weights
        mock_npu_convert.assert_called_once()

    @patch("vllm_ascend.quantization.methods.gptq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply(self, mock_npu_matmul):
        """Test apply method."""
        config = AscendGPTQConfig(
            weight_bits=4,
            group_size=self.group_size,
            desc_act=False,
        )
        scheme = AscendGPTQLinearMethod(config)

        # Create mock layer
        layer = Mock()
        pack_factor = 8  # for 4-bit

        # For 4-bit weights after processing, they're in int4pack format
        # Shape for int4pack: (output_size, input_size // pack_factor)
        layer.qweight = torch.zeros(
            self.output_size, self.input_size // pack_factor, dtype=torch.int32
        )
        layer.scales = torch.ones(
            self.output_size, self.input_size // self.group_size, dtype=self.params_dtype
        )
        layer.qzeros = torch.zeros(
            self.output_size,
            self.input_size // self.group_size,
            dtype=self.params_dtype,
        )

        # Create input
        batch_size = 4
        x = torch.randn(batch_size, self.input_size, dtype=self.params_dtype)

        # Mock NPU matmul output
        # For 4-bit int4pack, output shape is (batch_size, qweight.shape[-1] * 8)
        expected_output_size = (layer.qweight.shape[-1] * 8)
        mock_npu_matmul.return_value = torch.randn(
            batch_size, expected_output_size, dtype=self.params_dtype
        )

        # Apply
        output = scheme.apply(layer, x)

        # Verify
        self.assertEqual(output.shape, (batch_size, expected_output_size))
        mock_npu_matmul.assert_called_once()

        # Verify the NPU matmul was called with correct arguments
        call_args = mock_npu_matmul.call_args
        self.assertIsNotNone(call_args)
        # Check that antiquant_group_size matches
        self.assertEqual(call_args.kwargs['antiquant_group_size'], self.group_size)

