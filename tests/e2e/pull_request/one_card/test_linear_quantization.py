from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_linear_layer
from vllm_ascend.quantization.methods import (
    AscendW4A4FlatQuantDynamicLinearMethod,
    AscendW4A4LaosDynamicLinearMethod,
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A16LinearMethod,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


class TestW4A4FlatQuantDynamicWithNpu(TestBase):
    """
    Unit test suite for AscendW4A4FlatQuantDynamicLinearMethod and its helper functions.
    """

    def setUp(self):
        """Set up the test environment before each test."""
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()
        self.output_size = 64
        self.input_size = 768  # 768 = 24 * 32, divisible by 8
        self.params_dtype = torch.bfloat16

    def test_apply_with_npu(self):
        """Tests the apply method with NPU."""
        batch_size = 16
        layer = create_linear_layer(self.method, self.input_size, self.output_size, self.params_dtype)
        layer.clip_ratio = nn.Parameter(torch.tensor([0.95], dtype=torch.float32).npu(), requires_grad=False)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(batch_size, self.input_size, dtype=self.params_dtype).npu()
        output = self.method.apply(layer, x)

        self.assertEqual(output.shape, (batch_size, self.output_size))
        self.assertEqual(output.dtype, self.params_dtype)


class TestAscendW4A4LaosDynamicLinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW4A4LaosDynamicLinearMethod()

    def test_apply_with_npu(self):
        token_num = 32
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(token_num, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=params_dtype).npu()
        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (token_num, output_size))


class TestAscendW8A8LinearMethodWithNpu(TestBase):
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_static.get_current_vllm_config")
    def setUp(self, get_current_vllm_config):
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock()
        mock_vllm_config.quant_config.get_name.return_value = ASCEND_QUANTIZATION_METHOD
        get_current_vllm_config.return_value = mock_vllm_config
        self.method = AscendW8A8LinearMethod()
        self.mock_get_config = patch("vllm_ascend.utils.get_ascend_config")
        mock_config = self.mock_get_config.start()
        mock_ascend_config = MagicMock()
        mock_ascend_config.weight_nz_mode = 0
        mock_config.return_value = mock_ascend_config

    def tearDown(self):
        self.mock_get_config.stop()

    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        layer.params_dtype = params_dtype
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))


class TestAscendW8A8DynamicLinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod()
        self.mock_get_config = patch("vllm_ascend.utils.get_ascend_config")
        mock_config = self.mock_get_config.start()
        mock_ascend_config = MagicMock()
        mock_ascend_config.weight_nz_mode = 0
        mock_config.return_value = mock_ascend_config

    def tearDown(self):
        self.mock_get_config.stop()

    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))


class TestAscendW8A16LinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW8A16LinearMethod()
        self.mock_get_config = patch("vllm_ascend.utils.get_ascend_config")
        mock_config = self.mock_get_config.start()
        mock_ascend_config = MagicMock()
        mock_ascend_config.weight_nz_mode = 0
        mock_config.return_value = mock_ascend_config

    def tearDown(self):
        self.mock_get_config.stop()

    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))
