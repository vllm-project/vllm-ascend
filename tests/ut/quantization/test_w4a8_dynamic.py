from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a8_dynamic import AscendW4A8DynamicLinearMethod


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

    @patch("torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading(
            self, mock_npu_convert_weight_to_int4pack):
        layer = MagicMock()

        layer.weight.data = torch.randn(128, 256)
        layer.weight_scale.data = torch.randn(128, 1)
        layer.weight_scale_second.data = torch.randn(128, 16)
        layer.weight_offset.data = torch.randn(128, 1)

        mock_npu_convert_weight_to_int4pack.return_value = MagicMock
        self.method.process_weights_after_loading(layer)

        self.assertTrue(layer.weight_scale_second.data.dtype, torch.float32)
        self.assertEqual(layer.weight_scale_second.data.shape, (16, 128))
