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
