import unittest
from typing import ClassVar
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401 -- registers torch.npu used by the module under test

from vllm_ascend.ops.fused_moe.moe_utils import (
    _custom_gmm_swiglu_enabled,
    _prepare_dequant_swiglu_weight_scale,
    cumsum_group_list,
)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestFusionFlags(unittest.TestCase):
    def test_custom_gmm_swiglu_requires_fusion_dynamic_eplb(self):
        self.assertFalse(_custom_gmm_swiglu_enabled(False, True))
        self.assertFalse(_custom_gmm_swiglu_enabled(True, False))
        with patch("vllm_ascend.ops.fused_moe.moe_utils.enable_custom_op", return_value=True):
            self.assertTrue(_custom_gmm_swiglu_enabled(True, True, activation="silu"))


class TestSwigluScaleHelpers(unittest.TestCase):
    def test_prepare_dequant_swiglu_weight_scale_stacks_and_casts(self):
        scales = [torch.randn(4, dtype=torch.float16) for _ in range(2)]
        out = _prepare_dequant_swiglu_weight_scale(scales, True)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.dim(), 2)

        single = torch.randn(4, dtype=torch.float16)
        out_single = _prepare_dequant_swiglu_weight_scale([single], True)
        self.assertEqual(out_single.dtype, torch.float32)
        self.assertEqual(out_single.shape, (1, 4))

    def test_prepare_dequant_swiglu_weight_scale_keeps_flat_for_non_swigluoai(self):
        single = torch.randn(4, dtype=torch.float16)
        out = _prepare_dequant_swiglu_weight_scale([single], False)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (4,))


if __name__ == "__main__":
    unittest.main()
