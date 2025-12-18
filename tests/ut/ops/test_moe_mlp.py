import unittest
from typing import ClassVar

import torch

from vllm_ascend.ops.fused_moe.moe_mlp import cumsum_group_list


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]])
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupport_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type],
                                           src_list_type,
                                           dst_list_type,
                                           expert_num=4)
                self.assertTrue(
                    torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received",
                      str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(
            self):
        for src_list_type, dst_list_type in self.unsupport_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type,
                                      dst_list_type)
                self.assertIn("This feature is under development.",
                              str(excinfo.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
