import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                    AscendMergedColumnParallelLinear,
                                    AscendRowParallelLinear)


class BaseLinearTest(unittest.TestCase):

    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.oproj_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.linear.get_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True)
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendRowParallelLinear(BaseLinearTest):

    def test_mlp_optimize(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.comm_group, parallel_state._MLP_TP)
        self.assertEqual(linear.forward_type, "mlp_tp")

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    def test_oproj_tp(self):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.oproj_tensor_parallel_size = 2

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.comm_group, parallel_state._OTP)
        self.assertEqual(linear.forward_type, "oproj_tp")

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendColumnParallelLinear(BaseLinearTest):

    def test_mlp_tp_init(self):
        linear = AscendColumnParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.comm_group, parallel_state._MLP_TP)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):

    def test_merged_mlp_tp_init(self):
        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.comm_group, parallel_state._MLP_TP)
        self.assertEqual(linear.forward_type, "mlp_tp")


if __name__ == '__main__':
    unittest.main()
