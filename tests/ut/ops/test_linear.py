import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                    AscendMergedColumnParallelLinear,
                                    AscendRowParallelLinear,
                                    AscendUnquantizedLinearMethod)


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


class TestAscendUnquantizedLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()

    @mock.patch("torch_npu.npu_format_cast")
    @mock.patch("torch.version")
    def test_process_weights_after_loading_is_cann_8_3(self, mock_version,
                                                       mock_format_cast):
        layer = mock.MagicMock()

        mock_version.cann = "8.3.RC1"
        self.method.process_weights_after_loading(layer)
        mock_format_cast.assert_called_once()

    @mock.patch("torch.version")
    def test_process_weights_after_loading_not_cann_8_3(self, mock_version):
        layer = mock.MagicMock()

        mock_version.cann = "8.2.RC1"
        # Should not raise exception
        self.method.process_weights_after_loading(layer)

    @mock.patch("torch.matmul")
    @mock.patch("torch.version")
    def test_apply_with_bias_is_cann_8_3(self, mock_version, mock_npu_matmul):
        layer = mock.MagicMock()
        layer.weight = torch.randn(128, 256)

        x = torch.randn(32, 128)
        bias = torch.randn(256)

        expected_y_output = torch.randn(32, 256)
        mock_npu_matmul.return_value = expected_y_output

        mock_version.cann = "8.3.RC1"
        output = self.method.apply(layer, x, bias)

        expected_y_output += bias
        self.assertTrue(torch.equal(output, expected_y_output))

    @mock.patch("torch.matmul")
    @mock.patch("torch.version")
    def test_apply_without_bias_is_cann_8_3(self, mock_version,
                                            mock_npu_matmul):
        layer = mock.MagicMock()
        layer.weight = torch.randn(128, 256)

        x = torch.randn(32, 128)

        expected_y_output = torch.randn(32, 256)
        mock_npu_matmul.return_value = expected_y_output

        mock_version.cann = "8.3.RC1"
        output = self.method.apply(layer, x)

        self.assertTrue(torch.equal(output, expected_y_output))

    @mock.patch("torch.nn.functional.linear")
    @mock.patch("torch.version")
    def test_apply_not_cann_8_3(self, mock_version, mock_npu_linear):
        layer = mock.MagicMock()
        layer.weight = torch.randn(128, 256)

        x = torch.randn(32, 128)

        expected_y_output = torch.randn(32, 256)
        mock_npu_linear.return_value = expected_y_output

        mock_version.cann = "8.2.RC1"
        output = self.method.apply(layer, x)

        self.assertTrue(torch.equal(output, expected_y_output))


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
