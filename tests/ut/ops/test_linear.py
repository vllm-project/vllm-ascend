import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm import config, forward_context
from vllm.distributed import parallel_state as vllm_parallel_state

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendMergedColumnParallelLinear,
                                    AscendReplicatedLinear,
                                    AscendRowParallelLinear,
                                    AscendUnquantizedLinearMethod)


class BaseLinearTest(unittest.TestCase):

    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        self.mock_dp_group = mock.MagicMock()
        self.mock_dp_group.world_size = 4
        self.mock_dp_group.rank_in_group = 0

        self._forward_context = mock.MagicMock()
        forward_context._forward_context = self._forward_context
        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group
        vllm_parallel_state._DP = self.mock_dp_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.oproj_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group",
                  return_value=self.mock_group),
            patch("vllm.distributed.parallel_state.get_dp_group",
                  return_value=self.mock_dp_group),
            patch("vllm.forward_context.get_forward_context",
                  return_value=self._forward_context),
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
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype

    @mock.patch("vllm_ascend.ops.linear.is_enable_nz")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_enable_nz(self, mock_format_cast,
                                                     mock_is_nz):
        mock_is_nz.return_value = 1
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()

    @mock.patch("vllm_ascend.ops.linear.is_enable_nz")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_disable_nz(self, mock_format_cast,
                                                      mock_is_nz):
        mock_is_nz.return_value = 0
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()


class TestAscendRowParallelLinear(BaseLinearTest):

    def test_mlp_optimize(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    def test_oproj_tp(self):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        dp_batch_sizes = [3, 5, 7, 9]
        otp_groups = [[0, 1], [2, 3]]
        outputs = []

        forward_context._forward_context.dp_metadata.cu_tokens_across_dp_cpu = torch.tensor(
            [3, 8, 15, 24], device='cpu')

        for group in otp_groups:
            for dp_rank in group:
                with patch.object(self.mock_dp_group, "rank_in_group",
                                  dp_rank):
                    input_tensor = torch.randn(dp_batch_sizes[dp_rank], 16)
                    linear = AscendRowParallelLinear(
                        input_size=16,
                        output_size=8,
                        prefix="o_proj",
                    )
                    self.assertEqual(linear.custom_op.comm_group,
                                     parallel_state._OTP)
                    output = linear(input_tensor)
                    outputs.append(output[0])
                    self.assertEqual(output[0].shape[0],
                                     dp_batch_sizes[dp_rank])
                    self.assertEqual(output[0].shape[1], 8)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):

    def test_merged_mlp_tp_init(self):
        os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestAscendReplicatedLinear(BaseLinearTest):

    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


if __name__ == '__main__':
    unittest.main()
