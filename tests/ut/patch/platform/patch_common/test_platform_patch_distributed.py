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
import importlib
import os
from unittest.mock import MagicMock, patch

import torch
from torch.distributed import ProcessGroup, ReduceOp
from vllm.config import ParallelConfig

from tests.ut.base import TestBase


class TestPatchPlatformDistributed(TestBase):

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_HCCL_ALLREDUCE": "1"})
    @patch(
        "vllm.distributed.utils.stateless_init_torch_distributed_process_group"
    )
    def test_ascend_stateless_init_dp_group_called_when_optimized(
            self, mock_init_process_group):
        # We have to patch and reload because the patch will take effect
        # only after VLLM_ASCEND_ENABLE_HCCL_ALLREDUCE is set.
        import vllm_ascend.patch.platform.patch_common.patch_distributed
        importlib.reload(
            vllm_ascend.patch.platform.patch_common.patch_distributed)

        test_parallel_config = ParallelConfig()

        test_parallel_config.data_parallel_master_ip = "127.0.0.1"
        test_parallel_config.data_parallel_rank = 0
        test_parallel_config.data_parallel_size = 2

        mock_port = 12345
        test_parallel_config.get_next_dp_init_port = MagicMock(
            return_value=mock_port)

        mock_pg_instance = MagicMock(spec=ProcessGroup)
        mock_init_process_group.return_value = mock_pg_instance

        result = test_parallel_config.stateless_init_dp_group()

        self.assertIs(result, mock_pg_instance)

        mock_init_process_group.assert_called_once_with("127.0.0.1",
                                                        mock_port,
                                                        0,
                                                        2,
                                                        backend="hccl")

        test_parallel_config.get_next_dp_init_port.assert_called_once()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_HCCL_ALLREDUCE": "1"})
    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    def test_ascend_has_unfinished_dp_when_optimized2(self, mock_tensor,
                                                      mock_all_reduce):
        # We have to patch and reload because the patch will take effect
        # only after VLLM_ASCEND_ENABLE_HCCL_ALLREDUCE is set.
        import vllm_ascend.patch.platform.patch_common.patch_distributed
        importlib.reload(
            vllm_ascend.patch.platform.patch_common.patch_distributed)

        mock_tensor_instance = MagicMock()
        mock_tensor_instance.dtype = torch.int32
        mock_tensor_instance.device.type = "npu"
        mock_tensor_instance.item.return_value = 1
        mock_tensor.return_value = mock_tensor_instance

        test_parallel_config = ParallelConfig()
        mock_pg_instance = MagicMock(spec=ProcessGroup)
        mock_all_reduce.return_value = None

        result = test_parallel_config.has_unfinished_dp(mock_pg_instance,
                                                        has_unfinished=True)

        self.assertTrue(result)

        mock_tensor.assert_called_once_with([True],
                                            dtype=torch.int32,
                                            device="npu")

        mock_all_reduce.assert_called_once()
        args, kwargs = mock_all_reduce.call_args
        self.assertEqual(kwargs["op"], ReduceOp.MAX)
        self.assertEqual(kwargs["group"], mock_pg_instance)
