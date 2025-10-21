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

from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch


class TestPatchDistributed(TestBase):

    def setUp(self):
        self.mock_group_ranks = [[0, 1]]
        self.mock_local_rank = 0
        self.mock_backend = "hccl"
        self.mock_use_device_comm = False

        patcher_get_rank = patch("torch.distributed.get_rank", return_value=0)
        patcher_new_group = patch("torch.distributed.new_group",
                                  return_value=MagicMock())
        patcher_is_cuda_alike = patch(
            "vllm.platforms.current_platform.is_cuda_alike", return_value=True)
        patcher_device_comm_cls = patch(
            "vllm.distributed.parallel_state.resolve_obj_by_qualname",
            return_value=MagicMock())
        patcher_calculate_dp_buffer = patch(
            "vllm_ascend.utils.calculate_dp_buffer_size", return_value=64)
        patcher_npu_current_device = patch("torch.npu.current_device",
                                           return_value=MagicMock())

        self.mock_get_rank = patcher_get_rank.start()
        self.mock_new_group = patcher_new_group.start()
        self.mock_is_cuda_alike = patcher_is_cuda_alike.start()
        self.mock_resolve_obj = patcher_device_comm_cls.start()
        self.mock_calculate_dp_buffer = patcher_calculate_dp_buffer.start()
        self.mock_npu_current_device = patcher_npu_current_device.start()

        self.addCleanup(patcher_get_rank.stop)
        self.addCleanup(patcher_new_group.stop)
        self.addCleanup(patcher_is_cuda_alike.stop)
        self.addCleanup(patcher_device_comm_cls.stop)
        self.addCleanup(patcher_calculate_dp_buffer.stop)
        self.addCleanup(patcher_npu_current_device.stop)

        self.group_coordinator = GroupCoordinatorPatch(
            group_ranks=self.mock_group_ranks,
            local_rank=self.mock_local_rank,
            torch_distributed_backend=self.mock_backend,
            use_device_communicator=self.mock_use_device_comm)

    def test_GroupCoordinator_patched(self):
        self.assertIs(GroupCoordinator, GroupCoordinatorPatch)

    def test_all_to_all_returns_input_when_world_size_1(self):
        self.group_coordinator.world_size = 1
        input_tensor = torch.randn(2, 3)
        output = self.group_coordinator.all_to_all(input_tensor)
        self.assertTrue(torch.equal(output, input_tensor))

    def test_all_to_all_raises_assertion_on_invalid_scatter_dim(self):
        input_tensor = torch.randn(2, 3)
        with self.assertRaises(AssertionError) as cm:
            self.group_coordinator.all_to_all(input_tensor, scatter_dim=2)
        self.assertIn("Invalid scatter dim", str(cm.exception))

    def test_all_to_all_raises_assertion_on_invalid_gather_dim(self):
        input_tensor = torch.randn(2, 3)
        with self.assertRaises(AssertionError) as cm:
            self.group_coordinator.all_to_all(input_tensor, gather_dim=2)
        self.assertIn("Invalid gather dim", str(cm.exception))

    def test_all_to_all_calls_device_communicator_with_correct_args(self):
        mock_communicator = MagicMock()
        self.group_coordinator.device_communicator = mock_communicator

        input_tensor = torch.randn(2, 3)
        scatter_dim = 0
        gather_dim = 1
        scatter_sizes = [1, 1]
        gather_sizes = [1, 1]

        self.group_coordinator.all_to_all(input_tensor,
                                          scatter_dim=scatter_dim,
                                          gather_dim=gather_dim,
                                          scatter_sizes=scatter_sizes,
                                          gather_sizes=gather_sizes)

        mock_communicator.all_to_all.assert_called_once_with(
            input_tensor, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

    def test_all_to_all_calls_device_communicator_without_sizes(self):
        mock_communicator = MagicMock()
        self.group_coordinator.device_communicator = mock_communicator

        input_tensor = torch.randn(2, 3)
        scatter_dim = 0
        gather_dim = 1

        self.group_coordinator.all_to_all(input_tensor,
                                          scatter_dim=scatter_dim,
                                          gather_dim=gather_dim)

        mock_communicator.all_to_all.assert_called_once_with(
            input_tensor, scatter_dim, gather_dim, None, None)
