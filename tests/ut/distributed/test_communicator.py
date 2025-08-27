import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from vllm_ascend.distributed.communicator import NPUCommunicator


class TestNPUCommunicator(unittest.TestCase):

    @patch("vllm.config.get_current_vllm_config", return_value=None)
    @patch("torch.npu.current_device", return_value=MagicMock())
    @patch("torch.npu.set_device", return_value=MagicMock())
    @patch("torch.distributed.get_process_group_ranks",
           return_value={
               0: 0,
               1: 1
           })
    @patch("torch.distributed.get_group_rank", return_value={0: 0, 1: 1})
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.npu.device")
    def test_all_to_all_with_sizes(self, *_):

        def patched_all_to_all(output_tensor_list,
                               input_tensor_list,
                               group=None,
                               async_op=False):
            output_tensor_list[:] = ([
                torch.tensor([10, 20]),
                torch.tensor([50, 60])
            ])

        torch.distributed.all_to_all = patched_all_to_all

        scatter_sizes = [2, 2]
        gather_sizes = [2, 2]
        input_ = torch.tensor([10, 20, 30, 40])

        comm = NPUCommunicator(cpu_group=dist.group.WORLD)

        output = comm.all_to_all(input_,
                                 scatter_sizes=scatter_sizes,
                                 gather_sizes=gather_sizes)

        assert output.tolist() == [10, 20, 50, 60]

    @patch("vllm.config.get_current_vllm_config", return_value=None)
    @patch("torch.npu.current_device", return_value=MagicMock())
    @patch("torch.npu.set_device", return_value=MagicMock())
    @patch("torch.distributed.get_process_group_ranks",
           return_value={
               0: 0,
               1: 1
           })
    @patch("torch.distributed.get_group_rank", return_value={0: 0, 1: 1})
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.npu.device")
    def test_all_to_all_without_sizes(self, *_):

        def patched_all_to_all(output_tensor_list,
                               input_tensor_list,
                               group=None,
                               async_op=False):
            output_tensor_list[:] = ([
                torch.tensor([[10, 20]]),
                torch.tensor([[50, 60]])
            ])

        torch.distributed.all_to_all = patched_all_to_all

        input_ = torch.tensor([[10, 20], [30, 40]])

        comm = NPUCommunicator(cpu_group=dist.group.WORLD)
        output = comm.all_to_all(input_, scatter_dim=0, gather_dim=0)

        assert output.tolist() == [[10, 20], [50, 60]]
