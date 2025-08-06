import unittest
from unittest.mock import MagicMock, Mock, patch

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
    def test_dispatch(self, *_):
        comm = NPUCommunicator(cpu_group=dist.group.WORLD)
        comm.all2all_manager = Mock()
        hidden_states = torch.randn(2, 4, 8)
        router_logits = torch.randn(2, 4, 2)

        mock_dispatch_result = (torch.randn(2, 4, 8), torch.randn(2, 4, 2))
        comm.all2all_manager.dispatch.return_value = mock_dispatch_result

        result_hidden, result_logits = comm.dispatch(hidden_states,
                                                     router_logits)

        assert torch.allclose(result_hidden, mock_dispatch_result[0])
        assert torch.allclose(result_logits, mock_dispatch_result[1])

        comm.all2all_manager.dispatch.assert_called_once_with(
            hidden_states, router_logits)

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
    def test_combine(self, *_):
        comm = NPUCommunicator(cpu_group=dist.group.WORLD)
        comm.all2all_manager = Mock()
        hidden_states = torch.randn(2, 4, 8)

        mock_combine_result = torch.randn(2, 4, 8)
        comm.all2all_manager.combine.return_value = mock_combine_result

        result = comm.combine(hidden_states)

        assert torch.allclose(result, mock_combine_result)

        comm.all2all_manager.combine.assert_called_once_with(hidden_states)
