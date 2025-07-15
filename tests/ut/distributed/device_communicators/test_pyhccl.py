import os
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.device_communicators.pyhccl import \
    PyHcclCommunicator


class MockHcclLib:
    """Mock HCCLLibrary 的返回值"""
    def __init__(self, path):
        pass

    def hcclGetUniqueId(self):
        uid = MagicMock()
        uid.internal = list(range(128))  # 128 字节随意填充
        return uid

    def hcclCommInitRank(self, world_size, uid, rank):
        return f"fake_comm_{rank}"

    def hcclAllReduce(self, *args, **kw):
        pass


class MockUniqueId:
    def __init__(self):
        self.internal = [0] * 128


class StatelessProcessGroup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def broadcast_obj(self, unique_id, src):
        pass


class TestPyHcclCommunicator:
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_world_size_1_return_early(self):
        """单卡时直接 disabled，不调用任何 HCCL API"""
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 1),
            device="npu:0",
        )
        print("Hello")
        assert comm.disabled is True
        assert comm.available is False

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_load_hccl_fail(self):
        """HCCL 库不存在时 disabled=True"""
        with patch("vllm.distributed.pyhccl.HCCLLibrary",
                   side_effect=OSError("libhccl.so not found")):
            comm = PyHcclCommunicator(
                group=StatelessProcessGroup(0, 2),
                device="npu:0",
            )
            assert comm.disabled is True

    @patch("vllm.distributed.pyhccl.HCCLLibrary", MockHcclLib)
    @patch("vllm.distributed.pyhccl.hcclUniqueId", MockUniqueId)
    @patch("torch.npu.device")
    @patch("vllm.distributed.pyhccl.current_stream",
           return_value=MagicMock(npu_stream=5678))
    def test_stateless_group(self, *_):
        """使用 StatelessProcessGroup 的初始化路径"""
        group = StatelessProcessGroup(rank=3, world_size=4)
        group.broadcast_obj = MagicMock(side_effect=lambda obj, src: obj)

        comm = PyHcclCommunicator(group=group, device=3)

        assert comm.rank == 3
        assert comm.world_size == 4
        # broadcast_obj 被调用
        group.broadcast_obj.assert_called_once()

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"})
    @patch("vllm.distributed.pyhccl.HCCLLibrary", MockHcclLib)
    @patch("vllm.distributed.pyhccl.hcclUniqueId", MockUniqueId)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.broadcast")
    @patch("torch.npu.device")  # 上下文管理器
    @patch("vllm.distributed.pyhccl.current_stream",
           return_value=MagicMock(npu_stream=1234))
    def test_multi_gpu_pg_torch(
        self,
        mock_stream,
        mock_npu_ctx,
        mock_dist_broadcast,
        *_,
    ):
        """使用 PyTorch 官方 ProcessGroup 的初始化路径"""
        fake_pg = MagicMock()
        comm = PyHcclCommunicator(group=fake_pg, device="npu:1")

        # 断言属性
        assert comm.rank == 1
        assert comm.world_size == 2
        assert comm.device == torch.device("npu:1")
        assert comm.available is True
        assert comm.disabled is False

        # 校验 broadcast 被调用，且 src 是全局 rank 0
        mock_dist_broadcast.assert_called_once()
        args, kwargs = mock_dist_broadcast.call_args
        assert kwargs["src"] == 0

        # 校验 warm-up all_reduce 被调用
        mock_stream.assert_called()


if __name__ == "__main__":
    testHccl = TestPyHcclCommunicator()
    testHccl.test_world_size_1_return_early()
