import os
from unittest.mock import MagicMock, patch

from vllm.distributed.utils import StatelessProcessGroup

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl import \
    PyHcclCommunicator


class MockHcclLib:
    pass


class MockUniqueId:
    pass


class TestPyHcclCommunicator(TestBase):

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_world_size_1_return_early(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 1, None, None),
            device="npu:0",
        )
        self.assertTrue(comm.disabled)
        self.assertFalse(comm.available)

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_load_hccl_fail(self):
        comm = PyHcclCommunicator(group=StatelessProcessGroup(
            0, 2, None, None),
                                  device="npu:0",
                                  library_path="/not/exist/path/libhccl.so")
        self.assertTrue(comm.disabled)

    @patch(
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary",
        MockHcclLib)
    @patch(
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId",
        MockUniqueId)
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream",
           return_value=MagicMock(npu_stream=5678))
    def test_stateless_group(self, *_):
        group = StatelessProcessGroup(rank=3,
                                      world_size=4,
                                      store=None,
                                      socket=None)

        comm = PyHcclCommunicator(group=group, device=3)

        self.assertEqual(comm.rank, 3)
        self.assertEqual(comm.world_size, 4)

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"})
    @patch(
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary",
        MockHcclLib)
    @patch(
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId",
        MockUniqueId)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.broadcast")
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream",
           return_value=MagicMock(npu_stream=1234))
    def test_multi_gpu_pg_torch(
        self,
        *_,
    ):
        fake_pg = MagicMock()
        comm = PyHcclCommunicator(group=fake_pg, device="npu:1")

        self.assertEqual(comm.rank, 1)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)
