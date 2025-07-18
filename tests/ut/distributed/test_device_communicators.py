import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.utils import StatelessProcessGroup

from vllm_ascend.distributed.device_communicators.pyhccl import \
    PyHcclCommunicator


class TestPyHcclCommunicator(unittest.TestCase):

    class MockHcclUniqueId:

        def __init__(self, internal=None):
            self.internal = internal or [0] * 128

    class MockStatelessUniqueId:

        def __init__(self, internal=None):
            self.internal = internal or [0] * 128

        def __getstate__(self):
            return {'internal': self.internal}

        def __setstate__(self, state):
            self.__dict__.update(state)

    def setUp(self):
        self.mock_dist_patcher = patch('torch.distributed.is_initialized',
                                       return_value=True)
        self.mock_backend_patcher = patch('torch.distributed.get_backend',
                                          return_value='nccl')
        self.mock_rank_patcher = patch('torch.distributed.get_rank',
                                       return_value=0)
        self.mock_world_size_patcher = patch(
            'torch.distributed.get_world_size', return_value=2)
        self.mock_broadcast_patcher = patch('torch.distributed.broadcast',
                                            return_value=None)

        self.mock_is_initialized = self.mock_dist_patcher.start()
        self.mock_get_backend = self.mock_backend_patcher.start()
        self.mock_get_rank = self.mock_rank_patcher.start()
        self.mock_get_world_size = self.mock_world_size_patcher.start()
        self.mock_dist_broadcast = self.mock_broadcast_patcher.start()

        self.addCleanup(self.mock_dist_patcher.stop)
        self.addCleanup(self.mock_backend_patcher.stop)
        self.addCleanup(self.mock_rank_patcher.stop)
        self.addCleanup(self.mock_world_size_patcher.stop)
        self.addCleanup(self.mock_broadcast_patcher.stop)

        # Patch get_process_group_ranks
        self.patch_get_pgr = patch('torch.distributed.get_process_group_ranks',
                                   return_value={
                                       0: 0,
                                       1: 1
                                   })
        self.mock_get_pgr = self.patch_get_pgr.start()
        self.addCleanup(self.patch_get_pgr.stop)

        # Mock HCCLLibrary
        self.mock_hccl_patcher = patch(
            'vllm_ascend.distributed.device_communicators.pyhccl.HCCLLibrary')
        self.mock_hccl_lib = self.mock_hccl_patcher.start()
        self.addCleanup(self.mock_hccl_patcher.stop)

        self.hccl_instance = MagicMock()
        self.hccl_instance.hcclGetUniqueId.return_value = self.MockHcclUniqueId(
        )
        self.hccl_instance.hcclCommInitRank.return_value = MagicMock()
        self.hccl_instance.hcclAllReduce.return_value = None
        self.hccl_instance.hcclBroadcast.return_value = None
        self.mock_hccl_lib.return_value = self.hccl_instance

        # Mock current_stream
        self.mock_stream_patcher = patch('vllm_ascend.utils.current_stream')
        self.mock_stream = self.mock_stream_patcher.start()
        self.addCleanup(self.mock_stream_patcher.stop)

        self.mock_npu_device_patcher = patch('torch.npu.device',
                                             lambda x: MagicMock())
        self.mock_npu_device = self.mock_npu_device_patcher.start()
        self.addCleanup(self.mock_npu_device_patcher.stop)

        self.mock_npu_current_stream_patcher = patch(
            'torch.npu.current_stream',
            lambda device=None: MagicMock(npu_stream=None))
        self.mock_npu_current_stream = self.mock_npu_current_stream_patcher.start(
        )
        self.addCleanup(self.mock_npu_current_stream_patcher.stop)

    def test_init_with_process_group(self):
        group = MagicMock()
        comm = PyHcclCommunicator(group, device="cpu")
        self.assertEqual(comm.rank, 0)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.disabled)
        self.hccl_instance.hcclCommInitRank.assert_called_once()

    def test_all_reduce_disabled(self):
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        comm.disabled = True
        result = comm.all_reduce(torch.rand(1))
        self.assertIsNone(result)

    def test_all_reduce_device_mismatch(self):
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        tensor = torch.rand(1)
        result = comm.all_reduce(tensor)
        self.assertIsNotNone(result)

    def test_all_reduce_normal(self):
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        tensor = torch.rand(1, device="cpu")
        result = comm.all_reduce(tensor)
        self.assertIsNotNone(result)
        self.hccl_instance.hcclCommInitRank.assert_called_once()

    def test_broadcast_device_mismatch(self):
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        tensor = torch.rand(1)
        result = comm.broadcast(tensor, src=0)
        self.assertIsNone(result)

    def test_broadcast_normal(self):
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        tensor = torch.rand(1, device="cpu")
        result = comm.broadcast(tensor, src=0)
        self.assertIsNone(result)
        self.hccl_instance.hcclBroadcast.assert_called_once()

    def test_init_with_custom_library_path(self):
        library_path = "/custom/path/to/hccl.so"
        comm = PyHcclCommunicator(MagicMock(),
                                  device="cpu",
                                  library_path=library_path)
        self.assertIsInstance(comm.hccl, MagicMock)

    def test_init_with_stateless_group(self):
        group = MagicMock(spec=StatelessProcessGroup)
        group.rank = 0
        group.world_size = 2

        class MockBroadcastObj:

            def __init__(self):
                self.unique_id = TestPyHcclCommunicator.MockStatelessUniqueId()

            def __call__(self, obj, src):
                return self.unique_id

        group.broadcast_obj.return_value = TestPyHcclCommunicator.MockStatelessUniqueId(
        )
        comm = PyHcclCommunicator(group, device="cpu")
        self.assertEqual(comm.rank, 0)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.disabled)

    def test_init_world_size_1(self):
        self.mock_get_world_size.return_value = 1
        comm = PyHcclCommunicator(MagicMock(), device="cpu")
        self.assertTrue(comm.disabled)
        self.assertFalse(comm.available)

    def test_init_hccl_load_fail(self):
        self.mock_hccl_patcher.stop()
        self.mock_hccl_patcher = patch(
            'vllm_ascend.distributed.device_communicators.pyhccl.HCCLLibrary',
            side_effect=OSError("Load failed"))
        self.mock_hccl_patcher.start()

        try:
            comm = PyHcclCommunicator(MagicMock(), device="cpu")
            self.assertTrue(comm.disabled)
            self.assertFalse(comm.available)
        finally:
            self.mock_hccl_patcher.stop()
