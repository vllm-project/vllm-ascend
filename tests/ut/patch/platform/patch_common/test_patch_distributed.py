import unittest
from unittest.mock import patch

from vllm.config import ParallelConfig


class TestNullHandle(unittest.TestCase):

    def test_null_handle_initialization(self):
        from vllm_ascend.patch.platform.patch_common.patch_distributed import \
            NullHandle
        handle = NullHandle()
        self.assertIsInstance(handle, NullHandle)

    def test_null_handle_wait(self):
        from vllm_ascend.patch.platform.patch_common.patch_distributed import \
            NullHandle
        handle = NullHandle()
        handle.wait()


class TestParallelConfigGetDpPort(unittest.TestCase):

    @patch('vllm.envs.VLLM_DP_MASTER_PORT', None)
    def test_get_dp_port_no_env(self):
        config = ParallelConfig()
        config.data_parallel_master_port = 29500

        port1 = config.get_next_dp_init_port()
        self.assertEqual(port1, 29500)

        port2 = config.get_next_dp_init_port()
        self.assertEqual(port2, 29501)

        port3 = config.get_next_dp_init_port()
        self.assertEqual(port3, 29502)

    @patch('vllm.envs.VLLM_DP_MASTER_PORT', 30000)
    def test_get_dp_port_with_env(self):
        config = ParallelConfig()
        config.data_parallel_master_port = 29500

        port = config.get_next_dp_init_port()
        self.assertEqual(port, 30000)

        port2 = config.get_next_dp_init_port()
        self.assertEqual(port2, 30000)


class TestCommunicationAdaptation(unittest.TestCase):

    def setUp(self):
        import torch.distributed as dist
        self.original_broadcast = dist.broadcast
        self.original_all_reduce = dist.all_reduce

    def test_communication_adaptation_310p(self):
        import torch.distributed as dist

        from vllm_ascend.patch.platform.patch_common.patch_distributed import (
            communication_adaptation_310p, is_310p)
        if is_310p():
            communication_adaptation_310p()
            self.assertNotEqual(dist.broadcast, self.original_broadcast)
            self.assertNotEqual(dist.all_reduce, self.original_all_reduce)
        else:
            self.assertEqual(dist.broadcast, self.original_broadcast)
            self.assertEqual(dist.all_reduce, self.original_all_reduce)


if __name__ == '__main__':
    unittest.main()
