"""Unit tests for the staged (HBM-relay) receive path in mooncake_connector.

Ported from the glm52 branch increment to tests/ut/kv_offload/
test_mooncake_connector.py (commit chain 1e9ebb64..8956d8f3). The upstream
main tree has no test_mooncake_connector.py baseline, so these tests live
in a standalone file with a minimal fake receiving thread (same approach
as the smoke tool). The glm52 TestGlobalTE cases targeting the private
combined-engine facade were rewritten against the main-tree GlobalTE
(ptr, size) dedup in mooncake_transfer_engine.py.
"""

import os
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    GlobalTE,
)
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    clear_dsa_pd_mooncake_kv_buffer_registries_for_testing,
    register_dsa_pd_mooncake_offloaded_kv_tensor,
)


def _make_fake_recv_thread(staging_chunk_bytes: int) -> KVCacheRecvingThread:
    thread = object.__new__(KVCacheRecvingThread)
    thread.engine = MagicMock()
    thread.kv_caches = {}
    thread.block_len_per_addr = [[int(staging_chunk_bytes)]]
    thread._dsa_pd_mooncake_staging_storage = None
    thread._dsa_pd_mooncake_staging_tensor = None
    thread._dsa_pd_mooncake_staging_bytes = 0
    thread._dsa_pd_mooncake_staging_lock = threading.Lock()
    return thread


class TestKVCacheRecvingThreadStaging(unittest.TestCase):
    def setUp(self):
        clear_dsa_pd_mooncake_kv_buffer_registries_for_testing()
        self.staging_chunk_bytes = 2 * 1024 * 1024
        self.thread = _make_fake_recv_thread(self.staging_chunk_bytes)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_offloaded_kv_transfer_uses_reused_fixed_staging_chunk(self, mock_register_buffer):
        offloaded_kv = torch.empty((3 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((self.staging_chunk_bytes,), dtype=torch.int8)
        copied: list[tuple[int, int, int]] = []

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 1

        def record_copy(dst: int, length: int, staging_offset: int = 0) -> None:
            copied.append((dst, length, staging_offset))

        self.thread._copy_dsa_pd_mooncake_staging_to_offloaded_kv = record_copy
        self.thread.engine.batch_transfer_sync_read.return_value = 0
        remote_base_addr = 0xA00000

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [remote_base_addr],
            [offloaded_kv.numel()],
            "req-staged",
        )

        self.assertEqual(ret, 0)
        self.assertEqual(self.thread.engine.batch_transfer_sync_read.call_count, 3)
        expected_calls = [
            unittest.mock.call(
                "remote:7777",
                [staging.data_ptr()],
                [remote_base_addr + offset],
                [self.staging_chunk_bytes],
            )
            for offset in range(0, offloaded_kv.numel(), self.staging_chunk_bytes)
        ]
        self.thread.engine.batch_transfer_sync_read.assert_has_calls(expected_calls)
        self.assertEqual(
            copied,
            [
                (offloaded_kv.data_ptr() + offset, self.staging_chunk_bytes, 0)
                for offset in range(0, offloaded_kv.numel(), self.staging_chunk_bytes)
            ],
        )
        mock_register_buffer.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_offloaded_staging_read_synchronizes_once_per_batch(self, mock_register_buffer, mock_sync):
        offloaded_kv = torch.empty((3 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((4 * self.staging_chunk_bytes,), dtype=torch.int8)

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 4
        self.thread.engine.batch_transfer_sync_read.return_value = 0

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-staged",
        )

        self.assertEqual(ret, 0)
        self.thread.engine.batch_transfer_sync_read.assert_called_once()
        mock_sync.assert_called_once()
        mock_register_buffer.assert_not_called()

    def test_offloaded_staging_read_holds_shared_lock(self):
        offloaded_kv = torch.empty((4096,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)

        def assert_lock_held(*args, **kwargs):
            self.assertTrue(self.thread._dsa_pd_mooncake_staging_lock.locked())
            return 0

        self.thread._batch_transfer_sync_read_with_optional_staging_unlocked = (
            assert_lock_held
        )
        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-staged-lock",
        )

        self.assertEqual(ret, 0)
        self.assertFalse(self.thread._dsa_pd_mooncake_staging_lock.locked())

    def test_staging_window_uses_central_ascend_env(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW": "3"}):
            self.assertEqual(self.thread._get_dsa_pd_mooncake_staging_window(), 3)


class TestGlobalTERegisterBufferDedup(unittest.TestCase):
    """Main-tree GlobalTE: (ptr, size) dedup so lazily built buffers register late.

    Rewritten from the glm52 TestGlobalTE facade tests (protocol forwarding /
    locations API) which target the private combined transfer_engine_backend
    that does not exist on main.
    """

    def _make_backend(self) -> GlobalTE:
        backend = GlobalTE()
        backend.transfer_engine = MagicMock()
        backend.transfer_engine.register_memory.return_value = 0
        return backend

    def test_same_ptr_size_pair_is_registered_once(self):
        backend = self._make_backend()

        backend.register_buffer([0x1000, 0x2000], [64, 128])
        backend.register_buffer([0x1000, 0x2000, 0x3000], [64, 128, 256])

        self.assertEqual(
            backend.transfer_engine.register_memory.call_args_list,
            [
                unittest.mock.call(0x1000, 64),
                unittest.mock.call(0x2000, 128),
                unittest.mock.call(0x3000, 256),
            ],
        )
        self.assertTrue(backend.is_register_buffer)

    def test_failed_registration_is_not_cached(self):
        backend = self._make_backend()
        backend.transfer_engine.register_memory.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            backend.register_buffer([0x1000], [64])

        backend.transfer_engine.register_memory.return_value = 0
        backend.register_buffer([0x1000], [64])

        # the failed pair was not memoized: the retry reaches the engine again
        self.assertEqual(
            backend.transfer_engine.register_memory.call_args_list,
            [
                unittest.mock.call(0x1000, 64),
                unittest.mock.call(0x1000, 64),
            ],
        )


if __name__ == "__main__":
    unittest.main()
