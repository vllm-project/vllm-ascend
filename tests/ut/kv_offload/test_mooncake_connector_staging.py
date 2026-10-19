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
    thread._landing_stream = MagicMock()
    thread._landing_streams = [thread._landing_stream, MagicMock()]
    thread._pipeline_banks = 2
    thread._async_landing_enabled = False
    thread._pipeline_landing_enabled = False
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

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.stream")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_pipeline_four_banks_round_robin(
        self, mock_register_buffer, mock_sync, mock_stream, mock_event_cls
    ):
        # BANKS=4 with WINDOW=4 -> one slot per bank; 6 chunks become six
        # single-chunk reads round-robining over banks 0,1,2,3,0,1. Wrapping
        # back to a bank (read 5 -> bank 0) reclaims its slot just-in-time;
        # each bank's event is recorded on that bank's own stream.
        offloaded_kv = torch.empty((6 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((4 * self.staging_chunk_bytes,), dtype=torch.int8)

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 4
        self.thread._pipeline_landing_enabled = True
        self.thread._pipeline_banks = 4
        streams = [MagicMock() for _ in range(4)]
        self.thread._landing_streams = streams

        order: list[str] = []

        def make_event(name: str):
            event = MagicMock()
            event.record.side_effect = lambda *a, **k: order.append(f"rec{name}")
            event.synchronize.side_effect = lambda: order.append(f"wait{name}")
            return event

        events = [make_event(str(i)) for i in range(6)]
        mock_event_cls.side_effect = events
        self.thread.engine.batch_transfer_sync_read.side_effect = lambda *a, **k: (
            order.append("read"),
            0,
        )[1]

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-banks4",
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            order,
            [
                # banks 0,1,2,3: fresh banks, no waits
                "read", "rec0",
                "read", "rec1",
                "read", "rec2",
                "read", "rec3",
                # read 5 wraps back to bank 0: reclaim slot 0 first
                "wait0", "read", "rec4",
                # read 6 wraps back to bank 1: reclaim slot 1 first
                "wait1", "read", "rec5",
                # final drain in slot order: 0,1,2,3
                "wait4", "wait5", "wait2", "wait3",
            ],
        )
        self.assertEqual(self.thread.engine.batch_transfer_sync_read.call_count, 6)
        # bank of chunk i = i % 4 -> stream index 0,1,2,3,0,1
        for i, event in enumerate(events):
            event.record.assert_called_once_with(streams[i % 4])
        self.assertEqual(
            mock_stream.call_args_list,
            [unittest.mock.call(s) for s in [streams[0], streams[1], streams[2], streams[3], streams[0], streams[1]]],
        )
        mock_sync.assert_not_called()
        mock_register_buffer.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.stream")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_async_landing_uses_dedicated_stream_and_event(
        self, mock_register_buffer, mock_sync, mock_stream, mock_event_cls
    ):
        offloaded_kv = torch.empty((3 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((2 * self.staging_chunk_bytes,), dtype=torch.int8)
        copied: list[tuple[int, int, int]] = []

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 2
        self.thread._async_landing_enabled = True
        self.thread.engine.batch_transfer_sync_read.return_value = 0

        def record_copy(dst: int, length: int, staging_offset: int = 0) -> None:
            copied.append((dst, length, staging_offset))

        self.thread._copy_dsa_pd_mooncake_staging_to_offloaded_kv = record_copy

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-async-on",
        )

        # 3 chunks over a window of 2 -> 2 windows, each awaited by its own
        # event on the dedicated landing stream; no device-wide sync at all.
        self.assertEqual(ret, 0)
        self.assertEqual(self.thread.engine.batch_transfer_sync_read.call_count, 2)
        self.assertEqual(mock_stream.call_count, 2)
        mock_stream.assert_called_with(self.thread._landing_stream)
        self.assertEqual(mock_event_cls.call_count, 2)
        self.assertEqual(mock_event_cls.return_value.record.call_count, 2)
        mock_event_cls.return_value.record.assert_called_with(self.thread._landing_stream)
        self.assertEqual(mock_event_cls.return_value.synchronize.call_count, 2)
        mock_sync.assert_not_called()
        mock_register_buffer.assert_not_called()
        self.assertEqual(
            copied,
            [
                # window 1: slots 0 and 1; window 2 wraps back to slot 0
                (offloaded_kv.data_ptr(), self.staging_chunk_bytes, 0),
                (
                    offloaded_kv.data_ptr() + self.staging_chunk_bytes,
                    self.staging_chunk_bytes,
                    self.staging_chunk_bytes,
                ),
                (offloaded_kv.data_ptr() + 2 * self.staging_chunk_bytes, self.staging_chunk_bytes, 0),
            ],
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.stream")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_async_landing_disabled_keeps_global_sync(
        self, mock_register_buffer, mock_sync, mock_stream, mock_event_cls
    ):
        offloaded_kv = torch.empty((3 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((4 * self.staging_chunk_bytes,), dtype=torch.int8)

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 4
        self.thread._async_landing_enabled = False
        self.thread.engine.batch_transfer_sync_read.return_value = 0

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-async-off",
        )

        # Kill switch off: legacy path — global device sync per window, the
        # dedicated stream and events are never touched.
        self.assertEqual(ret, 0)
        mock_sync.assert_called_once()
        mock_stream.assert_not_called()
        mock_event_cls.assert_not_called()
        mock_register_buffer.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.Event")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.stream")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer")
    def test_pipeline_landing_overlaps_windows(
        self, mock_register_buffer, mock_sync, mock_stream, mock_event_cls
    ):
        # WINDOW=4 means 4 total staging slots; pipeline splits them into two
        # banks of 2, so 6 chunks become three half-window reads. b1 uses
        # bank 0 (slots 0,1), b2 flips to bank 1 (slots 2,3) with no wait at
        # all, b3 comes back to bank 0 and must first reclaim slots 0 and 1.
        # The final drain waits out every remaining event before success is
        # reported.
        offloaded_kv = torch.empty((6 * self.staging_chunk_bytes,), dtype=torch.int8)
        register_dsa_pd_mooncake_offloaded_kv_tensor(offloaded_kv)
        staging = torch.empty((4 * self.staging_chunk_bytes,), dtype=torch.int8)

        self.thread._dsa_pd_mooncake_staging_tensor = staging
        self.thread._dsa_pd_mooncake_staging_bytes = staging.numel()
        self.thread._get_dsa_pd_mooncake_staging_chunk_bytes = lambda: self.staging_chunk_bytes
        self.thread._get_dsa_pd_mooncake_staging_window = lambda: 4
        self.thread._pipeline_landing_enabled = True

        copied: list[tuple[int, int, int]] = []

        def record_copy(dst: int, length: int, staging_offset: int = 0) -> None:
            copied.append((dst, length, staging_offset))

        self.thread._copy_dsa_pd_mooncake_staging_to_offloaded_kv = record_copy

        # Trace the interleaving of network reads, event records and event
        # waits to pin down the pipelining discipline.
        order: list[str] = []

        def make_event(name: str):
            event = MagicMock()
            event.record.side_effect = lambda *a, **k: order.append(f"rec{name}")
            event.synchronize.side_effect = lambda: order.append(f"wait{name}")
            return event

        events = [make_event(str(i)) for i in range(6)]
        mock_event_cls.side_effect = events
        self.thread.engine.batch_transfer_sync_read.side_effect = lambda *a, **k: (
            order.append("read"),
            0,
        )[1]

        ret = self.thread._batch_transfer_sync_read_with_optional_staging(
            "remote:7777",
            [offloaded_kv.data_ptr()],
            [0xA00000],
            [offloaded_kv.numel()],
            "req-pipeline",
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            order,
            [
                # b1 (bank 0): read, submit, record per slot
                "read", "rec0", "rec1",
                # b2 (bank 1): bank flip -> read starts with no wait
                "read", "rec2", "rec3",
                # b3 (bank 0 again): reclaim slots 0 and 1 before overwriting
                "wait0", "wait1", "read", "rec4", "rec5",
                # final drain: slots 0,1,2,3
                "wait4", "wait5", "wait2", "wait3",
            ],
        )
        self.assertEqual(self.thread.engine.batch_transfer_sync_read.call_count, 3)
        self.assertEqual(mock_event_cls.call_count, 6)
        # Each bank's copies are submitted and recorded on that bank's own
        # stream: bank 0 (slots 0,1) on stream 0, bank 1 (slots 2,3) on
        # stream 1, so the two banks' landings can execute in parallel.
        stream0, stream1 = self.thread._landing_streams
        expected_stream_per_event = [stream0, stream0, stream1, stream1, stream0, stream0]
        for event, expected_stream in zip(events, expected_stream_per_event):
            event.record.assert_called_once_with(expected_stream)
            event.synchronize.assert_called_once()
        mock_sync.assert_not_called()
        self.assertEqual(
            mock_stream.call_args_list,
            [unittest.mock.call(stream0), unittest.mock.call(stream1), unittest.mock.call(stream0)],
        )
        mock_register_buffer.assert_not_called()
        chunk = self.staging_chunk_bytes
        self.assertEqual(
            copied,
            [
                (offloaded_kv.data_ptr(), chunk, 0),
                (offloaded_kv.data_ptr() + chunk, chunk, chunk),
                (offloaded_kv.data_ptr() + 2 * chunk, chunk, 2 * chunk),
                (offloaded_kv.data_ptr() + 3 * chunk, chunk, 3 * chunk),
                (offloaded_kv.data_ptr() + 4 * chunk, chunk, 0),
                (offloaded_kv.data_ptr() + 5 * chunk, chunk, chunk),
            ],
        )


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
