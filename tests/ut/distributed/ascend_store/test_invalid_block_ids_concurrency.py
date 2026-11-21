#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Concurrency tests for the invalid KV block ID set (issue #14150).

The pool worker records block IDs whose KV load failed while the model runner
drains them through ``get_block_ids_with_load_errors``. These tests exercise
that producer/consumer pair with independent threads.
"""

import threading
import unittest
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

JOIN_TIMEOUT_SECONDS = 30


class _LockProbe:
    """Lock wrapper that records whether the calling thread holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = threading.local()

    def __enter__(self) -> None:
        self._lock.acquire()
        self._held.value = True

    def __exit__(self, *exc_info: object) -> None:
        self._held.value = False
        self._lock.release()

    def held_by_current_thread(self) -> bool:
        return getattr(self._held, "value", False)


class _GuardedBlockIdSet(set):
    """Set that fails if it is touched without holding the shared lock."""

    def __init__(self, probe: _LockProbe) -> None:
        super().__init__()
        self._probe = probe

    def _assert_locked(self) -> None:
        if not self._probe.held_by_current_thread():
            raise AssertionError("invalid block ID state accessed without its lock")

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        self._assert_locked()
        super().update(*args, **kwargs)

    def copy(self):
        self._assert_locked()
        return super().copy()

    def clear(self) -> None:
        self._assert_locked()
        super().clear()


def make_bare_worker() -> KVPoolWorker:
    """Create a pool worker with only the shared block-ID state initialized."""
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    return worker


def make_recv_thread(worker: KVPoolWorker) -> KVCacheStoreRecvingThread:
    token_database = MagicMock()
    token_database.group_block_len = [[1]]
    return KVCacheStoreRecvingThread(
        m_store=MagicMock(),
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        ready_event=threading.Event(),
        invalid_block_ids=worker._invalid_block_ids,
        invalid_block_ids_lock=worker._invalid_block_ids_lock,
    )


class TestInvalidBlockIdConcurrency(unittest.TestCase):
    def test_concurrent_load_failures_and_queries_report_each_block_once(self):
        worker = make_bare_worker()
        writers = 4
        readers = 2
        blocks_per_writer = 500
        total = writers * blocks_per_writer
        barrier = threading.Barrier(writers + readers + 1)
        writers_done = threading.Event()
        seen: list[int] = []
        seen_lock = threading.Lock()
        errors: list[BaseException] = []

        def writer(writer_id: int) -> None:
            try:
                barrier.wait()
                base = writer_id * blocks_per_writer
                for offset in range(blocks_per_writer):
                    worker._record_invalid_blocks_ids([base + offset])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                barrier.wait()
                while not writers_done.is_set():
                    batch = worker.get_block_ids_with_load_errors()
                    with seen_lock:
                        seen.extend(batch)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        writer_threads = [threading.Thread(target=writer, args=(index,)) for index in range(writers)]
        reader_threads = [threading.Thread(target=reader) for _ in range(readers)]
        for thread in [*reader_threads, *writer_threads]:
            thread.start()
        barrier.wait()
        for thread in writer_threads:
            thread.join(timeout=JOIN_TIMEOUT_SECONDS)
        writers_done.set()
        for thread in reader_threads:
            thread.join(timeout=JOIN_TIMEOUT_SECONDS)

        stuck = [thread for thread in [*reader_threads, *writer_threads] if thread.is_alive()]
        self.assertFalse(stuck, "worker thread did not finish")
        self.assertFalse(errors, errors)
        seen.extend(worker.get_block_ids_with_load_errors())
        self.assertEqual(sorted(seen), list(range(total)))
        self.assertEqual(len(seen), len(set(seen)))

    def test_repeated_block_ids_are_reported_once(self):
        worker = make_bare_worker()
        worker._record_invalid_blocks_ids([7, 7, 8])
        worker._record_invalid_blocks_ids([8, 8, 7])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {7, 8})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_reported_snapshot_is_detached_from_shared_state(self):
        worker = make_bare_worker()
        worker._record_invalid_blocks_ids([1, 2])
        snapshot = worker.get_block_ids_with_load_errors()
        snapshot.add(99)
        snapshot.clear()
        worker._record_invalid_blocks_ids([1])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {1})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_every_shared_state_access_holds_the_lock(self):
        probe = _LockProbe()
        worker = make_bare_worker()
        worker._invalid_block_ids = _GuardedBlockIdSet(probe)
        worker._invalid_block_ids_lock = probe  # type: ignore[assignment]
        worker._record_invalid_blocks_ids([1, 2])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {1, 2})

    def test_recv_thread_and_worker_share_block_state(self):
        worker = make_bare_worker()
        recv_thread = make_recv_thread(worker)
        self.assertIs(recv_thread._invalid_block_ids, worker._invalid_block_ids)
        self.assertIs(recv_thread._invalid_block_ids_lock, worker._invalid_block_ids_lock)
        worker._record_invalid_blocks_ids([5])
        self.assertEqual(recv_thread._invalid_block_ids, {5})


if __name__ == "__main__":
    unittest.main()
