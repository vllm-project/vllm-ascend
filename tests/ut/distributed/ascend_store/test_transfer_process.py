from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest import SkipTest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.transfer_planner import (
    TransferPlanner,
    _planner_main,
    iter_token_key_strings_with_block_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.transfer_supervisor import (
    TransferTimeoutError,
    TransferWorkerError,
    TransferSupervisor,
)


class _FakeTokenDatabase:
    hash_block_size = 1

    def get_block_size(self, group_id: int) -> int:
        return 1

    def _get_key_prefix(self, group_id: int) -> str:
        return "model@group:0@"

    def _iter_token_chunks(self, token_len, block_hashes, mask_num, group_id, **kwargs):
        for index, value in enumerate(block_hashes[:token_len]):
            if index >= mask_num:
                yield index, index + 1, value, kwargs.get("block_ids", [])[index]

    def process_token_key_strings_with_block_ids(self, *args, **kwargs):
        raise AssertionError("the planner path should be used")


class _FakeThreadTokenDatabase:
    group_block_len = {0: [1]}


class _FakeBackend:
    def set_device(self):
        return None


class TestTransferPlanner(unittest.TestCase):
    def test_planner_core_without_os_process(self):
        requests = queue.Queue()
        responses = queue.Queue()
        ready = queue.Queue()
        requests.put({
            "command_id": 1,
            "prefix": "model@",
            "token_len": 2,
            "block_hashes": [b"a", b"b"],
            "block_ids": [7, 8],
            "block_size": 1,
            "hash_block_size": 1,
        })
        requests.put(None)
        thread = threading.Thread(target=_planner_main, args=(requests, responses, ready))
        thread.start()
        self.assertEqual(ready.get(timeout=1), (True, ""))
        command_id, ok, entries, message = responses.get(timeout=1)
        thread.join(timeout=1)
        self.assertEqual((command_id, ok, message), (1, True, ""))
        self.assertEqual([entry[2] for entry in entries], ["model@61", "model@62"])

    def test_spawn_planner_generates_stable_keys(self):
        try:
            planner = TransferPlanner(timeout_s=5)
        except PermissionError as exc:
            raise SkipTest(f"multiprocessing is unavailable in this environment: {exc}") from exc
        try:
            entries = planner.plan(
                prefix="model@",
                token_len=2,
                block_hashes=[b"a", b"b"],
                block_ids=[7, 8],
                block_size=1,
                hash_block_size=1,
            )
            self.assertEqual(
                [(item[0], item[1], item[2], item[4]) for item in entries],
                [(0, 1, "model@61", 7), (1, 2, "model@62", 8)],
            )
        finally:
            planner.close()

    def test_planner_adapter_preserves_chunk_filter(self):
        try:
            planner = TransferPlanner(timeout_s=5)
        except PermissionError as exc:
            raise SkipTest(f"multiprocessing is unavailable in this environment: {exc}") from exc
        try:
            database = _FakeTokenDatabase()
            entries = list(
                iter_token_key_strings_with_block_ids(
                    database,
                    planner,
                    3,
                    ["a", "b", "c"],
                    [1, 2, 3],
                    chunk_filter=lambda start: start != 1,
                )
            )
            self.assertEqual([entry[2] for entry in entries], ["model@group:0@a", "model@group:0@c"])
        finally:
            planner.close()


class TestTransferSupervisor(unittest.TestCase):
    def test_fatal_error_wakes_waiter(self):
        supervisor = TransferSupervisor(timeout_s=1)
        event = threading.Event()

        def fail_later():
            time.sleep(0.02)
            supervisor.report_fatal(RuntimeError("boom"), "test-worker")

        thread = threading.Thread(target=fail_later)
        thread.start()
        try:
            with self.assertRaises(TransferWorkerError):
                supervisor.wait_for_event(event, description="test event")
        finally:
            thread.join()

    def test_event_timeout_is_bounded(self):
        supervisor = TransferSupervisor(timeout_s=0.02)
        with self.assertRaises(TransferTimeoutError):
            supervisor.wait_for_event(threading.Event(), description="never", timeout_s=0.02)

    def test_shutdown_wakes_event_waiter(self):
        supervisor = TransferSupervisor(timeout_s=1)
        supervisor.shutdown()
        with self.assertRaises(TransferWorkerError):
            supervisor.wait_for_event(threading.Event(), description="stopped")


class TestKVTransferThread(unittest.TestCase):
    def _load_thread_class(self):
        try:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
        except ModuleNotFoundError as exc:
            if exc.name in {"numpy", "torch"}:
                self.skipTest(f"KV transfer dependencies are unavailable: {exc.name}")
            raise
        return KVTransferThread

    def test_idle_thread_does_not_enter_fatal_state(self):
        KVTransferThread = self._load_thread_class()
        supervisor = TransferSupervisor(timeout_s=1)
        ready = threading.Event()
        thread = KVTransferThread(
            _FakeBackend(),
            _FakeThreadTokenDatabase(),
            block_size=1,
            tp_rank=0,
            ready_event=ready,
            supervisor=supervisor,
        )
        thread.start()
        self.assertTrue(ready.wait(timeout=1))
        time.sleep(0.25)
        self.assertTrue(thread.is_alive())
        supervisor.raise_if_failed()
        thread.stop()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    def test_handler_exception_completes_queue_task_once(self):
        KVTransferThread = self._load_thread_class()

        class _RaisingThread(KVTransferThread):
            def _handle_request(self, request):
                raise RuntimeError("request failed")

        supervisor = TransferSupervisor(timeout_s=1)
        ready = threading.Event()
        thread = _RaisingThread(
            _FakeBackend(),
            _FakeThreadTokenDatabase(),
            block_size=1,
            tp_rank=0,
            ready_event=ready,
            supervisor=supervisor,
        )
        thread.start()
        self.assertTrue(ready.wait(timeout=1))
        thread.add_request(object())
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)
        with self.assertRaises(TransferWorkerError):
            supervisor.raise_if_failed()
