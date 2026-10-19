"""CPU control-flow regression for the real connector methods, without NPU imports.

These tests cover retained outer operands and completion/error publication.
Helper-local transpose allocation lifetime requires the real NPU gates.
These tests execute methods extracted from the runtime source, rather than a
copy of their implementation. Fake device operations only model asynchronous
completion and Python tensor ownership; they do not validate CANN execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
import time
import unittest
import weakref
from collections import OrderedDict
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"


class Tensor:
    def __init__(self, parents=()):
        self.parents = parents
        self.device = "npu:3"
        self.dtype = "bf16"

    def view(self, *shape):
        return Tensor((self,))

    reshape = view

    def flatten(self):
        return Tensor((self,))

    def __mul__(self, other):
        return Tensor((self,))

    def __add__(self, other):
        return Tensor((self, other))


class Fixture:
    def __init__(self, *, block=False, fail=False, request_level=False):
        self.calls: list[str] = []
        self.pending = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.completed = False
        self.block, self.fail = block, fail
        self.stream = SimpleNamespace(synchronize=self.synchronize)
        torch = SimpleNamespace(
            tensor=self.make_tensor,
            empty=lambda *a, **k: Tensor(),
            empty_like=lambda *a, **k: Tensor(),
            arange=lambda *a, **k: Tensor(),
            int32="i32",
            npu=SimpleNamespace(
                synchronize=lambda: self.calls.append("existing_device_sync"),
                current_stream=self.current_stream,
            ),
        )
        self.torch, self.logger = torch, Mock()
        namespace = {
            "torch": torch,
            "torch_npu": SimpleNamespace(npu_gather_pa_kv_cache=self.gather),
            "logger": self.logger,
            "os": os,
            "time": time,
            "threading": threading,
            "json": json,
            "hashlib": hashlib,
            "OrderedDict": OrderedDict,
        }
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVCacheRecvingThread")
        tracker_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVCacheTaskTracker")
        method_names = {
            "reformat_kv_cache",
            "_handle_request",
            "_synchronize_reformat_stream",
            "_raise_if_reformat_failed",
            "_is_failed_recv_request",
            "_mark_failed_recv_request",
            "_mark_request_task_done",
            "get_and_clear_finished_requests",
            "get_and_clear_failed_recv_requests",
            "get_and_clear_invalid_block_ids",
        }
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                tracker_cls,
                *methods,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        tracker_type = namespace["KVCacheTaskTracker"]
        assert callable(tracker_type)
        tracker = tracker_type()
        tracker.add_req_to_process("local")
        update_done = tracker.update_done_task_count

        def publish_completion(request_id):
            update_done(request_id)
            self.calls.append("done")

        tracker.update_done_task_count = publish_completion
        self.worker = SimpleNamespace(
            kv_caches={"layer0": (Tensor(), Tensor()), "layer1": (Tensor(), Tensor())},
            block_size=128,
            reformat_error=None,
            _get_kv_cache_dims_from_tensors=lambda _: (4, 128, 128),
            _cat_kv_cache=self.scatter,
            _nz_kv_cache=lambda *a: self.calls.append("nz"),
            use_request_level_recv_errors=request_level,
            failed_recv_requests=set(),
            failed_recv_requests_lock=threading.Lock(),
            invalid_block_ids=set(),
            request_task_counts={},
            request_task_counts_lock=threading.Lock(),
            finished_request_markers=set(),
            _transfer_kv_cache_all_groups=lambda _: self.calls.append("transfer"),
            pending_reformat_lock=threading.Lock(),
            pending_reformat={},
            proc_not_transfer_request_lock=threading.Lock(),
            proc_not_transfer_request={},
            request_queue=SimpleNamespace(task_done=lambda: None),
            task_tracker=tracker,
            _send_done_signal_to_free_remote_port=lambda *a: None,
            _send_done_recv_signal=lambda *a: None,
        )
        for name in method_names:
            method_impl = namespace[name]
            assert callable(method_impl)
            setattr(self.worker, name, MethodType(method_impl, self.worker))
        mark_failed = self.worker._mark_failed_recv_request

        def record_failure(*args):
            mark_failed(*args)
            self.calls.append("invalid_blocks")

        self.worker._mark_failed_recv_request = record_failure
        self.worker._reformat_pending_kv_caches = lambda _: self.reformat()

    def make_tensor(self, *args, **kwargs):
        tensor = Tensor()
        tensor.device = kwargs.get("device", "npu:3")
        return tensor

    def current_stream(self):
        self.calls.append("current_stream")
        return self.stream

    def gather(self, k, v, block_table, block_len, **kwargs):
        self.calls.append("gather")
        self.pending.extend(
            weakref.ref(t)
            for t in (
                block_table,
                block_len,
                kwargs["seq_offset"],
                kwargs["key"],
                kwargs["value"],
            )
        )

    def scatter(self, *args):
        self.calls.append("scatter")
        self.pending.extend(
            weakref.ref(args[i]) for i in (7,)
        )  # Slot mapping; helper temporaries depend on real stream allocator semantics.

    def synchronize(self):
        self.calls.append("stream_sync_enter")
        if not self.pending or not all(ref() is not None for ref in self.pending):
            raise AssertionError("Reformat operands released before stream completion")
        self.entered.set()
        if self.block and not self.release.wait(5):
            raise TimeoutError("Test did not release fake stream")
        if self.fail:
            self.calls.append("stream_sync_error")
            raise RuntimeError("Asynchronous device failure")
        self.completed = True
        self.calls.append("stream_sync_done")

    def reformat(self):
        self.worker.reformat_kv_cache([[10, 11]], 4, need_cat_cache=True)

    def handle(self):
        self.worker._handle_request(
            dict(
                request_id="local",
                remote_request_id="remote",
                remote_host="localhost",
                remote_handshake_port=1,
                remote_port_send_num={},
                all_task_done=True,
                local_block_ids=[[10, 11]],
            )
        )


class CompletionFenceTests(unittest.TestCase):
    def test_operands_live_until_current_cache_device_stream_completes(self):
        fixture = Fixture()
        fixture.reformat()
        self.assertTrue(fixture.completed)
        self.assertEqual(
            fixture.calls,
            [
                "existing_device_sync",
                "gather",
                "scatter",
                "gather",
                "scatter",
                "current_stream",
                "stream_sync_enter",
                "stream_sync_done",
            ],
        )
        self.assertTrue(
            all(ref() is None for ref in fixture.pending),
            "Fake queued operands must not accidentally be held by test",
        )

    def test_tracker_cannot_publish_ready_while_stream_is_running(self):
        fixture = Fixture(block=True)
        thread = threading.Thread(target=fixture.handle)
        thread.start()
        try:
            self.assertTrue(fixture.entered.wait(5))
            self.assertNotIn("done", fixture.calls)
            self.assertTrue(all(ref() is not None for ref in fixture.pending))
        finally:
            fixture.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertLess(fixture.calls.index("stream_sync_done"), fixture.calls.index("done"))
        self.assertNotIn("invalid_blocks", fixture.calls)

    def test_fence_failure_never_publishes_completion(self):
        for request_level in (False, True):
            with self.subTest(request_level=request_level):
                fixture = Fixture(fail=True, request_level=request_level)
                fixture.handle()
                self.assertFalse(fixture.completed)
                self.assertNotIn("stream_sync_done", fixture.calls)
                self.assertLess(
                    fixture.calls.index("stream_sync_error"),
                    fixture.calls.index("invalid_blocks"),
                )
                self.assertNotIn("done", fixture.calls)
                self.assertIsInstance(fixture.worker.reformat_error, RuntimeError)
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                self.assertEqual(fixture.worker.task_tracker.get_and_clear_finished_requests(), set())
                with self.assertRaisesRegex(RuntimeError, "failed to complete") as raised:
                    fixture.worker.get_and_clear_finished_requests()
                self.assertIs(raised.exception.__cause__, fixture.worker.reformat_error)
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))

    def test_exception_after_enqueue_still_fences_before_invalid_completion(self):
        for request_level in (False, True):
            with self.subTest(request_level=request_level):
                fixture = Fixture(request_level=request_level)
                original = fixture.worker._cat_kv_cache

                def fail_after_scatter(*args, original=original):
                    original(*args)
                    raise ValueError("after enqueue")

                fixture.worker._cat_kv_cache = fail_after_scatter
                fixture.handle()
                self.assertTrue(fixture.completed)
                self.assertLess(
                    fixture.calls.index("stream_sync_done"),
                    fixture.calls.index("invalid_blocks"),
                )
                self.assertLess(fixture.calls.index("invalid_blocks"), fixture.calls.index("done"))
                self.assertIsNone(fixture.worker.reformat_error)
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                self.assertEqual(fixture.worker.get_and_clear_failed_recv_requests(set()), set())
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                finished = fixture.worker.get_and_clear_finished_requests()
                self.assertEqual(finished, {"local"})
                self.assertEqual(
                    fixture.worker.get_and_clear_failed_recv_requests(finished),
                    {"local"} if request_level else set(),
                )
                self.assertFalse(fixture.worker._is_failed_recv_request("local"))
                self.assertEqual(fixture.worker.get_and_clear_failed_recv_requests(finished), set())
                self.assertEqual(fixture.worker.get_and_clear_finished_requests(), set())
                self.assertEqual(fixture.worker.get_and_clear_invalid_block_ids(), set() if request_level else {10, 11})
                self.assertEqual(fixture.worker.get_and_clear_invalid_block_ids(), set())

    def test_receive_failure_is_retained_until_all_pull_tasks_complete(self):
        for request_level in (False, True):
            with self.subTest(request_level=request_level):
                fixture = Fixture(request_level=request_level)
                fixture.worker.request_task_counts["local"] = 2
                fixture.worker.finished_request_markers.add("local")
                fixture.worker._transfer_kv_cache_all_groups = Mock(side_effect=ValueError("transfer failed"))
                fixture.handle()
                self.assertNotIn("done", fixture.calls)
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                finished = fixture.worker.get_and_clear_finished_requests()
                self.assertEqual(finished, set())
                self.assertEqual(fixture.worker.get_and_clear_failed_recv_requests(finished), set())
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                fixture.handle()
                self.assertEqual(fixture.calls.count("done"), 1)
                self.assertTrue(fixture.worker._is_failed_recv_request("local"))
                finished = fixture.worker.get_and_clear_finished_requests()
                self.assertEqual(finished, {"local"})
                self.assertEqual(
                    fixture.worker.get_and_clear_failed_recv_requests(finished),
                    {"local"} if request_level else set(),
                )
                self.assertFalse(fixture.worker._is_failed_recv_request("local"))
                self.assertEqual(fixture.worker.get_and_clear_failed_recv_requests(finished), set())

    def test_empty_block_list_does_not_submit_device_work(self):
        fixture = Fixture()
        fixture.worker.reformat_kv_cache([[]], 2, need_cat_cache=True)
        self.assertEqual(fixture.calls, [])


if __name__ == "__main__":
    unittest.main()
