"""Execute the real control flow with fake streams; NPU validation is separate."""

import ast
import contextlib
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"


class Fixture:
    def __init__(self):
        self.events = []
        self.local = threading.local()
        self.created = []
        self.apply = Mock(side_effect=self.record_apply)
        npu = SimpleNamespace(Stream=self.create_stream, stream=self.context, current_stream=lambda: self.local.stream)
        namespace = {"torch": SimpleNamespace(npu=npu), "logger": Mock()}
        cls = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "KVCacheRecvingThread"
        )
        names = {
            "_reformat_pending_kv_caches",
            "_synchronize_reformat_stream",
            "_raise_if_reformat_failed",
            "get_and_clear_finished_requests",
            "get_and_clear_invalid_block_ids",
        }
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=[*methods], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.worker = SimpleNamespace(
            pending_reformat_lock=threading.Lock(),
            pending_reformat={},
            reformat_execution_lock=threading.Lock(),
            reformat_stream=None,
            reformat_error=None,
            kv_caches={"layer": (SimpleNamespace(device="npu:1"), None)},
            _apply_kv_cache_reformat=self.apply,
        )
        for method in methods:
            method_impl = namespace[method.name]
            assert callable(method_impl)
            setattr(self.worker, method.name, MethodType(method_impl, self.worker))
        self.run = self.worker._reformat_pending_kv_caches

    def create_stream(self, *, device):
        self.created.append(device)
        return SimpleNamespace(synchronize=lambda: self.events.append("fence"))

    @contextlib.contextmanager
    def context(self, stream):
        previous = getattr(self.local, "stream", None)
        self.local.stream = stream
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")
            self.local.stream = previous

    def record_apply(self, value):
        assert self.local.stream is self.worker.reformat_stream
        assert self.worker.reformat_execution_lock.locked()
        self.events.append(value)


class TestPrivateReformatStream(unittest.TestCase):
    def test_device_order_reuse_and_terminal_fence(self):
        f = Fixture()
        f.worker.pending_reformat = {"a": {2: "second", 0: "first"}, "b": {0: "third"}}
        f.run("a")
        f.run("b")
        self.assertEqual(f.created, ["npu:1"])
        self.assertEqual(
            f.events,
            ["enter", "first", "second", "fence", "exit", "enter", "third", "fence", "exit"],
        )
        self.assertIsNone(f.local.stream)
        self.assertEqual(f.worker.pending_reformat, {})

    def test_failure_fences_restores_context_and_propagates(self):
        f = Fixture()
        f.worker.pending_reformat = {"a": {0: "first"}}
        f.apply.side_effect = RuntimeError("guard failed")
        with self.assertRaisesRegex(RuntimeError, "guard failed"):
            f.run("a")
        self.assertEqual(f.events, ["enter", "fence", "exit"])
        self.assertIsNone(f.local.stream)
        self.assertFalse(f.worker.reformat_execution_lock.locked())

    def test_no_pending_work_does_not_allocate_or_fence(self):
        f = Fixture()
        f.run("missing")
        self.assertEqual(f.events, [])
        self.assertEqual(f.created, [])

    def test_two_receive_threads_cannot_overlap_reformats(self):
        f = Fixture()
        f.worker.pending_reformat = {"a": {0: "a"}, "b": {0: "b"}}
        entered, release, second_done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def apply(value):
            f.record_apply(value)
            if value == "a":
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release first reformat")
            else:
                second_done.set()

        def run(name):
            try:
                f.run(name)
            except Exception as exc:
                errors.append(exc)

        f.apply.side_effect = apply
        a = threading.Thread(target=run, args=("a",))
        b = threading.Thread(target=run, args=("b",))
        a.start()
        self.assertTrue(entered.wait(2))
        b.start()
        self.assertFalse(second_done.wait(0.05))
        release.set()
        a.join(2)
        b.join(2)
        self.assertEqual(errors, [])
        self.assertTrue(second_done.is_set())
        self.assertEqual(f.created, ["npu:1"])

    def test_failed_fence_is_sticky_and_reaches_model_worker(self):
        f = Fixture()
        f.worker.pending_reformat = {"a": {0: "first"}, "b": {0: "second"}}
        fault = RuntimeError("device fence failed")

        def apply(value):
            f.record_apply(value)
            f.worker.reformat_stream.synchronize = Mock(side_effect=fault)

        f.apply.side_effect = apply
        with self.assertRaisesRegex(RuntimeError, "device fence failed"):
            f.run("a")
        self.assertIs(f.worker.reformat_error, fault)
        for poll in (f.worker.get_and_clear_finished_requests, f.worker.get_and_clear_invalid_block_ids):
            with self.assertRaisesRegex(RuntimeError, "failed to complete") as caught:
                poll()
            self.assertIs(caught.exception.__cause__, fault)
        with self.assertRaisesRegex(RuntimeError, "failed to complete"):
            f.run("b")
        self.assertEqual(f.apply.call_count, 1)
        self.assertFalse(f.worker.reformat_execution_lock.locked())

    def test_python_failure_with_successful_fence_does_not_poison_device(self):
        f = Fixture()
        f.worker.pending_reformat = {"a": {0: "first"}, "b": {0: "second"}}
        f.apply.side_effect = ValueError("invalid request")
        with self.assertRaisesRegex(ValueError, "invalid request"):
            f.run("a")
        self.assertIsNone(f.worker.reformat_error)
        f.apply.side_effect = f.record_apply
        f.run("b")
        self.assertEqual(f.events[-4:], ["enter", "second", "fence", "exit"])


if __name__ == "__main__":
    unittest.main()
