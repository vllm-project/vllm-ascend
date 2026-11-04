# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python 3.10+; no vLLM, torch, NPU or network required."""

import ast
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_ascend").is_dir())
BASE = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/"


def load_class(filename, name, methods, bases, namespace):
    tree = ast.parse((ROOT / BASE / filename).read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    body: list[ast.stmt] = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    cls = ast.parse(f"class {name}(Base):\n    pass\n").body[0]
    assert isinstance(cls, ast.ClassDef)
    cls.body = body or [ast.Pass()]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = dict(namespace, Base=bases)
    exec(compile(ast.fix_missing_locations(module), str(ROOT / BASE / filename), "exec"), namespace)
    return namespace[name]


class AbortableQueue(queue.Queue):
    """A test-only escape hatch to stop the unpatched infinite receiver."""

    abort_marker = object()

    def get(self, *args, **kwargs):
        work = super().get(*args, **kwargs)
        if work is self.abort_marker:
            self.task_done()
            raise SystemExit
        return work


class MooncakeShutdownRepro(unittest.TestCase):
    def setUp(self):
        self.hardware = patch.dict(sys.modules, {"torch_npu": MagicMock(), "acl": MagicMock()})
        self.hardware.start()
        env = {"torch": MagicMock(), "queue": queue, "logger": MagicMock()}
        receiver_type = load_class(
            "pull_worker.py",
            "MooncakePullRecvingThread",
            {"run", "shutdown", "_get_remote_endpoint"},
            threading.Thread,
            env,
        )
        worker_base = load_class("base_worker.py", "MooncakeBaseConnectorWorker", {"shutdown"}, object, env)
        worker_type = load_class("pull_worker.py", "MooncakePullConnectorWorker", {"shutdown"}, worker_base, env)
        connector_type = load_class("connector.py", "MooncakeBaseConnector", {"shutdown"}, object, env)
        self.receiver = receiver_type.__new__(receiver_type)
        threading.Thread.__init__(self.receiver, daemon=True)
        self.receiver.request_queue = AbortableQueue()
        self.receiver.finished_requests = queue.SimpleQueue()
        self.receiver.ready_event = threading.Event()
        self.receiver.device = "cpu"
        self.receiver.executor = MagicMock()
        self.receiver.can_report_invalid_block_ids = False
        self.receiver._handle_requests = MagicMock(return_value=set())
        self.receiver._mark_request_failed = MagicMock()
        self.worker = worker_type()
        self.worker._recving_thread = self.receiver
        self.connector = connector_type()
        self.connector.connector_worker = self.worker
        self.connector.connector_scheduler = None

    def tearDown(self):
        if self.receiver.is_alive():
            self.receiver.request_queue.put(AbortableQueue.abort_marker)
            self.receiver.join(timeout=2)
        self.hardware.stop()

    def test_idle_shutdown_stops_receiver_and_is_repeatable(self):
        self.receiver.start()
        self.assertTrue(self.receiver.ready_event.wait(timeout=2))
        self.connector.shutdown()
        self.assertFalse(
            self.receiver.is_alive(), "public connector shutdown leaves the receiver blocked in queue.get()"
        )
        self.connector.shutdown()
        self.receiver.executor.shutdown.assert_called_once_with(wait=True)
        self.assertIsNone(self.worker._recving_thread)

    def test_shutdown_drains_already_queued_requests(self):
        request = SimpleNamespace(remote_host="127.0.0.1", remote_port=9000)
        self.receiver.request_queue.put(("producer", {"req": request}))
        self.receiver.start()
        self.assertTrue(self.receiver.ready_event.wait(timeout=2))
        self.connector.shutdown()
        self.assertFalse(self.receiver.is_alive())
        self.receiver._handle_requests.assert_called_once_with("producer", "127.0.0.1", 9000, {"req": request})
        self.assertEqual(self.receiver.finished_requests.get_nowait(), ("req", False))
        self.assertEqual(self.receiver.request_queue.unfinished_tasks, 0)

    def test_shutdown_before_thread_start_releases_executor(self):
        self.connector.shutdown()
        self.receiver.executor.shutdown.assert_called_once_with(wait=True)
        self.connector.shutdown()
        self.receiver.executor.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_retains_ownership_until_cleanup_finishes(self):
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()
        shutdown_errors: list[BaseException] = []

        def wait_for_cleanup(*args, **kwargs):
            cleanup_started.set()
            if not allow_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release executor cleanup")

        def stop_connector():
            try:
                self.connector.shutdown()
            except BaseException as error:
                shutdown_errors.append(error)

        self.receiver.executor.shutdown.side_effect = wait_for_cleanup
        shutdown_thread = threading.Thread(target=stop_connector, daemon=True)
        shutdown_thread.start()
        try:
            self.assertTrue(cleanup_started.wait(timeout=2))
            self.assertTrue(shutdown_thread.is_alive())
            self.assertIs(self.connector.connector_worker, self.worker)
            self.assertIs(self.worker._recving_thread, self.receiver)
        finally:
            allow_cleanup.set()
            shutdown_thread.join(timeout=2)
        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(shutdown_errors, [])
        self.assertIsNone(self.worker._recving_thread)
        self.receiver.executor.shutdown.assert_called_once_with(wait=True)

    def test_failed_shutdown_preserves_resources_for_retry(self):
        self.receiver.executor.shutdown.side_effect = [RuntimeError("executor cleanup failed"), None]
        with self.assertRaisesRegex(RuntimeError, "executor cleanup failed"):
            self.connector.shutdown()
        self.assertIs(self.connector.connector_worker, self.worker)
        self.assertIs(self.worker._recving_thread, self.receiver)
        self.connector.shutdown()
        self.assertEqual(self.receiver.executor.shutdown.call_count, 2)
        self.assertIsNone(self.worker._recving_thread)


if __name__ == "__main__":
    unittest.main()
