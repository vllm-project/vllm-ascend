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
    body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=body or [ast.Pass()],
        decorator_list=[],
    )
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


if __name__ == "__main__":
    unittest.main()
