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
"""Transfer-thread dispatch loop under I/O and handler failures.

A transfer thread is the only place where backend I/O errors surface. If it
keeps looping after a fatal error it will spin on a dead store; if it exits
silently the scheduler keeps handing it requests. These tests pin both ends:
the failure is recorded, the queue is not silently drained afterwards, and the
thread always releases its caller via ``ready_event``.
"""

import queue
import threading
import unittest
from typing import Any
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
)

# isort: on


class _FakeTransferThread(KVTransferThread):
    """Minimal subclass: records handled requests, no backend, no OS calls."""

    def __init__(self):
        threading.Thread.__init__(self, name="fake-transfer", daemon=True)
        self._stopping = threading.Event()
        self._request_submit_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self._fatal_error = None
        self.ready_event = threading.Event()
        self.m_store = MagicMock()
        self.handled = []
        self.cleanups = []
        self.raise_on_handle = None
        self.raise_on_cleanup = None

    def _set_os_thread_name(self):
        return None

    def _handle_request(self, request_data):
        if self.raise_on_handle is not None:
            raise self.raise_on_handle
        self.handled.append(request_data)

    def _handle_request_exception(self, request_data):
        self.cleanups.append(request_data)
        if self.raise_on_cleanup is not None:
            raise self.raise_on_cleanup


class TestTransferThreadRun(unittest.TestCase):
    def test_ready_event_is_set_before_dispatch(self):
        thread = _FakeTransferThread()
        thread.request_queue.put("r0")
        thread.request_queue.put(None)

        thread.run()

        self.assertTrue(thread.ready_event.is_set())
        thread.m_store.set_device.assert_called_once_with()
        self.assertEqual(thread.handled, ["r0"])

    def test_none_sentinel_stops_the_loop(self):
        thread = _FakeTransferThread()
        thread.request_queue.put(None)

        thread.run()

        self.assertEqual(thread.handled, [])

    def test_stopping_skips_queued_requests(self):
        thread = _FakeTransferThread()
        thread._stopping.set()
        thread.request_queue.put("r0")
        thread.request_queue.put(None)

        thread.run()

        self.assertEqual(thread.handled, [])

    def test_handler_failure_is_recorded_and_stops_the_thread(self):
        """A fatal dispatch error must not leave the thread consuming work."""
        thread = _FakeTransferThread()
        boom = RuntimeError("backend I/O failed")
        thread.raise_on_handle = boom
        thread.request_queue.put("r0")
        thread.request_queue.put("r1")

        thread.run()

        self.assertIs(thread._fatal_error, boom)
        self.assertEqual(thread.cleanups, ["r0"])
        # r1 was never dispatched: the loop returned after the failure.
        self.assertEqual(thread.handled, [])
        self.assertEqual(thread.request_queue.qsize(), 1)

    def test_cleanup_failure_does_not_escape_the_loop(self):
        thread = _FakeTransferThread()
        thread.raise_on_handle = RuntimeError("dispatch failed")
        thread.raise_on_cleanup = RuntimeError("cleanup failed")
        thread.request_queue.put("r0")

        thread.run()  # must not raise

        self.assertEqual(thread.cleanups, ["r0"])

    def test_add_request_is_rejected_after_fatal_error(self):
        thread = _FakeTransferThread()
        thread._fatal_error = RuntimeError("dead")

        thread.add_request("r0")

        self.assertEqual(thread.request_queue.qsize(), 0)

    def test_add_request_is_rejected_while_stopping(self):
        thread = _FakeTransferThread()
        thread._stopping.set()

        thread.add_request("r0")

        self.assertEqual(thread.request_queue.qsize(), 0)


class TestTransferThreadLookup(unittest.TestCase):
    def test_lookup_failure_degrades_to_all_misses(self):
        """A dead store must not raise into the caller; report misses."""
        thread = _FakeTransferThread()
        thread.m_store.exists.side_effect = RuntimeError("remote down")

        self.assertEqual(thread.lookup(["a", "b", "c"]), [False, False, False])

    def test_lookup_maps_store_flags(self):
        thread = _FakeTransferThread()
        thread.m_store.exists.return_value = [1, 0, 1]

        self.assertEqual(thread.lookup(["a", "b", "c"]), [True, False, True])


if __name__ == "__main__":
    unittest.main()
