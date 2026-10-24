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
"""Per-step GVA preparation boundaries.

``_alloc_gvas_for_save`` and ``_prepare_load_gvas`` are the two places where a
single request can disable the whole step's transfer. A regression here is
silent: blocks are simply never written or never read back. These tests pin the
batch propagation, the "one failure disables the step" rule, and the fact that
requests which were skipped are left untouched.
"""

import threading
import unittest
from typing import Any
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

# isort: on


def _request(req_id: str, can_save: bool = True, can_load: bool = True) -> MagicMock:
    request = MagicMock()
    request.req_id = req_id
    request.can_save = can_save
    request.save_keys = [f"{req_id}-key"]
    request.load_spec = MagicMock()
    request.load_spec.can_load = can_load
    return request


class _FakeWorker(KVPoolWorker):
    """Subclass with the two preparation hooks replaced by scripted fakes."""

    def __init__(self) -> None:
        self.use_layerwise_transfer = True
        self.kv_role = "kv_producer"
        self.consumer_is_to_put = False
        self.layerwise_offload = False
        self._save_disabled = threading.Event()
        self.alloc_results: list[bool | None] = []
        self.load_results: list[bool | None] = []
        self.alloc_calls = 0
        self.load_calls = 0
        self.recorded: list[Any] = []

    def _alloc_gvas_for_save_for_request(self, requests):
        self.alloc_calls += 1
        return self.alloc_results.pop(0) if self.alloc_results else None

    def _prepare_load_gvas_for_request(self, requests):
        self.load_calls += 1
        return self.load_results.pop(0) if self.load_results else None

    def _record_failed_request(self, request):
        self.recorded.append(request)


class TestAllocGvasForSave(unittest.TestCase):
    def test_noop_when_layerwise_disabled(self):
        worker = _FakeWorker()
        worker.use_layerwise_transfer = False
        requests = [_request("r0")]

        worker._alloc_gvas_for_save(requests)

        self.assertEqual(worker.alloc_calls, 0)
        self.assertEqual(requests[0].save_keys, ["r0-key"])

    def test_noop_for_consumer_that_does_not_put(self):
        worker = _FakeWorker()
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = False

        worker._alloc_gvas_for_save([_request("r0")])

        self.assertEqual(worker.alloc_calls, 0)

    def test_consumer_that_puts_allocates(self):
        worker = _FakeWorker()
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = True

        worker._alloc_gvas_for_save([_request("r0")])

        self.assertEqual(worker.alloc_calls, 1)

    def test_skips_requests_that_cannot_save(self):
        worker = _FakeWorker()
        requests = [_request("r0", can_save=False), _request("r1", can_save=True)]

        worker._alloc_gvas_for_save(requests)

        self.assertEqual(worker.alloc_calls, 1)
        self.assertFalse(worker._save_disabled.is_set())

    def test_one_failure_disables_the_whole_step(self):
        """A failed allocation must clear every pending save in the batch."""
        worker = _FakeWorker()
        worker.alloc_results = [False, None]
        requests = [_request("r0"), _request("r1"), _request("r2")]

        worker._alloc_gvas_for_save(requests)

        self.assertTrue(worker._save_disabled.is_set())
        for request in requests:
            self.assertFalse(request.can_save)
            self.assertEqual(request.save_keys, [])
        # The failing request stops the loop: r1/r2 are never attempted.
        self.assertEqual(worker.alloc_calls, 1)

    def test_disabled_flag_skips_remaining_requests(self):
        worker = _FakeWorker()
        worker._save_disabled.set()
        requests = [_request("r0"), _request("r1")]

        worker._alloc_gvas_for_save(requests)

        self.assertEqual(worker.alloc_calls, 0)
        for request in requests:
            self.assertFalse(request.can_save)
            self.assertEqual(request.save_keys, [])

    def test_offload_records_failed_requests(self):
        """Offload must remember which requests lost their save slot."""
        worker = _FakeWorker()
        worker.layerwise_offload = True
        worker.alloc_results = [False]
        requests = [_request("r0"), _request("r1")]

        worker._alloc_gvas_for_save(requests)

        self.assertEqual([request.req_id for request in worker.recorded], ["r0", "r1"])


class TestPrepareLoadGvas(unittest.TestCase):
    def test_failure_marks_request_unloadable_and_records_it(self):
        worker = _FakeWorker()
        worker.load_results = [False, True]
        requests = [_request("r0"), _request("r1")]

        worker._prepare_load_gvas(requests)

        self.assertFalse(requests[0].load_spec.can_load)
        self.assertEqual(worker.recorded, [requests[0]])
        self.assertTrue(requests[1].load_spec.can_load)

    def test_request_without_load_spec_is_not_touched(self):
        worker = _FakeWorker()
        worker.load_results = [False]
        request = _request("r0")
        request.load_spec = None

        worker._prepare_load_gvas([request])

        self.assertEqual(worker.recorded, [request])

    def test_empty_batch_is_a_noop(self):
        worker = _FakeWorker()

        worker._prepare_load_gvas([])

        self.assertEqual(worker.load_calls, 0)


if __name__ == "__main__":
    unittest.main()
