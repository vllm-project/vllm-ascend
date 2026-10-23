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
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

# isort: on


def _request(req_id: str, can_save=True, can_load=True) -> MagicMock:
    request = MagicMock()
    request.req_id = req_id
    request.can_save = can_save
    request.save_keys = [f"{req_id}-key"]
    request.load_spec = MagicMock()
    request.load_spec.can_load = can_load
    return request


class TestAllocGvasForSave(unittest.TestCase):
    def _worker(self) -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.use_layerwise_transfer = True
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.layerwise_offload = False
        worker._save_disabled = threading.Event()
        worker._record_failed_request = MagicMock()
        worker._alloc_gvas_for_save_for_request = MagicMock(return_value=None)
        return worker

    def test_noop_when_layerwise_disabled(self):
        worker = self._worker()
        worker.use_layerwise_transfer = False
        requests = [_request("r0")]

        worker._alloc_gvas_for_save(requests)

        worker._alloc_gvas_for_save_for_request.assert_not_called()
        self.assertEqual(requests[0].save_keys, ["r0-key"])

    def test_noop_for_consumer_that_does_not_put(self):
        worker = self._worker()
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = False

        worker._alloc_gvas_for_save([_request("r0")])

        worker._alloc_gvas_for_save_for_request.assert_not_called()

    def test_consumer_that_puts_allocates(self):
        worker = self._worker()
        worker.kv_role = "kv_consumer"
        worker.consumer_is_to_put = True

        worker._alloc_gvas_for_save([_request("r0")])

        worker._alloc_gvas_for_save_for_request.assert_called_once()

    def test_skips_requests_that_cannot_save(self):
        worker = self._worker()
        requests = [_request("r0", can_save=False), _request("r1", can_save=True)]

        worker._alloc_gvas_for_save(requests)

        self.assertEqual(worker._alloc_gvas_for_save_for_request.call_count, 1)
        self.assertFalse(worker._save_disabled.is_set())

    def test_one_failure_disables_the_whole_step(self):
        """A failed allocation must clear every pending save in the batch."""
        worker = self._worker()
        worker._alloc_gvas_for_save_for_request.side_effect = [False, None]
        requests = [_request("r0"), _request("r1"), _request("r2")]

        worker._alloc_gvas_for_save(requests)

        self.assertTrue(worker._save_disabled.is_set())
        for request in requests:
            self.assertFalse(request.can_save)
            self.assertEqual(request.save_keys, [])
        # The failing request stops the loop: r1/r2 are never attempted.
        self.assertEqual(worker._alloc_gvas_for_save_for_request.call_count, 1)

    def test_disabled_flag_skips_remaining_requests(self):
        worker = self._worker()
        worker._save_disabled.set()
        requests = [_request("r0"), _request("r1")]

        worker._alloc_gvas_for_save(requests)

        worker._alloc_gvas_for_save_for_request.assert_not_called()
        for request in requests:
            self.assertFalse(request.can_save)
            self.assertEqual(request.save_keys, [])

    def test_offload_records_failed_requests(self):
        """Offload must remember which requests lost their save slot."""
        worker = self._worker()
        worker.layerwise_offload = True
        worker._alloc_gvas_for_save_for_request.return_value = False
        requests = [_request("r0"), _request("r1")]

        worker._alloc_gvas_for_save(requests)

        recorded = [c.args[0].req_id for c in worker._record_failed_request.call_args_list]
        self.assertEqual(recorded, ["r0", "r1"])


class TestPrepareLoadGvas(unittest.TestCase):
    def _worker(self) -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._prepare_load_gvas_for_request = MagicMock(return_value=None)
        worker._record_failed_request = MagicMock()
        return worker

    def test_failure_marks_request_unloadable_and_records_it(self):
        worker = self._worker()
        worker._prepare_load_gvas_for_request.side_effect = [False, True]
        requests = [_request("r0"), _request("r1")]

        worker._prepare_load_gvas(requests)

        self.assertFalse(requests[0].load_spec.can_load)
        worker._record_failed_request.assert_called_once_with(requests[0])
        self.assertTrue(requests[1].load_spec.can_load)

    def test_request_without_load_spec_is_not_touched(self):
        worker = self._worker()
        worker._prepare_load_gvas_for_request.return_value = False
        request = _request("r0")
        request.load_spec = None

        worker._prepare_load_gvas([request])

        worker._record_failed_request.assert_called_once_with(request)

    def test_empty_batch_is_a_noop(self):
        worker = self._worker()

        worker._prepare_load_gvas([])

        worker._prepare_load_gvas_for_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
