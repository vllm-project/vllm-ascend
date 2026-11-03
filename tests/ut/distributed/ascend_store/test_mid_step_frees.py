# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the mid-step block-release bridge (running-request fence).

The coordinator frees blocks of RUNNING requests inside schedule()
(sliding-window advancement, R-SWA gap eviction); those frees must reach the
AscendStore released-block fence, or a queued store put can read a block the
scheduler already reallocated. These tests pin the bridge mechanics; the
end-to-end drain behavior is covered by test_request_lifecycle.py running
the real patched coordinator with hybrid full-and-swa layouts.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import mid_step_frees
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _ascend_remove_skipped_blocks


class FakeManager:
    """Single-type manager double: frees the first ``free_count`` blocks."""

    def __init__(self, blocks, free_count=0):
        self.req_to_blocks = {"r1": list(blocks)}
        self.calls = []
        self.free_count = free_count

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        self.calls.append((request_id, processed_computed_tokens, num_prompt_tokens))
        blocks = self.req_to_blocks.get(request_id, [])
        for i, block in enumerate(blocks):
            if i < self.free_count:
                block.is_null = True


def make_coordinator(managers, cacheable_flags):
    coordinator = KVCacheCoordinator.__new__(KVCacheCoordinator)
    coordinator.single_type_managers = tuple(managers)
    groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_cacheable=flag)) for flag in cacheable_flags]
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    return coordinator


class TestMidStepReleaseBridge(unittest.TestCase):
    def setUp(self):
        mid_step_frees.activate_mid_step_release_tracking()
        # Drain anything an earlier test left behind and start clean.
        mid_step_frees.take_mid_step_released_req_ids()
        self.addCleanup(mid_step_frees.take_mid_step_released_req_ids)

    def test_records_request_when_pool_saved_manager_frees(self):
        manager = FakeManager([SimpleNamespace(is_null=False) for _ in range(3)], free_count=2)
        coordinator = make_coordinator([manager], [True])

        _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        self.assertEqual(manager.calls, [("r1", 256, None)])
        self.assertIn("r1", mid_step_frees.take_mid_step_released_req_ids())

    def test_no_record_when_nothing_freed(self):
        manager = FakeManager([SimpleNamespace(is_null=False) for _ in range(3)], free_count=0)
        coordinator = make_coordinator([manager], [True])

        _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

    def test_no_record_for_non_cacheable_manager(self):
        # Private-state groups are never saved to the pool, so their frees
        # must not widen the fence (doing so would reintroduce a coarse
        # fence for models whose private blocks retire every step).
        manager = FakeManager([SimpleNamespace(is_null=False) for _ in range(3)], free_count=3)
        coordinator = make_coordinator([manager], [False])

        _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

    def test_mixed_managers_record_only_on_pool_saved_free(self):
        pool_saved = FakeManager([SimpleNamespace(is_null=False)], free_count=0)
        private = FakeManager([SimpleNamespace(is_null=False)], free_count=1)
        coordinator = make_coordinator([pool_saved, private], [True, False])

        _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

        pool_saved.free_count = 1
        _ascend_remove_skipped_blocks(coordinator, "r1", 512)
        self.assertIn("r1", mid_step_frees.take_mid_step_released_req_ids())

    def test_no_record_when_request_has_no_blocks(self):
        manager = FakeManager([], free_count=3)
        manager.req_to_blocks = {}
        coordinator = make_coordinator([manager], [True])

        # Must not crash and must not record anything.
        _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

    def test_passthrough_when_not_tracking(self):
        manager = FakeManager([SimpleNamespace(is_null=False) for _ in range(3)], free_count=3)
        coordinator = make_coordinator([manager], [True])

        with mock.patch.object(mid_step_frees, "is_tracking", return_value=False):
            _ascend_remove_skipped_blocks(coordinator, "r1", 256)

        # The upstream base method ran: the manager was called, but nothing
        # was recorded (no pool scheduling in this process).
        self.assertEqual(len(manager.calls), 1)
        self.assertTrue(all(block.is_null for block in manager.req_to_blocks["r1"]))
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

    def test_registry_take_clears(self):
        mid_step_frees.record_mid_step_release("r1")
        mid_step_frees.record_mid_step_release("r2")
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), {"r1", "r2"})
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())

    def test_record_is_noop_when_not_tracking(self):
        with mock.patch.object(mid_step_frees, "_tracking", False):
            mid_step_frees.record_mid_step_release("r1")
        self.assertEqual(mid_step_frees.take_mid_step_released_req_ids(), set())


if __name__ == "__main__":
    unittest.main()
