# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.distributed.kv_events import BlockStored

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cpu_events import (
    CPU_EVENT_RETENTION_SECONDS,
    CPUBlockEventTracker,
    CPUBlockStoreReport,
)


class TestCPUBlockEventTracker(unittest.TestCase):
    def report(self, completed, required=("A/head0", "A/head1"), request_id="r1", newly_stored=True):
        return CPUBlockStoreReport(
            event=BlockStored(
                block_hashes=[1],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=2,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            ),
            request_id=request_id,
            required_keys=frozenset(required),
            completed_keys=frozenset(completed),
            newly_stored=newly_stored,
        )

    def test_missing_partition_waits_across_batches_and_ignores_duplicates(self):
        tracker = CPUBlockEventTracker()
        first = self.report(["A/head0"])
        second = self.report(["A/head1"])
        self.assertIsNone(tracker.add(first))
        self.assertIsNone(tracker.add(first))
        self.assertIs(tracker.add(second), first.event)
        self.assertIsNone(tracker.add(first))
        self.assertIsNone(tracker.add(second))

    def test_disjoint_complete_blocks_publish_independently(self):
        tracker = CPUBlockEventTracker()
        for key in ("A", "B"):
            report = self.report([key], required=[key])
            self.assertIs(tracker.add(report), report.event)

    def test_existing_keys_can_complete_a_partially_new_block_in_either_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                tracker = CPUBlockEventTracker()
                reports = [self.report(["A/head0"], newly_stored=False), self.report(["A/head1"])]
                if reverse:
                    reports.reverse()
                self.assertIsNone(tracker.add(reports[0]))
                self.assertIs(tracker.add(reports[1]), reports[0].event)

    def test_lookup_hits_alone_do_not_publish(self):
        tracker = CPUBlockEventTracker()
        self.assertIsNone(tracker.add(self.report(["A/head0"], newly_stored=False)))
        self.assertIsNone(tracker.add(self.report(["A/head1"], newly_stored=False)))

    def test_requests_and_physical_key_sets_do_not_share_evidence(self):
        tracker = CPUBlockEventTracker()
        self.assertIsNone(tracker.add(self.report(["A/head0"])))
        self.assertIsNone(tracker.add(self.report(["A/head1"], request_id="r2")))
        self.assertIsNone(tracker.add(self.report(["A/head1"], required=["A/head0", "A/head1", "A/head2"])))

    def test_invalid_evidence_is_not_published(self):
        tracker = CPUBlockEventTracker()
        for completed, required in (([], ["A"]), (["A"], []), (["A", "other"], ["A"])):
            with self.subTest(completed=completed, required=required):
                self.assertIsNone(tracker.add(self.report(completed, required)))

    def test_expiry_discards_incomplete_evidence(self):
        tracker = CPUBlockEventTracker()
        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cpu_events"
        with patch(f"{module}.monotonic", return_value=0):
            self.assertIsNone(tracker.add(self.report(["A/head0"])))
        with patch(f"{module}.monotonic", return_value=CPU_EVENT_RETENTION_SECONDS):
            self.assertIsNone(tracker.add(self.report(["A/head1"])))

    def test_capacity_discards_oldest_incomplete_evidence(self):
        tracker = CPUBlockEventTracker()
        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cpu_events"
        with patch(f"{module}.MAX_TRACKED_CPU_EVENTS", 1):
            self.assertIsNone(tracker.add(self.report(["A/head0"])))
            self.assertIsNone(tracker.add(self.report(["A/head0"], request_id="r2")))
            self.assertIsNone(tracker.add(self.report(["A/head1"])))

    def test_clear_prevents_old_evidence_from_completing_after_reset(self):
        tracker = CPUBlockEventTracker()
        self.assertIsNone(tracker.add(self.report(["A/head0"])))
        tracker.clear()
        self.assertIsNone(tracker.add(self.report(["A/head1"])))

    def test_a_new_request_can_publish_a_restored_block(self):
        tracker = CPUBlockEventTracker()
        for request in ("first", "restored"):
            report = self.report(["A"], ["A"], request_id=request)
            self.assertIs(tracker.add(report), report.event)
