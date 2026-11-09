# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from tools.kv_pool.analyze_recv_trace import summarize
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.recv_trace import RecvTimeline, RecvTraceSession


class PublishedTimeline(RecvTimeline):
    def enqueue(self, *args, **kwargs):
        task_id = super().enqueue(*args, **kwargs)
        self.publish(task_id)
        return task_id


class TestRecvAnalysis(unittest.TestCase):
    def test_timeout_retries_are_accumulated_per_task(self):
        events = [
            {"task_id": 0, "ts_ns": timestamp, "event": event}
            for timestamp, event in (
                (0, "gate_record_wait_begin"),
                (100000000, "gate_record_wait_timeout"),
                (100000000, "gate_record_wait_begin"),
                (150000000, "gate_record_wait_end"),
            )
        ]
        result = summarize({"schema_version": 1, "dropped": 0, "events": events})
        self.assertEqual(result["stage_distributions"]["gate_record_wait"]["p50_ms"], 150)
        self.assertEqual(result["stage_distributions"]["gate_record_wait"]["count"], 1)
        self.assertEqual(result["span_distributions"]["gate_record_wait"]["count"], 2)

    def test_queue_and_backend_have_separate_distributions(self):
        events = [
            {"task_id": 0, "ts_ns": timestamp, "event": event}
            for timestamp, event in (
                (0, "enqueue"),
                (1000000, "dequeue"),
                (2000000, "backend_submit"),
                (5000000, "backend_complete"),
            )
        ]
        result = summarize({"schema_version": 1, "dropped": 0, "events": events})
        self.assertEqual(result["stage_distributions"]["queue"]["p50_ms"], 1)
        self.assertEqual(result["stage_distributions"]["backend"]["p50_ms"], 3)
        self.assertEqual(result["incomplete_spans"], 0)

    def test_hol_span_does_not_cross_wait_boundary(self):
        sample = {
            "task_id": 0,
            "event": "readiness_sample",
            "dependencies": [[1, {"gate": True}]],
            "ready_followers": [1],
        }
        events = [
            {**sample, "ts_ns": 0},
            {"task_id": 0, "event": "save_cpu_wait_end", "ts_ns": 1000000},
            {**sample, "ts_ns": 9000000},
        ]
        result = summarize({"schema_version": 1, "dropped": 0, "events": events})
        self.assertEqual(result["hol_positive_observations"], 2)
        self.assertEqual(result["hol_observed_span_ms"], 0)

    def test_unknown_native_metric_is_not_reported_as_zero(self):
        event = {
            "task_id": -1,
            "event": "resources",
            "ts_ns": 0,
            "host_peak_rss_bytes": 123,
            "backend_queue_depth": None,
        }
        result = summarize({"schema_version": 1, "dropped": 0, "events": [event]})
        self.assertEqual(result["resource_peaks"], {"host_peak_rss_bytes": 123})


class TestRecvTimeline(unittest.TestCase):
    def test_empty_payload_is_not_a_blocking_dependency(self):
        trace = PublishedTimeline()
        head = trace.enqueue(lambda: {"gate": True, "has_payload": False})
        trace.enqueue(lambda: {"gate": True, "has_payload": True})
        trace.dequeue(head)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [])

    def test_ready_follower_is_evidence(self):
        trace = PublishedTimeline()
        head = trace.enqueue(lambda: {"save_cpu": False, "gate": True})
        follower = trace.enqueue(lambda: {"save_cpu": True, "gate": True})
        trace.dequeue(head)
        trace.sample()
        row = trace.snapshot()["events"][-1]
        self.assertEqual(row["ready_followers"], [follower])
        self.assertFalse(row["head_recheck"]["save_cpu"])

    def test_blocked_and_unknown_followers_are_not_hol(self):
        trace = PublishedTimeline()
        head = trace.enqueue(lambda: {"save_cpu": False})
        trace.enqueue(lambda: {"gate": False})
        trace.enqueue(lambda: {"external": None})
        trace.dequeue(head)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [])

    def test_head_recheck_discards_stale_blockage(self):
        trace = PublishedTimeline()
        ready = False
        head = trace.enqueue(lambda: {"save_cpu": ready})

        def follower():
            nonlocal ready
            ready = True
            return {"gate": True}

        trace.enqueue(follower)
        trace.dequeue(head)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [])

    def test_dispatch_change_discards_observation(self):
        trace = PublishedTimeline()
        head = trace.enqueue(lambda: {"save_cpu": False})

        def follower():
            trace.finish(head)
            return {"gate": True}

        trace.enqueue(follower)
        trace.dequeue(head)
        trace.sample()
        self.assertFalse(any(row["event"] == "readiness_sample" for row in trace.snapshot()["events"]))

    def test_capacity_bounds_records_and_pending_callbacks(self):
        trace = PublishedTimeline(capacity=2)
        first = trace.enqueue(lambda: {"gate": False})
        trace.enqueue(lambda: {"gate": True})
        self.assertIsNone(trace.enqueue(lambda: {"gate": True}))
        trace.dequeue(first)
        trace.sample()
        trace.finish(first)
        self.assertEqual(len(trace.snapshot()["events"]), 2)
        self.assertGreater(trace.snapshot()["dropped"], 0)
        self.assertLessEqual(len(trace._pending), 2)

    def test_probe_failure_is_not_readiness(self):
        trace = PublishedTimeline()
        head = trace.enqueue(lambda: {"gate": False})

        def broken():
            raise RuntimeError("event destroyed")

        trace.enqueue(broken)
        trace.dequeue(head)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [])
        self.assertTrue(any(row["event"] == "probe_error" for row in trace.snapshot()["events"]))

    def test_ids_are_not_reused(self):
        trace = PublishedTimeline()
        first = trace.enqueue(lambda: {"gate": True})
        trace.dequeue(first)
        trace.finish(first)
        second = trace.enqueue(lambda: {"gate": True})
        self.assertNotEqual(first, second)

    def test_invalid_capacity(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RecvTimeline(value)

    def test_unpublished_task_cannot_be_hol_evidence(self):
        trace = RecvTimeline()
        head = trace.enqueue(lambda: {"gate": False})
        trace.publish(head)
        follower = trace.enqueue(lambda: {"gate": True})
        trace.dequeue(head)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [])
        trace.publish(follower)
        trace.sample()
        self.assertEqual(trace.snapshot()["events"][-1]["ready_followers"], [follower])

    def test_late_publish_does_not_retain_completed_task(self):
        trace = RecvTimeline()
        task = trace.enqueue(lambda: {"gate": True})
        trace.dequeue(task)
        trace.finish(task)
        trace.publish(task)
        self.assertEqual(len(trace._pending), 0)


class TestRecvTraceSession(unittest.TestCase):
    def test_live_snapshot_survives_without_shutdown_callback(self):
        flushed = threading.Event()
        original = RecvTraceSession._flush

        def flush(session):
            original(session)
            flushed.set()

        worker = SimpleNamespace(m_store=SimpleNamespace(set_device=lambda: None))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.recv_trace.TRACE_FLUSH_INTERVAL_S",
                0.01,
            ),
            patch.object(RecvTraceSession, "_flush", flush),
        ):
            path = Path(directory) / "trace.json"
            session = RecvTraceSession(worker, str(path), 100, 1)
            try:
                self.assertTrue(flushed.wait(2))
                self.assertTrue(path.exists())
                self.assertFalse(Path(str(path) + ".tmp").exists())
            finally:
                session.close()

    def test_failed_group_without_active_keys_is_not_ready(self):
        worker = SimpleNamespace(
            m_store=SimpleNamespace(set_device=lambda: None),
            load_disabled=threading.Event(),
            name="range",
            _group_load_indices={0: set()},
        )
        item = SimpleNamespace(
            shared_block_data=SimpleNamespace(block_ids_arr=[1]),
            group_id=0,
            block_ranges=[],
            use_key_major_ranges=True,
        )
        task = SimpleNamespace(wait_for_save_layer=None, attention_start_gate=None, layer_id=0, transfer_tasks=[item])
        with tempfile.TemporaryDirectory() as directory:
            session = RecvTraceSession(worker, str(Path(directory) / "trace.json"), 100, 1000)
            try:
                session.enqueue(task)
                session.publish(task)
                session.dequeue(task)
                session.timeline.sample()
                self.assertFalse(session.timeline.snapshot()["events"][-1]["head_recheck"]["has_payload"])
                worker._group_load_indices[0] = {0}
                session.timeline.sample()
                self.assertTrue(session.timeline.snapshot()["events"][-1]["head_recheck"]["has_payload"])
                session.finish()
            finally:
                session.close()

    def test_save_event_is_not_queried_before_record_and_wait_is_latched(self):
        calls = []
        event = threading.Event()
        worker = SimpleNamespace(
            m_store=SimpleNamespace(set_device=lambda: None),
            layer_save_finished_events=[event],
            sync_save_events=[SimpleNamespace(query=lambda: calls.append("query") or True)],
            load_disabled=threading.Event(),
            name="gva",
        )
        task = SimpleNamespace(wait_for_save_layer=0, attention_start_gate=None, layer_id=1, transfer_tasks=[])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "trace.json")
            session = RecvTraceSession(worker, path, 100, 1000)
            try:
                session.enqueue(task)
                session.publish(task)
                session.dequeue(task)
                session.timeline.sample()
                self.assertEqual(calls, [])
                event.set()
                session.mark("save_cpu_wait_end")
                session.mark("save_npu_wait_end")
                event.clear()
                session.timeline.sample()
                sample = session.timeline.snapshot()["events"][-1]
                self.assertTrue(sample["head_recheck"]["save_cpu"])
                self.assertTrue(sample["head_recheck"]["save_npu"])
                session.finish()
            finally:
                session.close()
            self.assertTrue(Path(path).exists())
            self.assertFalse(session._observer.is_alive())

    def test_external_waiter_is_never_executed_by_probe(self):
        def forbidden(_):
            raise AssertionError("observer invoked blocking callback")

        worker = SimpleNamespace(
            m_store=SimpleNamespace(set_device=lambda: None),
            external_slot_release_waiter=forbidden,
            load_disabled=threading.Event(),
            name="gva",
        )
        task = SimpleNamespace(wait_for_save_layer=None, attention_start_gate=None, layer_id=0, transfer_tasks=[])
        with tempfile.TemporaryDirectory() as directory:
            session = RecvTraceSession(worker, str(Path(directory) / "trace.json"), 100, 1000)
            try:
                session.enqueue(task)
                session.publish(task)
                session.dequeue(task)
                session.timeline.sample()
                self.assertIsNone(session.timeline.snapshot()["events"][-1]["head_recheck"]["external_slot"])
                session.finish()
            finally:
                session.close()


class TestAttentionGateTrace(unittest.TestCase):
    def test_split_wait_does_not_add_synchronization(self):
        calls = []
        gate = AttentionComputeStartGate()
        gate._event = cast(
            Any, SimpleNamespace(synchronize=lambda: calls.append("sync"), query=lambda: calls.append("query") or True)
        )
        marks = []
        self.assertTrue(gate.wait(observer=marks.append))
        self.assertEqual(calls, ["sync"])
        self.assertEqual(
            marks, ["gate_record_wait_begin", "gate_record_wait_end", "gate_npu_wait_begin", "gate_npu_wait_end"]
        )
        self.assertEqual(gate.readiness(), (True, True))
        self.assertEqual(calls, ["sync", "query"])

    def test_unrecorded_gate_query_does_not_create_an_event(self):
        gate = AttentionComputeStartGate()
        self.assertEqual(gate.readiness(), (False, False))
        marks = []
        self.assertFalse(gate.wait(timeout=0, observer=marks.append))
        self.assertEqual(marks, ["gate_record_wait_begin", "gate_record_wait_timeout"])
        self.assertIsNone(gate._event)
