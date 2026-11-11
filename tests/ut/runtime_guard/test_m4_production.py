#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Production Store -> manager CPU batch -> token_repeat lifecycle regressions."""

import importlib
import importlib.util
import unittest
from contextlib import contextmanager
from threading import Event, RLock, Thread
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from weakref import ref

from vllm_ascend.observability.runtime_config.schema import validate_detector_section
from vllm_ascend.observability.runtime_guard.detector.base import DetectorRegistry
from vllm_ascend.observability.runtime_guard.detector.manager import AfterSampleCpuSnapshot, DetectorManager
from vllm_ascend.observability.runtime_guard.detector.token_repeat import TokenRepeatDetector
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.state import Incident, RequestGuardStore

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor


class Backend:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.bad_slot = None

    def start(self):
        return SimpleNamespace(received=0, fired_at=-1, reason="", failed=False)

    def feed_events(self, states, rows):
        self.calls.append(rows)
        events = []
        for slot, (state, row) in enumerate(zip(states, rows)):
            if slot == self.bad_slot:
                state.failed = True
                raise ValueError("invalid token in this request")
            state.received += len(row)
            if row and row[-1] == 9:
                state.fired_at, state.reason = state.received, "test reason"
                events.append((slot, state.fired_at, state.reason))
            if self.fail:
                raise ValueError("partial consumption")
        return events

    def finish(self, state):
        return SimpleNamespace(is_ill=True, fired_at=state.received, reason="tail")

    def progress(self, state):
        if state.failed:
            return None
        return state.received, state.received, state.fired_at, state.reason


class _M4Fixture(unittest.TestCase):
    def setUp(self):
        RequestGuardStore.reset_for_tests()
        self.backend = Backend()
        self.loader = patch(
            "vllm_ascend.observability.runtime_guard.detector.m4_backend.load_m4", return_value=self.backend
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

        class VersionedSection(dict):
            def __setitem__(section, key, value):
                super().__setitem__(key, value)
                self.config.config_version += 1

            def update(section, *args, **kwargs):
                super().update(*args, **kwargs)
                self.config.config_version += 1

        self.section = VersionedSection(TokenRepeatDetector.schema.defaults_dict(), enabled=True, algorithm="m4")
        self.config = SimpleNamespace(
            config_version=0,
            detector_section=lambda _: self.section,
            hot_reload_enabled=False,
            needs_cumulative_io=lambda: True,
            any_detector_enabled=lambda: True,
        )
        self.det = TokenRepeatDetector(runtime_config=self.config)
        self.manager = object.__new__(DetectorManager)
        self.manager._runner = None
        self.manager._runtime_config = self.config
        self.manager._detection_gate = None
        self.manager._token_repeat_det = self.det
        self.manager._registry = DetectorRegistry()
        self.manager._registry.register(self.det)
        self.manager._logits_finite_det = MagicMock()
        self.manager._logits_finite_det.check_deferred.return_value = []
        self.addCleanup(RequestGuardStore.reset_for_tests)

    def snapshot(self, keys, rows) -> AfterSampleCpuSnapshot:
        _, snap = self.manager.after_sample_hot_path(rows, keys)
        assert snap is not None
        return snap

    def sampled(self, keys, rows):
        return self.manager.run_after_sample_cpu(self.snapshot(keys, rows))

    def processor(self) -> "RuntimeGuardProcessor":
        from tests.ut.runtime_guard._helpers import bare_processor

        processor = bare_processor()
        processor.detectors = self.manager
        del processor._reap_finished_requests  # Exercise the real reaper.
        return processor

    def replace_method(self, obj, name, **kwargs):
        replacement = patch.object(obj, name, **kwargs)
        self.addCleanup(replacement.stop)
        return replacement.start()


class ProductionM4Tests(_M4Fixture):
    def test_batch_sparse_alerts_and_cumulative_dedup(self):
        alerts = self.sampled(["a", "b"], [[1, 9], [2]])
        self.assertEqual(
            [(a.req_id, a.detail["fired_at"], a.detail["reason"]) for a in alerts], [("a", 2, "test reason")]
        )
        self.assertEqual(self.backend.calls, [[[1, 9], [2]]])
        self.det.check_all(None, ["a", "b"])
        self.assertEqual(len(self.backend.calls), 1)
        self.sampled(["a", "b"], [[3], [4]])
        self.assertEqual(self.backend.calls[-1], [[4]])

    def test_finish_consumes_unread_suffix_and_preserves_tail_incident(self):
        RequestIoSnapshotManager.get().append_batch(["a"], [[1, 2, 3]])
        alerts = self.manager.finish_requests(["a"])
        self.assertEqual(alerts[0].detail, dict(algorithm="m4", fired_at=3, reason="tail", finish_only=True))
        self.assertEqual(self.det._consumed_len["a"], 3)
        self.assertEqual(self.manager.finish_requests(["a"]), [])

    def test_cancel_is_cleanup_and_id_reuse_is_fresh(self):
        self.sampled(["a"], [[1]])
        RequestGuardStore.get().clear("a", detectors=self.manager)
        self.assertFalse(self.det._m4_states)
        self.assertEqual(self.sampled(["a"], [[9]])[0].detail["fired_at"], 1)

    def test_disable_reenable_rejects_queued_old_snapshot(self):
        old = self.snapshot(["a"], [[1]])
        epoch = self.det.epoch
        self.section["enabled"] = False
        self.det.refresh_from_config()
        self.assertFalse(self.det._m4_states)
        self.assertIsNone(self.det._m4)
        self.assertEqual(self.det.epoch, epoch + 1)
        state = RequestGuardStore.get().get_state("a")
        assert state is not None
        state.output_token_ids.clear()
        self.section["enabled"] = True
        self.det.refresh_from_config()
        self.assertEqual(self.manager.run_after_sample_cpu(old), [])
        self.assertFalse(self.backend.calls)
        self.sampled(["a"], [[9]])
        self.assertEqual(self.backend.calls, [[[9]]])

    def test_old_snapshot_cannot_consume_reused_id(self):
        old = self.snapshot(["a"], [[1]])
        RequestGuardStore.get().clear("a", detectors=self.manager)
        self.sampled(["a"], [[2]])
        self.manager.run_after_sample_cpu(old)
        self.assertEqual(self.backend.calls, [[[2]]])

    def test_partial_exception_keeps_committed_alert_and_never_replays(self):
        self.backend.fail = True
        alerts = self.sampled(["a", "b"], [[9], [2]])
        self.assertEqual(alerts[0].detail["fired_at"], 1)
        self.assertFalse(self.det._m4_failed)
        self.backend.fail = False
        self.det.check_all(None, ["a", "b"])
        self.assertEqual(self.backend.calls, [[[9], [2]], [[2]]])
        self.assertEqual(self.det._consumed_len, {"a": 1, "b": 1})
        self.manager.finish_requests(["a", "b"])
        self.assertEqual(len(self.backend.calls), 2)

    def test_buffer_shrink_restarts_alert_state(self):
        self.sampled(["a"], [[1, 9]])
        state = RequestGuardStore.get().get_state("a")
        assert state is not None
        state.output_token_ids.clear()
        alerts = self.sampled(["a"], [[9]])
        self.assertEqual(alerts[0].detail["fired_at"], 1)

    def test_config_rejects_silent_window_knob_override(self):
        self.section["window"] = 99
        with self.assertRaisesRegex(ValueError, "only supported"):
            validate_detector_section(TokenRepeatDetector.schema, dict(self.section))
        with self.assertRaisesRegex(ValueError, "only supported"):
            self.det.refresh_from_config()

        from copy import deepcopy

        from vllm_ascend.observability.runtime_config.config import RuntimeConfig

        config = RuntimeConfig(ensure_file=False)
        previous, version = config._data, config._version
        candidate = deepcopy(previous)
        candidate["detector"]["token_repeat"].update(enabled=True, algorithm="m4", window=99)
        with self.assertRaisesRegex(ValueError, "only supported"):
            config._apply_loaded(candidate, version=version + 1, announce=False)
        self.assertIs(config._data, previous)
        self.assertEqual(config._version, version)

    def test_schema_and_old_default(self):
        self.assertEqual(TokenRepeatDetector.schema.defaults_dict()["algorithm"], "window")
        with self.assertRaises(ValueError):
            validate_detector_section(TokenRepeatDetector.schema, {"algorithm": 123})

    def test_resource_failure_is_explicit(self):
        self.loader.stop()
        from vllm_ascend.observability.runtime_guard.detector.m4_backend import load_m4

        with self.assertRaisesRegex(ValueError, "requires"):
            load_m4(None, -1)

    def test_old_completion_does_not_unpin_reused_id(self):
        store = RequestGuardStore.get()
        old = store.get_or_create("a")
        store.add_cpu_jobs(["a"])
        store.clear("a", detectors=self.manager)
        new = store.get_or_create("a")
        store.add_cpu_jobs(["a"])
        store.finish_cpu_jobs(["a"], expected_states={"a": ref(old)})
        self.assertEqual(new.cpu_jobs, 1)

    def test_processor_reap_routes_tail_before_cleanup(self):
        import importlib.util

        if importlib.util.find_spec("vllm") is None:
            self.skipTest("full vLLM runtime unavailable")
        processor = self.processor()
        self.replace_method(processor, "_current_wave", return_value=1)
        routed: list[Incident] = []

        def handle(alert, **kwargs):
            self.assertIsNotNone(RequestGuardStore.get().get_state(alert.req_id))
            routed.append(alert)

        self.replace_method(processor, "_handle_alert", side_effect=handle)
        self.sampled(["a"], [[1, 2, 3]])
        RequestGuardStore.get().mark_finished(["a"], wave=0)
        processor._reap_finished_requests()
        self.assertEqual(routed[0].detail["reason"], "tail")
        self.assertIsNone(RequestGuardStore.get().get_state("a"))
        self.assertFalse(self.det._m4_states)
        RequestIoSnapshotManager.get().append_batch(["bad-resource"], [[1]])
        RequestGuardStore.get().mark_finished(["bad-resource"], wave=0)
        with patch.object(self.manager, "finish_requests", side_effect=ValueError("invalid M4 resource")):
            processor._reap_finished_requests()
        self.assertIsNone(RequestGuardStore.get().get_state("bad-resource"))

    def test_real_queue_drains_tail_and_invalidates_force_reaped_snapshot(self):
        from vllm_ascend.observability.runtime_guard.action.queue import ActionQueue

        for forced in (False, True):
            with self.subTest(forced=forced):
                store = RequestGuardStore.get()
                processor = self.processor()
                self.replace_method(processor, "_current_wave", return_value=store.max_deferred_waves if forced else 1)
                routed: list[Incident] = []
                self.replace_method(
                    processor, "_handle_alert", side_effect=lambda alert, routed=routed, **kwargs: routed.append(alert)
                )
                queue = ActionQueue(maxsize=1)
                processor.action_executor = SimpleNamespace(action_queue=queue)
                entered, release, drained = Event(), Event(), Event()

                def block_worker(entered=entered, release=release):
                    entered.set()
                    release.wait(5)

                try:
                    self.assertTrue(queue.submit(block_worker, heavy=True))
                    self.assertTrue(entered.wait(2))
                    snap = self.snapshot(["queued"], [[1, 2]])
                    processor._enqueue_after_sample_cpu(snap, {})
                    state = store.get_state("queued")
                    assert state is not None
                    self.assertEqual(state.cpu_jobs, 1)
                    store.mark_finished(["queued"], wave=0)
                    if forced:
                        with self.assertLogs(
                            "vllm_ascend.observability.runtime_guard.processor", level="WARNING"
                        ) as logs:
                            processor._reap_finished_requests()
                        self.assertIn("forced reap skips terminal M4 check", " ".join(logs.output))
                        self.assertIsNone(store.get_state("queued"))
                    else:
                        processor._reap_finished_requests()
                        self.assertIs(store.get_state("queued"), state)
                    release.set()
                    # Queue.join waits for actual CPU detect + finally/reap, not a timing guess.
                    waiter = Thread(target=lambda queue=queue, drained=drained: (queue._queue.join(), drained.set()))
                    waiter.start()
                    self.assertTrue(drained.wait(2))
                    waiter.join(2)
                    self.assertIsNone(store.get_state("queued"))
                    self.assertFalse(self.det._m4_states)
                    self.assertEqual([a.detail["reason"] for a in routed], [] if forced else ["tail"])
                finally:
                    release.set()
                    queue.stop()

    def test_cpu_alert_delivery_rechecks_request_generation_and_config(self):
        store = RequestGuardStore.get()
        run = self.manager.run_after_sample_cpu
        for change in ("healthy", "reuse", "stop", "disable"):
            with self.subTest(change=change):
                processor = self.processor()
                snap = self.snapshot([change], [[9]])

                def change_after_detection(snapshot, change=change):
                    alerts = run(snapshot)
                    self.assertEqual(len(alerts), 1)
                    if change == "reuse":
                        store.clear(change, detectors=self.manager)
                        store.get_or_create(change)
                    elif change == "stop":
                        store.mark_detection_stopped(change)
                    elif change == "disable":
                        self.section["enabled"] = False
                        self.det.refresh_from_config()
                    return alerts

                with (
                    patch.object(self.manager, "run_after_sample_cpu", side_effect=change_after_detection),
                    patch.object(processor, "_handle_alert") as handle,
                    patch.object(processor, "_reap_finished_requests"),
                ):
                    processor._enqueue_after_sample_cpu(snap, {})
                    job = processor.action_executor.action_queue.submit.call_args.args[0]
                    job()
                    self.assertEqual(handle.call_count, int(change == "healthy"))

    def test_delivery_lock_serializes_rebind_and_config_cascade(self):
        for rebind in (False, True):
            with self.subTest(rebind=rebind):
                processor = self.processor()
                processor.detectors = MagicMock()
                processor.report_writer = MagicMock()
                attempted, finished, lock = Event(), Event(), RLock()

                @contextmanager
                def observed_lock(attempted=attempted, lock=lock):
                    attempted.set()
                    with lock:
                        yield

                self.replace_method(processor, "_reap_lock", new=observed_lock())
                runner = SimpleNamespace(ascend_config=SimpleNamespace(runtime_config=MagicMock()))
                operation = (
                    lambda processor=processor, runner=runner, rebind=rebind: processor._rebind_runner(runner)
                    if rebind
                    else processor._apply_config_cascade(prev_needs_sample_hooks=False)
                )
                action = (
                    processor.action_executor.rebind_runner
                    if rebind
                    else processor.action_executor.apply_runtime_config
                )
                detector = processor.detectors.rebind_runner if rebind else processor.detectors.apply_runtime_config
                worker = Thread(target=lambda operation=operation, finished=finished: (operation(), finished.set()))
                with lock:  # Represents an incident being delivered.
                    worker.start()
                    self.assertTrue(attempted.wait(2))
                    self.assertFalse(finished.is_set())
                    action.assert_not_called()
                    detector.assert_not_called()
                worker.join(2)
                self.assertTrue(finished.is_set())
                action.assert_called_once()
                detector.assert_called_once()

    def test_window_cleanup_compatibility(self):
        self.section.update(algorithm="window", min_tokens=0, repeat_sum_threshold=0)
        self.det.refresh_from_config()
        self.sampled(["a"], [[1, 1]])
        self.assertTrue(self.det._states)
        RequestGuardStore.get().clear("a", detectors=self.manager)
        self.assertFalse(self.det._states)
        self.assertFalse(self.det._consumed_len)

    def test_old_cleanup_does_not_clear_new_native_state(self):
        self.sampled(["a"], [[1]])
        old = RequestGuardStore.get().get_state("a")
        RequestGuardStore.get().clear("a", detectors=self.manager)
        self.sampled(["a"], [[2]])
        native = self.det._m4_states["a"]
        self.manager.clear_finished("a", expected_state=old)
        self.assertIs(self.det._m4_states["a"], native)

    def test_finish_exception_keeps_earlier_tail_alerts(self):
        self.sampled(["a", "b"], [[1], [1, 2]])
        finish = self.backend.finish

        def finish_one(state):
            if state.received == 2:
                raise ValueError("finish failed")
            return finish(state)

        with patch.object(self.backend, "finish", side_effect=finish_one):
            alerts = self.manager.finish_requests(["a", "b"])
        self.assertEqual([a.req_id for a in alerts], ["a"])
        self.assertIn("b", self.det._m4_failed)

    def test_skipped_request_without_native_state_is_fully_cleaned(self):
        store = RequestGuardStore.get()
        RequestIoSnapshotManager.get().append_batch(["skipped"], [[1, 2]])
        self.det.check_all(None, ["skipped"], {"skipped"})
        self.assertNotIn("skipped", self.det._m4_states)
        self.assertEqual(self.det._consumed_len["skipped"], 2)
        store.clear("skipped", detectors=self.manager)
        self.assertNotIn("skipped", self.det._consumed_len)
        self.assertNotIn("skipped", self.det._m4_owners)

    def test_reenable_does_not_replay_output_retained_for_other_detectors(self):
        self.sampled(["a"], [[1, 2]])
        self.section["enabled"] = False
        self.det.refresh_from_config()
        RequestIoSnapshotManager.get().append_batch(["a"], [[3]])
        self.section["enabled"] = True
        self.det.refresh_from_config()
        self.backend.fail = True
        alerts = self.sampled(["a"], [[9]])
        self.assertEqual(self.backend.calls[-1], [[9]])
        self.assertEqual(alerts[0].detail["fired_at"], 4)
        self.assertEqual(self.det._consumed_len["a"], 4)

    def test_queued_snapshot_does_not_retain_reaped_token_buffers(self):
        snap = self.snapshot(["a"], [[1] * 1024])
        old = ref(RequestGuardStore.get().get_state("a"))
        RequestGuardStore.get().clear("a", detectors=self.manager)
        self.assertIsNone(old())
        self.assertEqual(self.manager.run_after_sample_cpu(snap), [])
        self.assertNotIn("a", self.det._m4_owners)

    def test_late_enqueue_cannot_recreate_or_pin_a_reused_request(self):
        store = RequestGuardStore.get()
        snap = self.snapshot(["a"], [[1]])
        store.clear("a", detectors=self.manager)
        store.add_cpu_jobs(["a"], expected_states=snap.request_states)
        self.assertIsNone(store.get_state("a"))
        state = store.get_or_create("a")
        store.add_cpu_jobs(["a"], expected_states=snap.request_states)
        self.assertEqual(state.cpu_jobs, 0)

    def test_resource_change_starts_at_current_output(self):
        self.sampled(["a"], [[1, 2]])
        self.section["m4_think_end"] = 17
        self.det.refresh_from_config()
        alerts = self.sampled(["a"], [[9]])
        self.assertEqual(self.backend.calls[-1], [[9]])
        self.assertEqual(alerts[0].detail["fired_at"], 3)

    def test_stop_after_snapshot_prevents_late_detection(self):
        snap = self.snapshot(["a"], [[9]])
        RequestGuardStore.get().mark_detection_stopped("a")
        self.assertEqual(self.manager.run_after_sample_cpu(snap), [])
        self.assertEqual(self.backend.calls, [])

    def test_failed_slot_does_not_disable_untouched_or_completed_requests(self):
        self.backend.bad_slot = 1
        self.sampled(["a", "bad", "c"], [[1], [2], [3]])
        self.assertEqual(self.det._m4_failed, {"bad"})
        self.backend.bad_slot = None
        alerts = self.sampled(["a", "bad", "c"], [[9], [9], [9]])
        self.assertEqual(self.backend.calls[-1], [[9], [3, 9]])
        self.assertEqual([(a.req_id, a.detail["fired_at"]) for a in alerts], [("a", 2), ("c", 2)])

    def test_finish_drains_healthy_slots_after_batch_failure(self):
        self.backend.bad_slot = 0
        RequestIoSnapshotManager.get().append_batch(["bad", "good"], [[1], [2, 3]])
        feed = self.backend.feed_events

        def fail_once(states, rows):
            try:
                return feed(states, rows)
            finally:
                self.backend.bad_slot = None

        with patch.object(self.backend, "feed_events", side_effect=fail_once):
            alerts = self.manager.finish_requests(["bad", "good"])
        self.assertEqual([(a.req_id, a.detail["fired_at"]) for a in alerts], [("good", 2)])
        self.assertEqual(self.det._m4_failed, {"bad"})

    def test_invalid_config_keeps_previous_enabled_algorithm_and_state(self):
        self.sampled(["a"], [[1]])
        state, epoch = self.det._m4_states["a"], self.det.epoch
        self.section.update(enabled=False, algorithm="invalid")
        with self.assertRaises(ValueError):
            self.det.refresh_from_config()
        self.assertTrue(self.det.enabled)
        self.assertEqual(self.det._algorithm, "m4")
        self.assertEqual(self.det.epoch, epoch)
        self.assertIs(self.det._m4_states["a"], state)

    def test_native_owner_does_not_pin_removed_output_buffer(self):
        self.sampled(["a"], [[1] * 1024])
        owner = ref(RequestGuardStore.get().get_state("a"))
        RequestGuardStore.get().clear("a")  # Even missing detector cleanup cannot retain Store IO.
        self.assertIsNone(owner())
        self.sampled(["a"], [[9]])
        self.assertEqual(self.det._consumed_len["a"], 1)

    def test_warmup_only_on_enable_and_model_rebind(self):
        self.sampled(
            ["a"],
            [
                [1],
            ],
        )
        self.det.check_all(None, ["a"])
        self.det.refresh_from_config()
        self.loader.target.load_m4.assert_called_once()
        self.det.rebind_runner(None)
        self.assertEqual(self.loader.target.load_m4.call_count, 2)
        self.loader.target.load_m4.side_effect = RuntimeError("extension unavailable")
        with self.assertLogs("vllm_ascend.observability.runtime_guard.detector.token_repeat", level="ERROR"):
            self.det.rebind_runner(None)
        self.det.refresh_from_config()
        self.assertEqual(self.loader.target.load_m4.call_count, 3)
        self.assertIsNone(self.det._m4)


class ResourceValidationTests(unittest.TestCase):
    def test_warmup_builds_only_from_served_tokenizer(self):
        import sys
        import tempfile
        from pathlib import Path
        from types import ModuleType

        import numpy as np

        from vllm_ascend.observability.runtime_guard.detector.m4_backend import load_m4

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tokenizer.json"
            source.write_text("{}")
            runner = SimpleNamespace(model_config=SimpleNamespace(tokenizer=tmp))
            tables = {"norm": np.arange(16)}
            adapter, vocab = ModuleType("m4_cpp"), ModuleType("m4_vocab")
            constructor, build = MagicMock(return_value="loaded"), MagicMock(return_value=tables)
            vars(adapter)["RepDetectorCPP"] = constructor
            vars(vocab)["build"] = build
            with patch.dict(sys.modules, {"m4_cpp": adapter, "m4_vocab": vocab}):
                self.assertEqual(load_m4(runner, -1), "loaded")
                build.assert_called_once_with(str(source))
                constructor.assert_called_once_with(tables, think_end=None)
                with self.assertRaisesRegex(ValueError, "outside"):
                    load_m4(runner, 16)
                with self.assertRaisesRegex(ValueError, "think_end"):
                    load_m4(runner, True)

    def test_rebind_uses_new_runtime_config_even_when_versions_match(self):
        """A new runner must not inherit the previous model's detector configuration."""
        from vllm_ascend.observability.runtime_config.config import RuntimeConfig

        RequestGuardStore.reset_for_tests()
        configs = [
            RuntimeConfig(
                ensure_file=False,
                startup_overlay={
                    "detector": {
                        "token_repeat": {"enabled": True, "algorithm": "m4", "m4_think_end": think_end},
                        "spec_acceptance": {"enabled": bool(think_end)},
                    }
                },
            )
            for think_end in (0, 7)
        ]
        assert configs[0]._version == configs[1]._version
        runners = [SimpleNamespace(ascend_config=SimpleNamespace(runtime_config=cfg)) for cfg in configs]
        try:
            with (
                patch("vllm_ascend.observability.runtime_guard.detector.m4_backend.load_m4") as load,
                patch(
                    "vllm_ascend.observability.runtime_guard.rank_gate.should_run_anomaly_check_on_rank",
                    return_value=True,
                ),
            ):
                manager = DetectorManager(runtime_config=configs[0], runner=runners[0])
                initial_epoch = manager._token_repeat_det.epoch
                manager.rebind_runner(runners[1])
                assert manager._runtime_config is configs[1]
                assert all(det._runtime_config is configs[1] for det in manager._registry)
                assert manager._token_repeat_det._m4_think_end == 7
                assert manager._spec_det.enabled
                assert manager._token_repeat_det.epoch > initial_epoch
                assert load.call_count == 2
                assert load.call_args.args == (runners[1], 7)
        finally:
            RequestGuardStore.reset_for_tests()


class NativeM4Tests(_M4Fixture):
    def setUp(self):
        import numpy as np

        super().setUp()
        if importlib.util.find_spec("_m4_native") is None:
            self.skipTest("independent native extension not built")
        self.native = importlib.import_module("m4_cpp").RepDetectorCPP
        self.tables: dict[str, np.ndarray] = dict(
            norm=np.arange(256, dtype=np.int32),
            chars=np.ones(256, np.int16),
            ws=np.zeros(256, np.int16),
            run_char=np.full(256, -1, np.int32),
            marker=np.zeros(256, bool),
            send=np.zeros(256, bool),
        )

    def test_native_block_boundaries_tail_and_first_reason(self):
        for length in (0, 63, 64, 65, 95, 96, 127, 128, 129):
            for width in (1, 17, 64, 129):
                self.det.on_disabled()
                RequestGuardStore.reset_for_tests()
                self.det._m4 = self.native(self.tables, think_end=None)
                tokens = (list(range(32)) * 5)[:length]
                alerts = []
                for offset in range(0, length, width):
                    alerts.extend(self.sampled(["native"], [tokens[offset : offset + width]]))
                alerts.extend(self.manager.finish_requests(["native"]))
                actual = [(a.detail["fired_at"], a.detail["reason"], a.detail["finish_only"]) for a in alerts]
                expected = [] if length < 96 else [(min(length, 128), "vllm[32,2000]x3", length < 128)]
                self.assertEqual(actual, expected, (length, width))

    def test_native_bad_slot_preserves_other_requests(self):
        self.det._m4 = self.native(self.tables)
        self.sampled(["before", "bad", "after"], [[1], [2, 256], list(range(32)) * 3])
        self.assertEqual(self.det._m4_failed, {"bad"})
        self.assertEqual(self.det._consumed_len["before"], 1)
        self.assertEqual(self.det._consumed_len.get("after", 0), 0)
        alerts = self.manager.finish_requests(["before", "bad", "after"])
        self.assertEqual(
            [(a.req_id, a.detail["fired_at"], a.detail["reason"]) for a in alerts], [("after", 96, "vllm[32,2000]x3")]
        )

    def test_native_preflight_rejects_invalid_batch_before_consumption(self):
        backend, foreign = self.native(self.tables), self.native(self.tables)
        for size in (1, 2, 128, 129):
            for invalid in ("duplicate", "foreign", "null", "length"):
                if invalid == "duplicate" and size == 1:
                    continue
                with self.subTest(size=size, invalid=invalid):
                    original = [backend.start() for _ in range(size)]
                    states, rows = list(original), [[1] for _ in original]
                    if invalid == "duplicate":
                        states[-1] = states[0]
                    elif invalid == "foreign":
                        states[-1] = foreign.start()
                    elif invalid == "null":
                        states[-1] = None
                    else:
                        rows.pop()
                    with self.assertRaises((ValueError, RuntimeError, TypeError)):
                        backend.feed_events(states, rows)
                    self.assertEqual([backend.progress(state)[0] for state in original], [0] * size)
