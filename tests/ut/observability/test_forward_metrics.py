# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


def load_source(name, filename):
    path = Path(__file__).parents[3] / "vllm_ascend" / "observability" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    fake_vllm = ModuleType("vllm")
    fake_logger = ModuleType("vllm.logger")
    fake_logger.logger = logging.getLogger("vllm")
    with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.logger": fake_logger}):
        spec.loader.exec_module(module)
    return module


metrics = load_source("test_forward_metrics_source", "forward_metrics.py")
ForwardTimeCollector = metrics.ForwardTimeCollector
run_timed_forward = metrics.run_timed_forward


class Event:
    def __init__(self, factory):
        self.factory = factory
        self.ready = False
        self.error = None

    def record(self):
        if self.factory.record_error:
            raise self.factory.record_error
        self.ready = self.factory.immediate

    def query(self):
        if self.error:
            raise self.error
        return self.ready

    def elapsed_time(self, end):
        assert end.ready, "elapsed_time was read before end event completed"
        return self.factory.elapsed

    def synchronize(self):
        raise AssertionError("synchronize is forbidden")

    def wait_event(self, *args):
        raise AssertionError("wait_event is forbidden")


class EventFactory:
    def __init__(self, immediate=False):
        self.events = []
        self.immediate = immediate
        self.elapsed = 1.25
        self.record_error = None
        self.create_error = None

    def __call__(self):
        if self.create_error:
            raise self.create_error
        event = Event(self)
        self.events.append(event)
        return event

    def complete(self):
        for event in self.events:
            event.ready = True


class CollectorTests(unittest.TestCase):
    def make(self, window=100, **kwargs):
        self.events, self.logs = EventFactory(), []
        self.collector = ForwardTimeCollector(
            rank=2, window_size=window, event_factory=self.events, emit_log=self.logs.append, **kwargs
        )
        return self.collector

    def sample(self, role="target", phase="decode", bs=4):
        handle = self.collector.start(role, phase, bs)
        self.collector.finish(handle)
        return handle

    def test_async_read_waits_for_query(self):
        self.make()
        self.sample()
        self.collector.drain_ready()
        self.assertEqual(self.collector.pending_count, 1)
        self.events.complete()
        rows = self.collector.flush_window(final=True)
        self.assertEqual((rows[0].count, rows[0].avg_ms), (1, 1.25))

    def test_capacity_includes_active(self):
        self.make(window=2)
        first = self.collector.start("target", "decode", 4)
        self.sample()
        self.assertIsNone(self.collector.start("draft", "decode", 4))
        self.assertLessEqual(self.collector.active_count + self.collector.pending_count, 2)
        self.collector.finish(first)
        self.events.complete()
        rows = self.collector.flush_window(final=True)
        self.assertEqual(sum(row.count for row in rows), 2)
        self.assertEqual(sum(row.dropped for row in rows), 1)

    def test_window_attribution_survives_async_boundary(self):
        self.make(window=2)
        self.sample(bs=4)
        self.sample(bs=5)
        self.events.complete()
        self.sample("draft", "decode", 6)
        self.events.complete()
        rows = self.collector.flush_window(final=True)
        self.assertEqual(
            [(r.window_id, r.key.batch_size) for r in sorted(rows, key=lambda r: r.key.batch_size)],
            [(1, 4), (1, 5), (2, 6)],
        )

    def test_pool_recycles_and_stays_bounded(self):
        self.make(window=3)
        for _ in range(30):
            self.sample()
            self.events.complete()
            self.collector.drain_ready()
        self.assertLessEqual(len(self.events.events), 6)
        self.assertLessEqual(self.collector.pool_size, 6)

    def test_filter_and_disabled_allocate_no_events(self):
        self.make(target_batch_sizes={5})
        self.assertIsNone(self.sample(bs=4))
        self.assertEqual(self.events.events, [])
        self.make(enabled=False)
        self.assertIsNone(self.sample())
        self.assertEqual(self.events.events, [])

    def test_tail_drops_unfinished_and_logs_error_only_keys(self):
        self.make()
        self.sample()
        rows = self.collector.flush_window(final=True)
        self.assertEqual((rows[0].count, rows[0].dropped), (0, 1))
        self.assertIn("count=0", self.logs[0])
        self.assertEqual(self.collector.flush_window(final=True), [])
        self.assertIsNone(self.collector.start("target", "decode", 4))

    def test_record_failure_is_logged_without_raising(self):
        self.make()
        self.events.record_error = RuntimeError("record")
        self.assertIsNone(self.sample())
        rows = self.collector.flush_window(final=True)
        self.assertEqual(rows[0].errors, 1)
        self.assertIn("errors=1", self.logs[0])

    def test_query_failure_discards_events(self):
        self.make()
        self.sample()
        self.events.events[-1].error = RuntimeError("query")
        rows = self.collector.flush_window(final=True)
        self.assertEqual(rows[0].errors, 1)
        self.assertEqual(self.collector.pool_size, 0)

    def test_invalid_elapsed_values(self):
        for value in (-1.0, math.nan, math.inf):
            with self.subTest(value=value):
                self.make()
                self.events.elapsed = value
                self.sample()
                self.events.complete()
                rows = self.collector.flush_window(final=True)
                self.assertEqual((rows[0].count, rows[0].errors), (0, 1))

    def test_changed_stream_is_dropped_without_end_event(self):
        current = [1]
        self.make(stream_getter=lambda: current[0])
        handle = self.collector.start("target", "decode", 4)
        current[0] = 2
        self.collector.finish(handle)
        rows = self.collector.flush_window(final=True)
        self.assertEqual((rows[0].count, rows[0].dropped), (0, 1))
        self.assertEqual(len(self.events.events), 1)

    def test_closed_abort_window_emits_immediately(self):
        self.make(window=1)
        handle = self.collector.start("target", "decode", 4)
        self.collector.abort(handle)
        self.assertEqual(self.collector.live_window_count, 0)
        self.assertIn("dropped=1", self.logs[0])

    def test_log_failure_does_not_lose_summary(self):
        self.make()
        self.collector._emit_log = MagicMock(side_effect=RuntimeError("logging"))
        self.sample()
        self.events.complete()
        self.assertEqual(self.collector.flush_window(final=True)[0].count, 1)

    def test_wrapper_failures_never_retry_forward(self):
        for operation in ("phase", "start", "finish"):
            with self.subTest(operation=operation):
                collector = MagicMock()
                phase = MagicMock(return_value="decode")
                if operation == "phase":
                    phase.side_effect = RuntimeError("phase")
                else:
                    getattr(collector, operation).side_effect = RuntimeError(operation)
                forward = MagicMock(return_value=object())
                self.assertIs(run_timed_forward(collector, "target", phase, 4, forward), forward.return_value)
                forward.assert_called_once()

    def test_model_exception_identity_survives_abort_failure(self):
        collector = MagicMock()
        collector.abort.side_effect = RuntimeError("abort")
        error = RuntimeError("model")
        forward = MagicMock(side_effect=error)
        with self.assertRaises(RuntimeError) as raised:
            run_timed_forward(collector, "target", lambda: "decode", 4, forward)
        self.assertIs(raised.exception, error)
        forward.assert_called_once()


class SettlementRegressionTests(unittest.TestCase):
    def make(self):
        self.events = EventFactory()
        return ForwardTimeCollector(0, event_factory=self.events, emit_log=lambda _: None)

    def test_finish_abort_and_repeated_settlement_are_idempotent(self):
        for first in ("finish", "abort"):
            for second in ("finish", "abort"):
                with self.subTest(first=first, second=second):
                    c = self.make()
                    handle = c.start("target", "decode", 1)
                    getattr(c, first)(handle)
                    getattr(c, second)(handle)
                    self.events.complete()
                    rows = c.flush_window(final=True)
                    self.assertEqual(c.active_count, 0)
                    self.assertEqual(c.pending_count, 0)
                    self.assertEqual(sum(row.count for row in rows), int(first == "finish"))
                    self.assertEqual(sum(row.dropped for row in rows), int(first == "abort"))

    def test_final_flush_accounts_for_active_and_pending_samples(self):
        c = self.make()
        active = c.start("target", "decode", 1)
        pending = c.start("draft", "decode", 1)
        c.finish(pending)
        rows = c.flush_window(final=True)
        self.assertEqual(sum(row.dropped for row in rows), 2)
        self.assertEqual(c.active_count, 0)
        self.assertEqual(c.pending_count, 0)
        c.finish(active)
        c.abort(active)
        self.assertEqual(c.flush_window(final=True), [])
        self.assertIsNone(c.start("target", "decode", 1))

    def test_foreign_handle_cannot_settle_another_collector(self):
        c = self.make()
        other = self.make()
        handle = c.start("target", "decode", 1)
        other.finish(handle)
        other.abort(handle)
        self.assertEqual(c.active_count, 1)
        self.assertEqual(other.active_count, 0)
        c.flush_window(final=True)

    def test_parallel_identity_is_in_each_summary_log(self):
        logs = []
        c = ForwardTimeCollector(
            1,
            event_factory=EventFactory(immediate=True),
            emit_log=logs.append,
            labels={"dp_rank": 7, "tp_rank": 1, "pp_rank": 0},
        )
        c.finish(c.start("target", "verify", 1))
        c.flush_window(final=True)
        self.assertIn("dp_rank=7", logs[0])
        self.assertIn("tp_rank=1", logs[0])
        self.assertIn("pp_rank=0", logs[0])


class ForwardConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_module = load_source("test_forward_metrics_config_source", "forward_metrics_config.py")

    def test_default_is_disabled_and_config_does_not_import_npu(self):
        with patch.dict(sys.modules, {"torch": None, "torch_npu": None}):
            c = self.config_module.ForwardTimeMetricsConfig.from_mapping(None)
            self.assertFalse(c.enabled)
            c = self.config_module.ForwardTimeMetricsConfig.from_mapping(
                {"enabled": True, "window_size": 10, "target_batch_sizes": [1, 4], "collect_draft": False}
            )
            self.assertEqual(c.target_batch_sizes, frozenset({1, 4}))
            self.assertFalse(c.collect_draft)

    def test_invalid_configuration_is_rejected(self):
        for value in [
            [],
            True,
            {"enabled": "true"},
            {"window_size": 0},
            {"window_size": True},
            {"window_size": 1.5},
            {"target_batch_sizes": [False]},
            {"target_batch_sizes": [0]},
            {"target_batch_sizes": "1"},
            {"collect_draft": 1},
            {"typo": True},
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.config_module.ForwardTimeMetricsConfig.from_mapping(value)

    def test_phase_mapping_preserves_speculative_semantics(self):
        from types import SimpleNamespace

        phase = self.config_module.forward_phase
        self.assertEqual(phase(SimpleNamespace(name="PrefillNoCache")), "prefill")
        self.assertEqual(phase(SimpleNamespace(name="ChunkedPrefill")), "mixed")
        self.assertEqual(phase(SimpleNamespace(name="SpecDecoding")), "decode")
        self.assertEqual(phase(SimpleNamespace(name="SpecDecoding"), True), "verify")
        self.assertEqual(phase(SimpleNamespace(name="SpecDecoding"), True, draft=True), "decode")
        self.assertEqual(phase(None), "mixed")


class NativeEntryTests(unittest.TestCase):
    """Execute the actual integration blocks without importing the NPU runner."""

    def function(self, path, name):
        root = Path(__file__).parents[3]
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    def compile_block(self, arguments, statements, result, globals_dict):
        function = ast.parse(f"def entry({arguments}):\n    pass").body[0]
        function.body = statements + ast.parse(f"return {result}").body
        tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(tree, "native-forward-timing-entry", "exec"), globals_dict)
        return globals_dict["entry"]

    def test_target_entry_preserves_output_and_classifies_verify(self):
        from types import SimpleNamespace

        cfg = load_source("test_native_forward_config", "forward_metrics_config.py")
        function = self.function("vllm_ascend/worker/model_runner_v1.py", "execute_model")
        block = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "collector is None"
        )
        entry = self.compile_block(
            "self, scheduler_output, collector, num_tokens_padded=1, input_ids=None, positions=None, "
            "intermediate_tensors=None, inputs_embeds=None, model_kwargs=None",
            [block],
            "hidden_states",
            {"run_timed_forward": run_timed_forward, "forward_phase": cfg.forward_phase},
        )
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                events = EventFactory(immediate=True)
                c = ForwardTimeCollector(0, event_factory=events, emit_log=lambda _: None) if enabled else None
                runner = SimpleNamespace(
                    attn_state=SimpleNamespace(name="SpecDecoding"),
                    input_batch=SimpleNamespace(num_reqs=1),
                    _model_forward=MagicMock(return_value=object()),
                )
                scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"req": [1, 2, 3]})
                result = entry(runner, scheduler, c, model_kwargs={})
                self.assertIs(result, runner._model_forward.return_value)
                runner._model_forward.assert_called_once()
                if c is not None:
                    rows = c.flush_window(final=True)
                    self.assertEqual((rows[0].count, rows[0].key.phase, rows[0].key.batch_size), (1, "verify", 1))
                else:
                    self.assertEqual(events.events, [])

    def test_draft_entry_counts_graph_replay_without_replacing_the_graph(self):
        from collections.abc import Callable
        from functools import partial
        from types import SimpleNamespace
        from typing import cast

        cfg = load_source("test_native_draft_config", "forward_metrics_config.py")
        function = self.function("vllm_ascend/spec_decode/llm_base_proposer.py", "_propose")
        context = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.With)
            and any(
                isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "runnable" for t in s.targets)
                for s in n.body
            )
        )
        index = next(
            i
            for i, n in enumerate(context.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "runnable" for t in n.targets)
        )
        entry = self.compile_block(
            "self, scheduler_output, model_inputs, forward_context=None, num_input_tokens=1, "
            "multi_steps_attn_metadata=None",
            context.body[index:],
            "draft_token_ids",
            {
                "run_timed_forward": run_timed_forward,
                "forward_phase": cfg.forward_phase,
                "partial": partial,
                "cast": cast,
                "Callable": Callable,
                "Any": object,
            },
        )

        class Graph:
            def __init__(self):
                self.body_calls = 0
                self.result = None

            def __call__(self):
                if self.result is None:
                    self.body_calls += 1
                    self.result = object()
                return self.result

        for enpu in [False, True]:
            with self.subTest(enpu=enpu):
                c = ForwardTimeCollector(0, event_factory=EventFactory(immediate=True), emit_log=lambda _: None)
                runner = SimpleNamespace(
                    forward_time_collector=c,
                    attn_state=SimpleNamespace(name="SpecDecoding"),
                    input_batch=SimpleNamespace(num_reqs=1),
                )
                graph = Graph()
                proposer = SimpleNamespace(
                    runner=runner, _runnable=graph, enable_enpu=enpu, _update_full_graph_params_if_needed=MagicMock()
                )
                # A capture/dummy call has no scheduler output and is excluded.
                entry(proposer, None, {})
                self.assertEqual(c.pending_count, 0)
                self.assertIs(entry(proposer, object(), {}), graph.result)
                self.assertIs(entry(proposer, object(), {}), graph.result)
                rows = c.flush_window(final=True)
                self.assertEqual(sum(row.count for row in rows), 2)
                self.assertEqual(graph.body_calls, 1)
                self.assertIs(proposer._runnable, graph)

    def test_worker_shutdown_keeps_cleanup_when_metrics_flush_fails(self):
        from types import SimpleNamespace

        function = self.function("vllm_ascend/worker/worker.py", "shutdown")
        runner = SimpleNamespace(forward_time_collector=MagicMock(), shutdown=MagicMock())
        runner.forward_time_collector.flush_window.side_effect = RuntimeError("event query failed")
        entry = self.compile_block(
            "self",
            function.body,
            "None",
            {
                "AscendEplbState": type("AscendEplbState", (), {}),
                "ensure_kv_transfer_shutdown": None,
                "ensure_ec_transfer_shutdown": None,
                "logger": MagicMock(),
            },
        )
        entry(SimpleNamespace(model_runner=runner, profiler=None))
        runner.shutdown.assert_called_once()

    def test_enabled_unsupported_runners_and_drafters_are_rejected(self):
        from types import SimpleNamespace

        function = self.function("vllm_ascend/ascend_config.py", "derive_and_validate")
        validation = function.body[1]
        # The source method first assigns vc and then validates this feature.
        entry = self.compile_block("self, vc", [validation], "None", {})
        for v2, method, collect_draft, enabled, rejected in [
            (True, "mtp", True, True, True),
            (False, "ngram", True, True, True),
            (False, "ngram", False, True, False),
            (False, "mtp", True, True, False),
            (True, "mtp", True, False, False),
        ]:
            cfg = SimpleNamespace(forward_time_metrics_config={"enabled": enabled, "collect_draft": collect_draft})
            vc = SimpleNamespace(use_v2_model_runner=v2, speculative_config=SimpleNamespace(method=method))
            with self.subTest(v2=v2, method=method, enabled=enabled):
                if rejected:
                    with self.assertRaises(ValueError):
                        entry(cfg, vc)
                else:
                    entry(cfg, vc)


class LogSummaryTests(unittest.TestCase):
    def test_weighting_parallel_filter_and_zero_sample_errors(self):
        import tempfile

        path = Path(__file__).parents[3] / "tools" / "forward_metrics_summary.py"
        spec = importlib.util.spec_from_file_location("forward_metrics_summary", path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        prefix = "(Worker_DP3_TP0 pid=50) Forward timing: window=1 rank=0 "
        lines = [
            prefix + "role=target phase=verify bs=1 count=2 avg_ms=10 min_ms=9 max_ms=11 dropped=0 errors=0",
            prefix + "role=target phase=verify bs=1 count=1 avg_ms=4 min_ms=4 max_ms=4 dropped=0 errors=0",
            prefix.replace("TP0", "TP1").replace("rank=0", "rank=1")
            + "role=target phase=verify bs=1 count=99 avg_ms=99 min_ms=99 max_ms=99 dropped=0 errors=0",
            prefix + "role=draft phase=decode bs=1 count=0 avg_ms=0 min_ms=0 max_ms=0 dropped=1 errors=1",
        ]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            log.write_text("\n".join(lines), encoding="utf-8")
            rows = tool.summarize([log])
            target = next(row for row in rows if row["role"] == "target")
            self.assertEqual((target["samples"], target["avg_ms"], target["dp_rank"]), (3, 8.0, 3))
            self.assertEqual(target["min_ms"], 4)
            draft = next(row for row in rows if row["role"] == "draft")
            self.assertIsNone(draft["avg_ms"])
            self.assertNotIn("total_ms", draft)
            self.assertEqual((draft["errors"], draft["dropped"]), (1, 1))
            log.write_text("Forward timing: window=1", encoding="utf-8")
            with self.assertRaises(ValueError):
                tool.summarize([log])
