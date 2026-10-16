# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regressions for observational timing, cleanup, and the separate diagnostic pass."""

import ast
import json
import os
import shlex
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.e2e.pull_request.one_card import ascendstore_v1_layerwise_timing as timing_module
from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise import AscendStoreLayerwiseWorkerExtension
from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise_timing import LayerwiseLoadTiming, _Interval, _union_ns


def _make_timing(*, execute_error=None):
    result = object()
    backend = SimpleNamespace(
        exists=Mock(return_value=[True]),
        start_key_range_load=Mock(return_value=(0,)),
        copy_key_range_load=Mock(return_value=(0,)),
        finish_key_range_load=Mock(),
    )
    timeline = SimpleNamespace(_prepare_layers=Mock())
    timeline._operation = lambda layer_id: backend.copy_key_range_load(["key"], [[100]], [[16]], [[0]])
    timeline._execute_layer = lambda job: timeline._operation(job.layer_id)
    worker = SimpleNamespace(
        _layerwise_load_timeline=timeline, _layer_ids_by_name={"layer.0": 0}, _build_load_batch=Mock()
    )

    def start_load():
        worker._build_load_batch(())
        backend.start_key_range_load(["key"])
        timeline._prepare_layers(None)

    def wait_for_layer_load(layer_name):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(timeline._execute_layer, SimpleNamespace(layer_id=0)).result(timeout=5)
        backend.finish_key_range_load(["key"])

    def execute_model(scheduler_output):
        if scheduler_output.kv_connector_metadata.load.commands:
            worker.start_load()
            worker.wait_for_layer_load("layer.0")
        if execute_error is not None:
            raise execute_error
        return result

    worker.start_load = start_load
    worker.wait_for_layer_load = wait_for_layer_load
    runner = SimpleNamespace(execute_model=execute_model, sample_tokens=Mock(return_value=result))
    lookup = SimpleNamespace(_lookup=lambda request: backend.exists(["key"]))
    timing = LayerwiseLoadTiming(worker, backend, lookup, runner)
    return timing, worker, backend, lookup, runner, result


def _schedule(*request_ids):
    commands = tuple(SimpleNamespace(request_id=request) for request in request_ids)
    return SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(load=SimpleNamespace(commands=commands)),
        num_scheduled_tokens={request: 1 for request in request_ids},
        total_num_scheduled_tokens=len(request_ids),
    )


def test_timing_preserves_results_and_captures_executor_layers_but_not_decode():
    timing, worker, backend, lookup, runner, result = _make_timing()
    owners = (worker, backend, lookup, runner, worker._layerwise_load_timeline)
    originals = [dict(vars(owner)) for owner in owners]
    timing.install()
    try:
        assert lookup._lookup(None) == [True]
        backend.exists(["store-presence-check"])
        assert runner.execute_model(_schedule("request-0")) is result
        assert runner.sample_tokens(None) is result
        count = timing.report()["record_count"]
        assert runner.execute_model(scheduler_output=_schedule()) is result
        assert runner.sample_tokens(None) is result
        assert timing.report()["record_count"] == count
    finally:
        timing.restore()
    assert [dict(vars(owner)) for owner in owners] == originals
    report = timing.report()
    assert report["dropped_records"] == report["failed_calls"] == 0
    assert report["stages"]["lookup_backend_exists"]["calls"] == 1
    assert report["stages"]["range_copy"]["calls"] == 1
    assert report["steps"][0]["request_ids"] == ["request-0"]
    assert "lookup_service" not in report["steps"][0]["stages"]
    assert report["layers"][0]["layer"] == 0
    assert report["layers"][0]["stages"]["range_copy"]["calls"] == 1
    backend.copy_key_range_load.assert_called_once_with(["key"], [[100]], [[16]], [[0]])


def test_timing_preserves_the_exact_exception_and_restores_inherited_methods():
    error = RuntimeError("copy failed")
    timing, worker, _, _, runner, _ = _make_timing(execute_error=error)
    original = worker.start_load

    class InheritedWorker:
        def start_load(self):
            return original()

    inherited = InheritedWorker()
    # Exercise an inherited descriptor independently from the simulated model callback.
    timing._patch(inherited, "start_load", "inherited", independent=True)
    timing.install()
    try:
        assert "start_load" in vars(inherited)
        with pytest.raises(RuntimeError) as raised:
            runner.execute_model(_schedule("request-0"))
        assert raised.value is error
        assert timing.report()["failed_calls"] == 1
    finally:
        timing.restore()
    assert "start_load" not in vars(inherited)
    assert timing._active_step is None


def test_partial_install_rolls_back_every_successful_hook():
    timing, worker, backend, lookup, runner, _ = _make_timing()
    del backend.copy_key_range_load
    owners = (worker, backend, lookup, runner, worker._layerwise_load_timeline)
    originals = [dict(vars(owner)) for owner in owners]
    with pytest.raises(AttributeError):
        timing.install()
    assert [dict(vars(owner)) for owner in owners] == originals


def test_report_uses_interval_unions_instead_of_adding_overlapping_stages():
    timing, *_ = _make_timing()
    timing._steps = [{"step": 0, "request_ids": ["r"]}]
    timing._records = [
        _Interval(stage, 0, None, start * 1_000_000, end * 1_000_000, False)
        for stage, start, end in (
            ("execute_model_host", 0, 100),
            ("load_start_host", 10, 30),
            ("layer_wait_host", 50, 90),
            ("range_copy", 20, 60),
            ("range_copy", 55, 80),
            ("sample_tokens_host", 120, 130),
        )
    ]
    step = timing.report()["steps"][0]
    assert step["foreground_load_ms"] == 60
    assert step["range_copy_union_ms"] == 60
    assert step["copy_overlaps_foreground_ms"] == 40
    assert step["execute_outside_load_calls_ms"] == 40
    assert step["execute_to_sample_gap_ms"] == 20
    assert _union_ns([(1, 10), (2, 3), (12, 15)]) == 12


def test_record_limit_does_not_change_calls_and_reports_dropped_intervals(monkeypatch):
    timing, _, _, lookup, runner, result = _make_timing()
    monkeypatch.setattr(timing_module, "_MAX_RECORDS", 2)
    timing.install()
    try:
        lookup._lookup(None)
        assert runner.execute_model(_schedule("request-0")) is result
        assert runner.sample_tokens(None) is result
    finally:
        timing.restore()
    report = timing.report()
    assert report["record_count"] == 2
    assert report["dropped_records"] > 0
    assert report["steps"][0]["execute_outside_load_calls_ms"] is None


def test_worker_extension_declares_runner_type_without_overriding_worker_owned_instance():
    assert AscendStoreLayerwiseWorkerExtension.__annotations__["model_runner"] == "NPUModelRunner"
    assert not hasattr(AscendStoreLayerwiseWorkerExtension(), "model_runner")


def test_stop_restores_hooks_even_if_out_of_timing_store_drain_fails(monkeypatch):
    extension = AscendStoreLayerwiseWorkerExtension()
    timing = Mock()
    extension._layerwise_load_timing = timing
    monkeypatch.setattr(extension, "_drain_stores", Mock(side_effect=RuntimeError("drain failed")))
    with pytest.raises(RuntimeError, match="drain failed"):
        extension.stop_ascendstore_layerwise_load_timing()
    timing.restore.assert_called_once_with()
    timing.report.assert_not_called()
    assert not hasattr(extension, "_layerwise_load_timing")


@pytest.mark.parametrize("failure", [None, "benchmark", "sources", "stop", "benchmark-and-stop", "coverage"])
def test_diagnostic_is_separate_and_always_stops_without_masking_request_errors(failure):
    source = Path(__file__).resolve().parents[3] / "tests/e2e/pull_request/one_card/test_ascendstore_v1_layerwise.py"
    entry = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_diagnose_external_load"
    )
    timing, _, _, lookup, runner, _ = _make_timing()
    timing.install()
    try:
        lookup._lookup(None)
        runner.execute_model(_schedule("request-0"))
        runner.sample_tokens(None)
    finally:
        timing.restore()
    report = timing.report()
    if failure == "coverage":
        report["stages"].pop("range_copy")
    calls = []

    def rpc(server, method):
        calls.append(method)
        if method == "stop_ascendstore_layerwise_load_timing":
            if failure in ("stop", "benchmark-and-stop"):
                raise RuntimeError("stop failed")
            return {"workers": [report]}
        return {}

    def benchmark(*args):
        calls.append("benchmark")
        if failure in ("benchmark", "benchmark-and-stop"):
            raise ValueError("request failed")
        return {"mean_ttft_ms": 1000}

    def assert_sources(measurement, *, external):
        assert external
        calls.append("assert_sources")
        if failure == "sources":
            raise AssertionError("wrong prompt sources")

    namespace: dict[str, Any] = {
        "Any": Any,
        "call_worker_rpc": rpc,
        "reset_local_prefix_cache": lambda _: calls.append("reset"),
        "run_benchmark": benchmark,
        "assert_prompt_sources": assert_sources,
    }
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
    measurements = [{"external_load": {"mean_ttft_ms": 900}}]
    run: dict[str, Any] = {"measurements": measurements, "external_load_diagnostic": {}}
    if failure is None:
        namespace[entry.name]("server", tokenizer=None, inputs=[None], run=run)
        assert run["external_load_diagnostic"]["reference_mean_ttft_ms"] == 900
    else:
        messages = {
            "benchmark": "request failed",
            "benchmark-and-stop": "request failed",
            "sources": "wrong prompt sources",
            "stop": "stop failed",
            "coverage": "Missing Load timing",
        }
        with pytest.raises((ValueError, RuntimeError, AssertionError), match=messages[failure]):
            namespace[entry.name]("server", tokenizer=None, inputs=[None], run=run)
    assert calls[:3] == ["drain_ascendstore_benchmark_stores", "reset", "start_ascendstore_layerwise_load_timing"]
    assert calls[-1] == "stop_ascendstore_layerwise_load_timing"
    assert run["measurements"] == [{"external_load": {"mean_ttft_ms": 900}}]


@pytest.mark.parametrize("implementation", ["production", "v1"])
def test_start_rpc_only_installs_on_v1_and_double_start_does_not_replace_hooks(monkeypatch, implementation):
    extension = AscendStoreLayerwiseWorkerExtension()
    _, worker, backend, lookup, runner, _ = _make_timing()
    transfer = ModuleType("vllm.distributed.kv_transfer")
    monkeypatch.setattr(transfer, "get_kv_transfer_group", lambda: SimpleNamespace(lookup_server=lookup), raising=False)
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)
    monkeypatch.setattr(extension, "_layerwise_resources", lambda: (worker, backend, implementation))
    monkeypatch.setattr(extension, "model_runner", runner, raising=False)
    if implementation == "production":
        original = runner.execute_model
        with pytest.raises(AssertionError, match="must not instrument production"):
            extension.start_ascendstore_layerwise_load_timing()
        assert runner.execute_model is original
        assert not hasattr(extension, "_layerwise_load_timing")
    else:
        assert extension.start_ascendstore_layerwise_load_timing()["status"] == "installed"
        installed = extension._layerwise_load_timing
        try:
            with pytest.raises(AssertionError, match="already active"):
                extension.start_ascendstore_layerwise_load_timing()
            assert extension._layerwise_load_timing is installed
            monkeypatch.setattr(extension, "_drain_stores", lambda: None)
            assert extension.stop_ascendstore_layerwise_load_timing()["record_count"] == 0
        finally:
            installed.restore()


@pytest.mark.parametrize("with_diagnostic", [False, True])
def test_shared_serving_helper_runs_callback_after_all_rounds_before_server_exit(tmp_path, with_diagnostic):
    source = Path(__file__).resolve().parents[3] / "tests/e2e/pull_request/one_card/test_ascendstore_v1.py"
    entry = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_exercise_connector"
    )
    events = []
    server = object()

    @contextmanager
    def server_context(*args, **kwargs):
        events.append("server_enter")
        yield server
        events.append("server_exit")

    @contextmanager
    def pool_context(*args, **kwargs):
        yield SimpleNamespace(server_envs={}, master_address="127.0.0.1:1234")

    def benchmark(server, tokenizer, inputs):
        events.append("warmup" if inputs == ["warmup"] else "benchmark")
        return {}

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "Callable": Callable,
        "TokenizerLike": Any,
        "SampleRequest": Any,
        "WORKER_EXTENSION": "extension",
        "RemoteOpenAIServer": server_context,
        "get_open_ports_list": lambda _: (1, 2, 3, 4),
        "MooncakeKVPoolConfig": lambda **kwargs: kwargs,
        "POOL_MEMORY_BYTES": 1,
        "LOCAL_BUFFER_BYTES": 1,
        "MODEL": "model",
        "CONCURRENCY": 8,
        "BLOCK_SIZE": 128,
        "ROUNDS_PER_RUN": 2,
        "shlex": shlex,
        "json": json,
        "os": os,
        "__file__": str(source),
        "import_module": lambda _: SimpleNamespace(SingleNodeMooncakeManager=pool_context),
        "run_benchmark": benchmark,
        "call_worker_rpc": lambda *args, **kwargs: {"workers": [{"removed_objects": 1}]},
        "reset_local_prefix_cache": lambda _: {},
        "assert_prompt_sources": lambda *args, **kwargs: None,
    }
    # The real server class also appears in an annotation; postpone annotation evaluation for this boundary double.
    entry.returns = None
    entry.args.kwonlyargs[-1].annotation = None
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
    observations: list[dict[str, Any]] = []
    callback = lambda received: events.append("diagnostic") if received is server else pytest.fail()
    namespace[entry.name](
        tmp_path,
        "model",
        None,
        ["inputs"],
        ["warmup"],
        ("v1", "v1", "module"),
        observations,
        after_measurements=callback if with_diagnostic else None,
    )
    expected = ["server_enter", "warmup", "benchmark", "benchmark", "benchmark", "benchmark"]
    assert events == expected + (["diagnostic"] if with_diagnostic else []) + ["server_exit"]
    assert len(observations) == 2


def test_timing_code_never_touches_device_data_or_synchronization():
    source = Path(timing_module.__file__).read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) and "torch" in ast.unparse(node) for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("synchronize", "item", "cpu", "kv_caches")
        for node in ast.walk(tree)
    )
