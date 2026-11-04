# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regressions for observational timing, comparison phases, and cleanup."""

import ast
import json
import os
import shlex
import sys
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

    def submit_task(operation, *args):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(operation, *args)

    executor = SimpleNamespace()
    transfer = SimpleNamespace(
        _open_write=Mock(side_effect=lambda *args: backend.start_key_range_load(["key"])),
        _write_layer=lambda layer_id: backend.copy_key_range_load(["key"], [[100]], [[16]], [[0]]),
        executor=executor,
    )
    worker = SimpleNamespace(
        transfer=transfer,
        context=SimpleNamespace(layer_ids_by_name={"layer.0": 0}),
        load=SimpleNamespace(_prepare=Mock()),
    )

    def start_load():
        worker.load._prepare(())
        worker.transfer._open_write(None)

    def run_layer(control, layer_id, gate):
        return worker.transfer._write_layer(layer_id)

    executor._write_layer = run_layer

    def wait_for_layer_load(layer_name):
        submit_task(executor._write_layer, None, 0, None).result(timeout=5)
        submit_task(lambda: backend.finish_key_range_load(["key"])).result(timeout=5)

    def execute_model(scheduler_output):
        if scheduler_output.kv_connector_metadata.load_commands:
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
        kv_connector_metadata=SimpleNamespace(load_commands=commands),
        num_scheduled_tokens={request: 1 for request in request_ids},
        total_num_scheduled_tokens=len(request_ids),
    )


def test_timing_preserves_results_and_captures_executor_layers_but_not_decode():
    timing, worker, backend, lookup, runner, result = _make_timing()
    owners = (worker, worker.load, worker.transfer, worker.transfer.executor, backend, lookup, runner)
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


def test_timing_does_not_accumulate_executor_wrappers_between_steps():
    timing, worker, _, _, runner, result = _make_timing()
    original = worker.transfer.executor._write_layer
    timing.install()
    installed = len(timing._restorations)
    observed = worker.transfer.executor._write_layer
    try:
        for index in range(3):
            assert runner.execute_model(_schedule(f"request-{index}")) is result
            assert worker.transfer.executor._write_layer is observed
            assert len(timing._restorations) == installed
        assert timing.report()["stages"]["layer_task_host"]["calls"] == 3
    finally:
        timing.restore()
    assert worker.transfer.executor._write_layer is original


def test_partial_install_rolls_back_every_successful_hook():
    timing, worker, backend, lookup, runner, _ = _make_timing()
    del backend.copy_key_range_load
    owners = (worker, worker.load, worker.transfer, worker.transfer.executor, backend, lookup, runner)
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


@pytest.mark.parametrize("mode", ["accuracy", "performance", "request_failure"])
def test_shared_serving_helper_separates_accuracy_and_performance_and_cleans_up(tmp_path, mode):
    source = Path(__file__).resolve().parents[3] / "tests/e2e/pull_request/one_card/test_ascendstore_v1.py"
    entry = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_exercise_connector"
    )
    events, sources = [], []
    server = object()

    @contextmanager
    def server_context(*args, **kwargs):
        events.append("server_enter")
        try:
            yield server
        finally:
            events.append("server_exit")

    @contextmanager
    def pool_context(*args, **kwargs):
        try:
            yield SimpleNamespace(server_envs={}, master_address="127.0.0.1:1234")
        finally:
            events.append("pool_exit")

    def benchmark(received, tokenizer, inputs):
        assert received is server
        events.append("warmup" if inputs == ["warmup"] else "benchmark")
        if mode == "request_failure" and inputs == ["inputs"]:
            raise ValueError("request failed")
        return {}

    def rpc(received, method, **kwargs):
        assert received is server
        events.append("pool_reset" if method == "reset_ascendstore_benchmark_pool" else "store_drain")
        return {"workers": [{"removed_objects": 1}]}

    def reset(received):
        assert received is server
        events.append("local_reset")
        return {}

    def accuracy(producer, consumer, tokenizer, directory, observations):
        assert producer is consumer is server
        assert directory == tmp_path / "saved-kv"
        events.append("accuracy")
        observations.append({"checked": True})

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
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
        "call_worker_rpc": rpc,
        "run_byte_accuracy": accuracy,
        "reset_local_prefix_cache": reset,
        "assert_prompt_sources": lambda measurement, *, external: sources.append(external),
    }
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
    observations: list[dict[str, Any]] = []
    arguments = (tmp_path, "model", None, ["inputs"], ["warmup"], ("v1", "AscendStoreV1Connector"), observations)
    if mode == "request_failure":
        with pytest.raises(ValueError, match="request failed"):
            namespace[entry.name](*arguments)
        assert events == [
            "server_enter",
            "warmup",
            "pool_reset",
            "local_reset",
            "benchmark",
            "server_exit",
            "pool_exit",
        ]
        assert len(observations) == 1 and not sources
    elif mode == "accuracy":
        namespace[entry.name](*arguments, accuracy_directory=tmp_path / "saved-kv")
        assert events == ["server_enter", "accuracy", "server_exit", "pool_exit"]
        assert observations == [{"checked": True}] and not sources
    else:
        namespace[entry.name](*arguments)
        phase = ["pool_reset", "local_reset", "benchmark", "store_drain", "local_reset", "benchmark"]
        assert events == ["server_enter", "warmup", *phase, *phase, "server_exit", "pool_exit"]
        assert sources == [False, True, False, True]
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
