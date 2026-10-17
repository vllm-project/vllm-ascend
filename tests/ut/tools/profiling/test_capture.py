"""Configuration, serving topology, and runtime capture behavior."""

import json
import shlex
import time
from dataclasses import fields
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tools.profiling import capture


@pytest.fixture(autouse=True)
def profile_environment(monkeypatch):
    for field in fields(capture.ProfileSpec):
        monkeypatch.delenv(f"ASCEND_PROFILE_{field.name.upper()}", raising=False)


@pytest.mark.parametrize("trigger", [{"start_after": "50"}, {"start_after_requests": "100"}])
def test_profile_configuration_round_trip(monkeypatch, trigger):
    assert capture.ProfileSpec.from_env() == capture.ProfileSpec(enabled=False)
    options = {
        "enabled": "TRUE",
        "with_stack": "false",
        **trigger,
        "duration": 8,
        "output": "raw",
        "benchmark_names": " throughput, latency ",
    }
    spec = capture.ProfileSpec.from_options(options)
    assert (spec.enabled, spec.with_stack, spec.duration, spec.output) == (True, False, 8, "raw")
    if "start_after_requests" in trigger:
        assert spec.start_after is None and spec.start_after_requests == 100
        assert "start_after" not in spec.options()
    else:
        assert spec.start_after == 50 and spec.start_after_requests is None
    assert spec.benchmark_names == ("throughput", "latency")
    assert spec.includes_benchmark("throughput", "performance")
    assert not spec.includes_benchmark("other", "performance")
    assert not spec.includes_benchmark("throughput", "accuracy")
    assert not capture.ProfileSpec().includes_benchmark("throughput", "performance")
    for name, value in spec.environment().items():
        monkeypatch.setenv(name, value)
    assert capture.ProfileSpec.from_env() == capture.ProfileSpec.from_options(spec.options()) == spec


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "yes"},
        {"duration": 0},
        {"start_after": -1},
        {"duration": 1.5},
        {"start_after_requests": 0},
        {"start_after_requests": -1},
        {"start_after_requests": True},
        {"start_after_requests": 1.5},
        {"start_after_requests": "1.5"},
        {"start_after": 60, "start_after_requests": 100},
        {"output": "invalid"},
        {"benchmark_names": ["throughput"]},
        {"unexpected_option": True},
    ],
)
def test_invalid_user_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        capture.ProfileSpec.from_options(options)


@pytest.fixture
def serving_instances(tmp_path):
    return (
        capture.ServeInstance.from_endpoint("dp-0", "http://leader:8000", root=tmp_path),
        capture.ServeInstance.from_endpoint("dp-1", "http://leader:8000", node_index=1, root=tmp_path),
        capture.ServeInstance.from_endpoint("decode", "http://decoder:8000", node_index=1, root=tmp_path),
    )


@pytest.mark.parametrize("shell_command", [False, True])
def test_all_serving_instances_receive_independent_trace_directories(tmp_path, serving_instances, shell_command):
    path = tmp_path / "serve_manifest.json"
    capture.ServeManifest(serving_instances).write(path)
    instances = capture.ServeManifest.read(path).instances
    assert instances == serving_instances
    command = ["vllm", "serve", "model with spaces", "--profiler-config", '{"torch_profiler_with_memory":true}']
    original = shlex.join(command) if shell_command else command
    directories = []
    for instance in instances:
        configured = capture.configure_profiler_command(original, instance, capture.ProfileSpec(with_stack=True))
        args = shlex.split(configured) if shell_command else configured
        assert args[:3] == command[:3]
        assert args.count("--profiler-config") == 1
        options = json.loads(args[args.index("--profiler-config") + 1])
        assert options["profiler"] == "torch"
        assert options["torch_profiler_with_memory"] is options["torch_profiler_with_stack"] is True
        directories.append(options["torch_profiler_dir"])
    assert len(set(directories)) == len(instances)
    assert command[-1] == '{"torch_profiler_with_memory":true}'


@pytest.mark.parametrize("outcome", ["complete", "start_error", "stop_error", "benchmark_ended"])
def test_capture_lifecycle_controls_each_api_once(tmp_path, monkeypatch, serving_instances, outcome):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    capture.record_first_request(marker)
    started, stopped = Event(), Event()

    def start(endpoint):
        started.set()
        if outcome == "start_error" and endpoint == "http://leader:8000":
            raise OSError("start unavailable")

    def stop(endpoint):
        stopped.set()
        if outcome == "stop_error" and endpoint == "http://leader:8000":
            raise OSError("stop unavailable")

    client = SimpleNamespace(start_profile=Mock(side_effect=start), stop_profile=Mock(side_effect=stop))
    controller = capture.ProfileController(
        capture.ProfileSpec(start_after=0, duration=1), serving_instances, marker, client
    )
    controller.start()
    try:
        assert started.wait(5)
        if outcome != "benchmark_ended":
            assert stopped.wait(5)
    finally:
        result = controller.finish()
    endpoints = [call("http://leader:8000"), call("http://decoder:8000")]
    assert client.start_profile.call_count == client.stop_profile.call_count == 2
    client.start_profile.assert_has_calls(endpoints, any_order=True)
    client.stop_profile.assert_has_calls(endpoints, any_order=True)
    assert result["status"] == ("success" if outcome == "complete" else "partial")
    if outcome == "complete":
        assert result["actual_duration"] >= 1
    elif outcome in ("start_error", "stop_error"):
        assert result["targets"]["dp-0"][outcome] == result["targets"]["dp-1"][outcome]
        assert "unavailable" in result["targets"]["dp-0"][outcome]


@pytest.mark.parametrize("has_request", [False, True])
def test_benchmark_finishing_before_capture_is_partial(tmp_path, monkeypatch, serving_instances, has_request):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    if has_request:
        marker.write_text(str(time.perf_counter()))
    client = Mock()
    controller = capture.ProfileController(capture.ProfileSpec(start_after=60), serving_instances, marker, client)
    controller.start()
    result = controller.finish()
    assert result["status"] == "partial"
    assert result["reason"] in ("no_request_before_benchmark_end", "benchmark_ended_before_start")
    client.start_profile.assert_not_called()
    client.stop_profile.assert_not_called()


def test_capture_waits_for_the_offset_from_the_first_request(tmp_path, monkeypatch, serving_instances):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    started, timestamps = Event(), []

    def start(endpoint):
        timestamps.append(time.perf_counter())
        started.set()

    client = SimpleNamespace(start_profile=start, stop_profile=Mock())
    controller = capture.ProfileController(capture.ProfileSpec(start_after=1), serving_instances[:1], marker, client)
    controller.start()
    try:
        assert not started.wait(0.01)
        capture.record_first_request(marker)
        assert started.wait(5)
    finally:
        controller.finish()
    assert timestamps[0] >= float(marker.read_text()) + 1
    client.stop_profile.assert_called_once_with("http://leader:8000")


def test_request_trigger_sums_workers_without_waiting_for_default_offset(tmp_path, monkeypatch, serving_instances):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    capture.record_first_request(marker)
    progress = tmp_path / "completed_requests"
    progress.mkdir()
    (progress / "worker-1.count").write_text("2")
    (progress / "worker-2.count").write_text("2")
    (progress / "worker-2.tmp").write_text("100")
    started = Event()
    client = SimpleNamespace(start_profile=Mock(side_effect=lambda endpoint: started.set()), stop_profile=Mock())
    spec = capture.ProfileSpec.from_options({"start_after_requests": 5})
    controller = capture.ProfileController(spec, serving_instances[:1], marker, client)
    controller.start()
    try:
        assert not started.wait(0.02)
        (progress / "worker-2.tmp").write_text("3")
        (progress / "worker-2.tmp").replace(progress / "worker-2.count")
        assert started.wait(5)
    finally:
        result = controller.finish()
    assert result["completed_requests"] == 5
    assert 0 <= result["actual_start_after"] < 60
    client.start_profile.assert_called_once()
    client.stop_profile.assert_called_once()


def test_unreached_request_threshold_is_incomplete(tmp_path, monkeypatch, serving_instances):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    capture.record_first_request(marker)
    progress = tmp_path / "completed_requests"
    progress.mkdir()
    (progress / "worker.count").write_text("2")
    observed = Event()
    original_read = capture.Path.read_text

    def read_count(path, *args, **kwargs):
        if path.suffix == ".count":
            observed.set()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(capture.Path, "read_text", read_count)
    client = Mock()
    controller = capture.ProfileController(
        capture.ProfileSpec.from_options({"start_after_requests": 5}), serving_instances, marker, client
    )
    controller.start()
    try:
        assert observed.wait(5)
    finally:
        result = controller.finish()
    assert result["status"] == "partial"
    assert result["reason"] == "benchmark_ended_before_request_threshold"
    assert result["completed_requests"] == 2
    client.start_profile.assert_not_called()
