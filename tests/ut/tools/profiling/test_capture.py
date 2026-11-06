"""Capture configuration, endpoint control, and trigger boundaries."""

import json
import shlex
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tools.profiling import capture


@pytest.mark.parametrize("trigger", [{"start_after": "60"}, {"start_after_requests": "5"}])
def test_options_environment_and_benchmark_selection(monkeypatch, trigger):
    options = {"enabled": True, "benchmark_names": "perf, latency", "duration": 8, **trigger}
    spec = capture.ProfileSpec.from_options(options)
    assert spec.includes_benchmark("perf", "performance")
    assert not spec.includes_benchmark("other", "performance")
    assert not spec.includes_benchmark("perf", "accuracy")
    for name, value in spec.environment().items():
        monkeypatch.setenv(name, value)
    assert capture.ProfileSpec.from_env() == capture.ProfileSpec.from_options(spec.options()) == spec
    assert (spec.start_after, spec.start_after_requests) == ((60, None) if "start_after" in trigger else (None, 5))


@pytest.mark.parametrize(
    "options",
    [
        {"start_after": -1},
        {"duration": 0},
        {"start_after_requests": 0},
        {"start_after_requests": True},
        {"duration": 1.5},
        {"start_after": 60, "start_after_requests": 5},
        {"output": "invalid"},
        {"unexpected": True},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        capture.ProfileSpec.from_options(options)


@pytest.fixture
def targets(tmp_path):
    return tuple(
        capture.ServeInstance.from_endpoint(name, endpoint, node_index=index, root=tmp_path)
        for index, (name, endpoint) in enumerate(
            (("dp-0", "http://leader:8000"), ("dp-1", "http://leader:8000"), ("decode", "http://decoder:8000"))
        )
    )


@pytest.mark.parametrize("shell", [False, True])
def test_profiler_configuration_preserves_options_and_node_directories(tmp_path, targets, shell):
    path = tmp_path / "serve_manifest.json"
    capture.ServeManifest(targets).write(path)
    assert capture.ServeManifest.read(path).instances == targets
    command = ["vllm", "serve", "model with spaces", "--profiler-config", '{"torch_profiler_with_memory":true}']
    directories = []
    for target in targets:
        configured = capture.configure_profiler_command(
            shlex.join(command) if shell else command, target, capture.ProfileSpec(with_stack=True)
        )
        args = shlex.split(configured) if shell else configured
        assert args[:3] == command[:3] and args.count("--profiler-config") == 1
        options = json.loads(args[args.index("--profiler-config") + 1])
        assert options["torch_profiler_with_memory"] is options["torch_profiler_with_stack"] is True
        directories.append(options["torch_profiler_dir"])
    assert len(set(directories)) == 3


@pytest.mark.parametrize("failure", [None, "start", "stop"])
def test_capture_controls_shared_endpoints_once_and_reports_errors(tmp_path, monkeypatch, targets, failure):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    capture.record_first_request(marker)
    stopped = Event()

    def operation(stage, endpoint):
        if stage == "stop":
            stopped.set()
        if failure == stage and endpoint == targets[0].endpoint:
            raise OSError(f"{stage} unavailable")

    client = SimpleNamespace(
        start_profile=Mock(side_effect=lambda endpoint: operation("start", endpoint)),
        stop_profile=Mock(side_effect=lambda endpoint: operation("stop", endpoint)),
    )
    controller = capture.ProfileController(capture.ProfileSpec(start_after=0, duration=1), targets, marker, client)
    controller.start()
    try:
        assert stopped.wait(5)
    finally:
        result = controller.finish()
    for method in (client.start_profile, client.stop_profile):
        assert method.call_count == 2
        method.assert_has_calls([call("http://leader:8000"), call("http://decoder:8000")], any_order=True)
    assert result["status"] == ("partial" if failure else "success")
    if failure:
        assert result["targets"]["dp-0"][f"{failure}_error"] == result["targets"]["dp-1"][f"{failure}_error"]
        assert "unavailable" in result["targets"]["dp-0"][f"{failure}_error"]


@pytest.mark.parametrize("trigger", ["time", "requests"])
def test_capture_waits_for_its_trigger_and_stops_when_benchmark_ends(tmp_path, monkeypatch, targets, trigger):
    monkeypatch.setattr(capture, "POLL_INTERVAL_SECONDS", 0.001)
    marker = tmp_path / "first_request"
    capture.record_first_request(marker)
    progress = tmp_path / "completed_requests"
    progress.mkdir()
    (progress / "worker-0.count").write_text("2")
    (progress / "worker-1.count").write_text("2")
    started = Event()
    client = SimpleNamespace(start_profile=Mock(side_effect=lambda endpoint: started.set()), stop_profile=Mock())
    spec = capture.ProfileSpec.from_options({"start_after": 1} if trigger == "time" else {"start_after_requests": 5})
    controller = capture.ProfileController(spec, targets[:1], marker, client)
    controller.start()
    try:
        assert not started.wait(0.02)
        if trigger == "requests":
            temporary = progress / "worker-1.tmp"
            temporary.write_text("3")
            temporary.replace(progress / "worker-1.count")
        assert started.wait(5)
        if trigger == "time":
            assert time.perf_counter() >= float(marker.read_text()) + 1
    finally:
        result = controller.finish()
    assert result["status"] == "partial" and result["reason"] == "benchmark_ended"
    if trigger == "requests":
        assert result["completed_requests"] == 5
    client.stop_profile.assert_called_once()


def test_benchmark_without_requests_does_not_start_capture(tmp_path, targets):
    client = Mock()
    controller = capture.ProfileController(capture.ProfileSpec(), targets, tmp_path / "absent", client)
    controller.start()
    assert controller.finish()["reason"] == "no_request_before_benchmark_end"
    client.start_profile.assert_not_called()
    client.stop_profile.assert_not_called()
