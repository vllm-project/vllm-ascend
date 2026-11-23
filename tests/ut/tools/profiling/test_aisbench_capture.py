"""Capture selection and failure propagation through AISBench."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import aisbench
from tools.profiling import capture


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    root = tmp_path / "profile"
    spec = capture.ProfileSpec(enabled=True, output="raw")
    for name, value in spec.environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ASCEND_PROFILE_ROOT", str(root))
    capture.initialize_profile_run(root, spec)
    target = capture.ServeInstance.from_endpoint("serve-0", "http://server:8000", root=root)
    capture.ServeManifest((target,)).write(root / "serve_manifest.json")
    trace = Path(target.raw_trace_dir)
    trace.mkdir(parents=True)
    (trace / "raw.bin").write_bytes(b"trace")
    (tmp_path / "chat.py").write_text("models = [dict(\n    model='model',\n)]\n")
    (tmp_path / "perf.py").write_text(
        "datasets = [dict(\n    path='data',\n    inferencer=dict(type=GenInferencer),\n)]\n"
    )
    (tmp_path / "perf.csv").write_text(",Average\nTPOT,10ms\n")
    (tmp_path / "perf.json").write_text(json.dumps({"Output Token Throughput": {"total": "20token/s"}}))
    for name in ("REQUEST_CONF_DIR", "DATASET_CONF_DIR"):
        monkeypatch.setattr(aisbench, name, str(tmp_path))
    process, controller = Mock(), Mock()
    controller.finish.return_value = {"status": "success", "reason": "duration_reached"}
    factory = Mock(return_value=controller)
    monkeypatch.setattr(aisbench.subprocess, "Popen", process)
    monkeypatch.setattr(aisbench, "ProfileController", factory)
    log = f"Current exp folder: {tmp_path}\nPerformance Result files located in {tmp_path}."
    monkeypatch.setattr(aisbench.AisbenchRunner, "_check_runtime_stdout", lambda self: log)
    config = dict(
        benchmark_name="perf",
        benchmark_artifact_name="model__perf",
        case_type="performance",
        dataset_path_local=str(tmp_path),
        model_path=str(tmp_path),
        request_conf="chat",
        dataset_conf="perf",
        max_out_len=32,
        batch_size=2,
        baseline=10,
        threshold=1,
    )
    return SimpleNamespace(root=root, config=config, process=process, controller=controller, factory=factory)


@pytest.mark.parametrize("mode", ["disabled", "unselected", "selected"])
def test_only_selected_benchmarks_use_the_capture_adapter(benchmark, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", str(mode != "disabled"))
    monkeypatch.setenv("ASCEND_PROFILE_BENCHMARK_NAMES", "other" if mode == "unselected" else "perf")
    with aisbench.AisbenchRunner("model", 8000, benchmark.config):
        pass
    environment = benchmark.process.call_args.kwargs["env"]
    dataset = (tmp_path / "perf_custom.py").read_text()
    manifest = json.loads((benchmark.root / "profile_manifest.json").read_text())
    if mode == "selected":
        marker = benchmark.root / "benchmarks/model__perf/first_request"
        assert environment["ASCEND_PROFILE_REQUEST_MARKER"] == str(marker)
        assert "type=TimedGenInferencer" in dataset
        assert benchmark.factory.call_args.args[2] == marker
        benchmark.controller.start.assert_called_once()
        benchmark.controller.finish.assert_called_once()
        record = manifest["benchmarks"][0]
        assert record["status"] == "success" and record["benchmark_name"] == "perf"
        assert (benchmark.root / record["targets"][0]["archive"]).is_file()
        assert (benchmark.root / record["serve_manifest"]).is_file()
    else:
        assert environment is None and "type=GenInferencer" in dataset
        assert "TimedGenInferencer" not in dataset
        benchmark.factory.assert_not_called()
        assert manifest["benchmarks"] == []


def test_missing_inventory_stops_before_launch(benchmark):
    (benchmark.root / "serve_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    benchmark.process.assert_not_called()


def test_request_trigger_prepares_worker_progress_directory(benchmark, monkeypatch):
    monkeypatch.delenv("ASCEND_PROFILE_START_AFTER")
    monkeypatch.setenv("ASCEND_PROFILE_START_AFTER_REQUESTS", "100")
    with aisbench.AisbenchRunner("model", 8000, benchmark.config):
        pass
    assert (benchmark.root / "benchmarks/model__perf/completed_requests").is_dir()
    assert benchmark.factory.call_args.args[0].start_after_requests == 100


def test_partial_capture_raises_and_retains_available_raw(benchmark):
    benchmark.controller.finish.return_value = {"status": "partial", "reason": "benchmark_ended"}
    with pytest.raises(capture.ArtifactError, match="incomplete"):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    manifest = json.loads((benchmark.root / "profile_manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert (benchmark.root / manifest["benchmarks"][0]["targets"][0]["archive"]).is_file()


def test_benchmark_error_preserves_the_original_exception_and_stops_capture(benchmark, monkeypatch):
    monkeypatch.setattr(
        aisbench.AisbenchRunner, "_check_runtime_stdout", Mock(side_effect=RuntimeError("request failed"))
    )
    benchmark.controller.finish.side_effect = OSError("cleanup failed")
    with pytest.raises(RuntimeError, match="request failed"):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    benchmark.controller.finish.assert_called_once()
