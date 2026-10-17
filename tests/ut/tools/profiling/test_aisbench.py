"""Profiling integration through the benchmark runner's public entry point."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import aisbench
from tools.profiling import capture


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    root, results, charts = (tmp_path / name for name in ("capture", "results", "charts"))
    spec = capture.ProfileSpec(enabled=True)
    for name, value in spec.environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ASCEND_PROFILE_ROOT", str(root))
    capture.initialize_profile_run(root, spec)
    target = capture.ServeInstance.from_endpoint("serve-0", "http://server:8000", root=root)
    capture.ServeManifest((target,)).write(root / "serve_manifest.json")
    trace = Path(target.raw_trace_dir) / "rank0_ascend_pt"
    trace.mkdir(parents=True)
    (trace / "raw.bin").write_bytes(b"profile")
    results.mkdir()
    (results / "dataset.csv").write_text(",Average\nTPOT,10ms\n")
    (results / "dataset.json").write_text(json.dumps({"Output Token Throughput": {"total": "20token/s"}}))
    (results / "dataset_details.jsonl").write_text(
        '{"success":true,"time_points":[100,130]}\n{"success":true,"time_points":[105,125]}\n'
    )
    (tmp_path / "chat.py").write_text("models = [dict(\n    type=VLLMCustomAPIChat,\n    model='model',\n)]\n")
    (tmp_path / "dataset.py").write_text(
        "datasets = [dict(\n    path='dataset',\n    infer_cfg=dict(inferencer=dict(type=GenInferencer)),\n)]\n"
    )
    monkeypatch.setattr(aisbench, "REQUEST_CONF_DIR", str(tmp_path))
    monkeypatch.setattr(aisbench, "DATASET_CONF_DIR", str(tmp_path))
    monkeypatch.setattr(aisbench, "STEADY_STATE_OUTPUT_DIR", charts)
    config = {
        "benchmark_name": "throughput",
        "benchmark_artifact_name": "model__throughput",
        "case_type": "performance",
        "dataset_path_local": str(tmp_path),
        "model_path": str(tmp_path),
        "request_conf": "chat",
        "dataset_conf": "dataset",
        "max_out_len": 32,
        "batch_size": 2,
        "baseline": 10,
        "threshold": 1,
    }

    def launch(command, **kwargs):
        Path(kwargs["env"]["ASCEND_PROFILE_REQUEST_MARKER"]).write_text("100")
        return Mock()

    process = Mock(side_effect=launch)
    monkeypatch.setattr(aisbench.subprocess, "Popen", process)
    log = f"Current exp folder: {results}\nPerformance Result files located in {results}."
    monkeypatch.setattr(aisbench.AisbenchRunner, "_check_runtime_stdout", lambda self: log)
    controller = Mock()
    controller.finish.return_value = {"status": "success", "actual_duration": 8, "reason": "duration_reached"}
    factory = Mock(return_value=controller)
    monkeypatch.setattr(aisbench, "ProfileController", factory)
    return SimpleNamespace(
        root=root,
        results=results,
        charts=charts,
        config=config,
        process=process,
        controller=controller,
        factory=factory,
    )


@pytest.mark.parametrize("mode", ["disabled", "unselected", "selected"])
def test_runner_connects_benchmark_identity_timing_and_capture(benchmark, monkeypatch, mode):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", str(mode != "disabled"))
    monkeypatch.setenv("ASCEND_PROFILE_BENCHMARK_NAMES", "other" if mode == "unselected" else "throughput")
    original = deepcopy(benchmark.config)
    with aisbench.AisbenchRunner("model", 8000, benchmark.config) as runner:
        assert runner.result[1]["Output Token Throughput"]["total"] == "20token/s"
    assert benchmark.config == original
    environment = benchmark.process.call_args.kwargs["env"]
    marker = benchmark.charts / "model__throughput/first_request"
    assert environment["ASCEND_PROFILE_REQUEST_MARKER"] == str(marker)
    assert environment["ASCEND_PROFILE_ENABLED"] == str(mode == "selected").lower()
    summary = json.loads((marker.parent / "summary.json").read_text())
    assert summary["benchmark_name"] == "throughput"
    assert summary["steady_state"]["start"]["time_s"] == 5
    assert summary["steady_state"]["end"]["time_s"] == 25
    manifest = json.loads((benchmark.root / "profile_manifest.json").read_text())
    if mode == "selected":
        assert benchmark.factory.call_args.args[2] == marker
        benchmark.controller.start.assert_called_once()
        benchmark.controller.finish.assert_called_once()
        record = manifest["benchmarks"][0]
        assert (record["benchmark_name"], record["benchmark_artifact_name"], record["status"]) == (
            "throughput",
            "model__throughput",
            "success",
        )
        assert (benchmark.root / record["serve_manifest"]).is_file()
        assert (benchmark.root / record["targets"][0]["archive"]).is_file()
    else:
        benchmark.factory.assert_not_called()
        assert manifest["benchmarks"] == []


def test_selected_capture_requires_serving_inventory(benchmark):
    (benchmark.root / "serve_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    benchmark.process.assert_not_called()


def test_request_trigger_starts_with_fresh_worker_counts(benchmark, monkeypatch):
    monkeypatch.delenv("ASCEND_PROFILE_START_AFTER")
    monkeypatch.setenv("ASCEND_PROFILE_START_AFTER_REQUESTS", "100")
    progress = benchmark.charts / "model__throughput/completed_requests"
    progress.mkdir(parents=True)
    (progress / "previous-worker.count").write_text("1000")
    with aisbench.AisbenchRunner("model", 8000, benchmark.config):
        pass
    assert progress.is_dir() and not list(progress.iterdir())
    environment = benchmark.process.call_args.kwargs["env"]
    assert environment["ASCEND_PROFILE_START_AFTER_REQUESTS"] == "100"
    assert "ASCEND_PROFILE_START_AFTER" not in environment
    assert benchmark.factory.call_args.args[0].start_after_requests == 100


def test_incomplete_capture_fails_benchmark_and_saves_available_trace(benchmark):
    benchmark.controller.finish.return_value = {"status": "partial", "reason": "benchmark_ended", "actual_duration": 2}
    with pytest.raises(capture.ArtifactError, match="incomplete"):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    manifest = json.loads((benchmark.root / "profile_manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["benchmarks"][0]["reason"] == "benchmark_ended"
    assert (benchmark.root / manifest["benchmarks"][0]["targets"][0]["archive"]).is_file()


def test_benchmark_error_still_finishes_capture(benchmark, monkeypatch):
    monkeypatch.setattr(
        aisbench.AisbenchRunner, "_check_runtime_stdout", Mock(side_effect=RuntimeError("benchmark failed"))
    )
    with pytest.raises(RuntimeError, match="benchmark failed"):
        aisbench.AisbenchRunner("model", 8000, benchmark.config)
    benchmark.controller.finish.assert_called_once()


@pytest.mark.parametrize("throughput", [5, 20])
def test_chart_errors_do_not_change_performance_verdict(benchmark, monkeypatch, caplog, throughput):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", "false")
    (benchmark.results / "dataset.json").write_text(
        json.dumps({"Output Token Throughput": {"total": f"{throughput}token/s"}})
    )
    monkeypatch.setattr(aisbench, "render_terminal", Mock(side_effect=OSError("chart output unavailable")))
    if throughput < 10:
        with pytest.raises(AssertionError, match="Performance verification failed"):
            aisbench.AisbenchRunner("model", 8000, benchmark.config)
    else:
        with aisbench.AisbenchRunner("model", 8000, benchmark.config) as runner:
            assert runner.result[1]["Output Token Throughput"]["total"] == "20token/s"
    assert "chart output unavailable" in caplog.text
    assert all(record.levelname == "WARNING" for record in caplog.records)


def test_missing_timing_data_produces_unavailable_summary(benchmark, monkeypatch):
    monkeypatch.setenv("ASCEND_PROFILE_ENABLED", "false")
    (benchmark.results / "dataset_details.jsonl").unlink()
    with aisbench.AisbenchRunner("model", 8000, benchmark.config):
        pass
    summary = json.loads((benchmark.charts / "model__throughput/summary.json").read_text())
    assert summary["status"] == "unavailable"
