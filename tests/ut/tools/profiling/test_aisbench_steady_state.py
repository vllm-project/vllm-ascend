"""Report generation through an ordinary AISBench performance run."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import aisbench


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    (tmp_path / "chat.py").write_text("models = [dict(\n    model='model',\n)]\n")
    (tmp_path / "perf.py").write_text(
        "datasets = [dict(\n    path='data',\n    inferencer=dict(type=GenInferencer),\n)]\n"
    )
    (tmp_path / "perf.csv").write_text(",Average\nTPOT,10ms\n")
    (tmp_path / "perf.json").write_text(json.dumps({"Output Token Throughput": {"total": "20token/s"}}))
    (tmp_path / "perf_details.jsonl").write_text(
        '{"success":true,"time_points":[100,130]}\n{"success":true,"time_points":[105,125]}\n'
    )
    monkeypatch.setattr(aisbench, "DATASET_CONF_DIR", str(tmp_path))
    monkeypatch.setattr(aisbench, "REQUEST_CONF_DIR", str(tmp_path))
    monkeypatch.setattr(aisbench, "STEADY_STATE_OUTPUT_DIR", tmp_path / "charts")
    process = Mock()
    monkeypatch.setattr(aisbench.subprocess, "Popen", process)
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
    return SimpleNamespace(config=config, process=process, summary=tmp_path / "charts/model__perf/summary.json")


def test_report_is_postprocessing_of_the_original_request_path(benchmark, tmp_path):
    with aisbench.AisbenchRunner("model", 8000, benchmark.config) as runner:
        assert runner.result_json["Output Token Throughput"]["total"] == "20token/s"
    assert benchmark.process.call_args.kwargs["shell"] is True
    assert benchmark.process.call_args.kwargs.get("env") is None
    assert "type=GenInferencer" in (tmp_path / "perf_custom.py").read_text()
    summary = json.loads(benchmark.summary.read_text())
    assert summary["benchmark_name"] == "perf"
    assert (summary["steady_state"]["start"]["time_s"], summary["steady_state"]["end"]["time_s"]) == (5, 25)


@pytest.mark.parametrize("throughput", [5, 20])
def test_plot_failure_preserves_the_baseline_verdict(benchmark, tmp_path, monkeypatch, caplog, throughput):
    (tmp_path / "perf.json").write_text(json.dumps({"Output Token Throughput": {"total": f"{throughput}token/s"}}))
    monkeypatch.setattr(aisbench, "render_terminal", Mock(side_effect=OSError("cannot render")))
    if throughput < 10:
        with pytest.raises(AssertionError, match="Performance verification failed"):
            aisbench.AisbenchRunner("model", 8000, benchmark.config)
    else:
        with aisbench.AisbenchRunner("model", 8000, benchmark.config):
            pass
    assert "cannot render" in caplog.text
    assert all(record.levelname == "WARNING" for record in caplog.records)


def test_missing_details_produce_an_unavailable_report(benchmark, tmp_path):
    (tmp_path / "perf_details.jsonl").unlink()
    with aisbench.AisbenchRunner("model", 8000, benchmark.config):
        pass
    assert json.loads(benchmark.summary.read_text())["status"] == "unavailable"
