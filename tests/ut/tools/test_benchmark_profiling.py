import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from tests.e2e.common.single_node import benchmark_profiling as profiling


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "profiling" / "test-model"
    settings = {
        "benchmark_key": "perf",
        "output_dir": "profiling/test-model",
        "delay_iterations": 1024,
        "max_iterations": 256,
        "expected_workers": 4,
        "timeout_seconds": 300,
    }
    server_profiler = {
        "profiler": "torch",
        "torch_profiler_dir": str(output_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": 1024,
        "max_iterations": 256,
    }
    benchmarks = {
        "perf-warm": {"case_type": "performance", "num_prompts": 18},
        "perf": {"case_type": "performance", "num_prompts": 180},
        "accuracy": {"case_type": "accuracy"},
    }
    return settings, server_profiler, benchmarks, output_dir


def parse(case):
    settings, server_profiler, benchmarks, _ = case
    return profiling.parse_benchmark_profiling_config(
        settings,
        benchmarks=benchmarks,
        server_cmd=["--profiler-config", json.dumps(server_profiler)],
        service_mode="openai",
    )


def create_traces(output_dir: Path, *, count: int = 4, data: str = "name,duration\nmatmul,100\n") -> None:
    for rank in range(count):
        table = output_dir / f"worker-rank-{rank}_ascend_pt" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        table.parent.mkdir(parents=True)
        table.write_text(data, encoding="utf-8")


def read_manifest(output_dir: Path):
    return json.loads((output_dir / profiling.MANIFEST_FILENAME).read_text(encoding="utf-8"))


def test_disabled_profiling_needs_no_profiler_command():
    assert profiling.parse_benchmark_profiling_config(None, benchmarks={}, server_cmd=[], service_mode="openai") is None


def test_parse_matches_absolute_server_path(case):
    config = parse(case)
    assert config.output_dir == case[3]
    assert config.delay_iterations == 1024
    assert config.max_iterations == 256
    assert config.expected_workers == 4


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("benchmark_key", "missing"),
        ("benchmark_key", "accuracy"),
        ("benchmark_key", "perf-warm"),
        ("output_dir", "../outside"),
        ("output_dir", "profiling/../outside"),
        ("output_dir", "profiling"),
        ("delay_iterations", -1),
        ("delay_iterations", True),
        ("max_iterations", 0),
        ("max_iterations", 1.5),
        ("expected_workers", False),
        ("timeout_seconds", 0),
        ("timeout_seconds", True),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", float("nan")),
        ("duration_seconds", 3),
    ],
)
def test_reject_invalid_yaml(case, key, value):
    case[0][key] = value
    with pytest.raises(ValueError):
        parse(case)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("profiler", "cuda"),
        ("torch_profiler_with_stack", True),
        ("torch_profiler_with_memory", True),
        ("ignore_frontend", False),
        ("delay_iterations", 30),
        ("max_iterations", True),
        ("torch_profiler_dir", "profiling/test-model"),
        ("torch_profiler_dir", "/different-directory"),
    ],
)
def test_reject_mismatched_server_config(case, key, value):
    case[1][key] = value
    with pytest.raises(ValueError):
        parse(case)


@pytest.mark.parametrize("server_cmd", [[], ["--profiler-config"], ["--profiler-config", "bad-json"]])
def test_reject_missing_or_invalid_profiler_command(case, server_cmd):
    with pytest.raises(ValueError):
        profiling.parse_benchmark_profiling_config(
            case[0], benchmarks=case[2], server_cmd=server_cmd, service_mode="openai"
        )


def test_reject_epd_service(case):
    with pytest.raises(ValueError, match="OpenAI service"):
        profiling.parse_benchmark_profiling_config(case[0], benchmarks=case[2], server_cmd=[], service_mode="epd")


def test_profile_only_formal_benchmark_and_flush_before_return(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    events = []

    def post(url, timeout):
        endpoint = url.rsplit("/", 1)[-1]
        events.append(endpoint)
        assert timeout == (10, 300)
        if endpoint == "stop_profile":
            create_traces(config.output_dir)
        return Mock(status_code=200)

    def run_cases(**kwargs):
        benchmark = kwargs["aisbench_cases"][0]
        events.append(benchmark.get("num_prompts", "accuracy"))
        return [benchmark.get("num_prompts", "accuracy")]

    monkeypatch.setattr(profiling.requests, "post", post)
    result = profiling.run_profiled_benchmarks(
        model="test-model",
        port=8000,
        benchmark_keys=list(case[2]),
        benchmarks=case[2],
        profiling=config,
        run_cases=run_cases,
    )

    assert events == [18, "start_profile", 180, "stop_profile", "accuracy"]
    assert result == [18, 180, "accuracy"]
    manifest = read_manifest(config.output_dir)
    assert manifest["status"] == "success"
    assert manifest["valid_worker_trace_count"] == 4
    assert manifest["unique_worker_ranks"] == [0, 1, 2, 3]
    assert manifest["start_profile"]["http_status"] == 200
    assert manifest["stop_profile"]["status"] == "success"


def test_start_timeout_still_stops_and_does_not_launch_benchmark(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    start_error = requests.Timeout("start timed out")
    post = Mock(side_effect=[start_error, Mock(status_code=200)])
    runner = Mock()
    monkeypatch.setattr(profiling.requests, "post", post)

    with pytest.raises(requests.Timeout) as caught:
        profiling.run_profiled_benchmarks(
            model="test-model",
            port=8000,
            benchmark_keys=["perf"],
            benchmarks=case[2],
            profiling=config,
            run_cases=runner,
        )

    assert caught.value is start_error
    runner.assert_not_called()
    assert [call.args[0].rsplit("/", 1)[-1] for call in post.call_args_list] == ["start_profile", "stop_profile"]
    manifest = read_manifest(config.output_dir)
    assert manifest["start_profile"]["status"] == "failed"
    assert manifest["stop_profile"]["status"] == "success"
    assert manifest["status"] == "failed"


def test_stop_timeout_preserves_benchmark_failure(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    post = Mock(side_effect=[Mock(status_code=200), requests.Timeout("stop timed out")])
    monkeypatch.setattr(profiling.requests, "post", post)
    benchmark_error = AssertionError("performance gate failed")

    with pytest.raises(AssertionError) as caught, profiling.BenchmarkProfiler(config, 8000):
        raise benchmark_error

    assert caught.value is benchmark_error
    assert any("stop timed out" in note for note in benchmark_error.__notes__)
    manifest = read_manifest(config.output_dir)
    assert manifest["benchmark_error"] == "AssertionError: performance gate failed"
    assert manifest["stop_profile"]["status"] == "failed"


def test_successful_benchmark_with_failed_stop_fails_profiling(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    post = Mock(side_effect=[Mock(status_code=200), requests.Timeout("stop timed out")])
    monkeypatch.setattr(profiling.requests, "post", post)

    with pytest.raises(RuntimeError, match="stop timed out"), profiling.BenchmarkProfiler(config, 8000):
        create_traces(config.output_dir)

    assert read_manifest(config.output_dir)["status"] == "failed"


@pytest.mark.parametrize(("count", "data"), [(0, ""), (3, "kernel\nmatmul\n"), (4, "kernel\n"), (4, "")])
def test_http_success_cannot_hide_missing_or_empty_worker_data(case, monkeypatch: pytest.MonkeyPatch, count, data):
    config = parse(case)
    monkeypatch.setattr(profiling.requests, "post", Mock(return_value=Mock(status_code=200)))

    with pytest.raises(RuntimeError, match="worker traces"), profiling.BenchmarkProfiler(config, 8000):
        create_traces(config.output_dir, count=count, data=data)

    manifest = read_manifest(config.output_dir)
    assert manifest["status"] == "failed"
    assert manifest["trace_directory_count"] == count


def test_reject_old_traces_before_arming(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    create_traces(config.output_dir)
    post = Mock()
    monkeypatch.setattr(profiling.requests, "post", post)

    with pytest.raises(ValueError, match="fresh directory"), profiling.BenchmarkProfiler(config, 8000):
        pytest.fail("old data was accepted")

    post.assert_not_called()


def test_duplicate_rank_sessions_cannot_replace_missing_worker(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    monkeypatch.setattr(profiling.requests, "post", Mock(return_value=Mock(status_code=200)))

    with pytest.raises(RuntimeError, match="worker traces"), profiling.BenchmarkProfiler(config, 8000):
        create_traces(config.output_dir)
        (config.output_dir / "worker-rank-3_ascend_pt").rename(config.output_dir / "worker-rank-0_session2_ascend_pt")

    manifest = read_manifest(config.output_dir)
    assert manifest["trace_directory_count"] == 4
    assert manifest["valid_worker_trace_count"] == 3
    assert manifest["unique_worker_ranks"] == [0, 1, 2]


def test_manifest_write_failure_preserves_benchmark_failure(case, monkeypatch: pytest.MonkeyPatch):
    config = parse(case)
    monkeypatch.setattr(profiling.requests, "post", Mock(return_value=Mock(status_code=200)))
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=OSError("disk full")))
    benchmark_error = AssertionError("benchmark failed")

    with pytest.raises(AssertionError) as caught, profiling.BenchmarkProfiler(config, 8000):
        raise benchmark_error

    assert caught.value is benchmark_error
    assert any("disk full" in note for note in benchmark_error.__notes__)
