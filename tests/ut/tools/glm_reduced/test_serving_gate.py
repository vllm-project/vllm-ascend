# SPDX-License-Identifier: Apache-2.0

import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.glm_reduced import nightly
from tools.glm_reduced.prepare import read_json
from tools.glm_reduced.profiles import get_profile
from tools.glm_reduced.reducer import execute_plan, load_source, plan_reduction
from tools.glm_reduced.serving_gate import (
    METRICS,
    aggregate_perf,
    checkpoint_identity,
    compare_output,
    compare_perf,
    normalized_completion,
    validate_baseline,
    validate_perf,
)


@pytest.fixture
def completion_request():
    return {"prompt": [1, 2, 3], "max_tokens": 64}


@pytest.fixture
def response():
    return {
        "id": "random-id",
        "created": 123,
        "choices": [{"index": 0, "text": "first\r\n  second ", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 64},
    }


def test_output_ignores_dynamic_fields_and_only_normalizes_newlines(response, completion_request):
    expected = normalized_completion(response, completion_request)
    response.update(id="different", created=999, timings={"duration": 42})
    response["choices"][0]["text"] = "first\n  second "
    compare_output(normalized_completion(response, completion_request), expected)
    with pytest.raises(ValueError, match="character"):
        compare_output(expected.rstrip(), expected)


@pytest.mark.parametrize("change", ["empty", "missing", "early", "truncated", "prompt", "multi"])
def test_bad_response_fails(response, completion_request, change):
    if change == "empty":
        response["choices"][0]["text"] = ""
    elif change == "missing":
        response.pop("usage")
    elif change == "early":
        response["choices"][0]["finish_reason"] = "stop"
    elif change == "truncated":
        response["usage"]["completion_tokens"] = 63
    elif change == "prompt":
        response["usage"]["prompt_tokens"] = 2
    else:
        response["choices"].append(response["choices"][0])
    with pytest.raises(ValueError):
        normalized_completion(response, completion_request)


def sample(workload):
    count = workload["num_prompts"]
    return {
        "completed": count,
        "total_input_tokens": count * workload["random_input_len"],
        "total_output_tokens": count * workload["random_output_len"],
        "input_lens": [workload["random_input_len"]] * count,
        "output_lens": [workload["random_output_len"]] * count,
        "output_throughput": 100.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 100.0,
    }


@pytest.fixture
def suite():
    return read_json(nightly.DATA / "suite.json")


@pytest.fixture
def reports(suite):
    return [
        {
            "passed": True,
            "errors": [],
            "server_run_id": str(start),
            "identity": {"checkpoint_id": "content-hash", "engine": {"tp": 8}},
            "runtime": {"vllm": "0.29.0"},
            "outputs": {p["id"]: ["fixed result"] * 5 for p in suite["prompts"]},
            "performance": {name: [sample(w) for _ in range(5)] for name, w in suite["workloads"].items()},
        }
        for start in range(3)
    ]


def test_calibration_and_identity(reports, suite):
    baseline = nightly.finalize_calibration(reports, suite)
    validate_baseline(baseline, reports[0]["identity"], suite)
    changed = copy.deepcopy(reports[0]["identity"])
    changed["engine"]["tp"] = 4
    with pytest.raises(ValueError, match="identity"):
        validate_baseline(baseline, changed, suite)
    changed = copy.deepcopy(reports[0]["identity"])
    changed["checkpoint_id"] = "different"
    with pytest.raises(ValueError, match="identity"):
        validate_baseline(baseline, changed, suite)


@pytest.mark.parametrize("change", ["output", "missing_repeat", "same_start", "config", "noise"])
def test_calibration_rejects_unqualified_runs(reports, suite, change):
    if change == "output":
        reports[1]["outputs"]["json"][2] = "unstable"
    elif change == "missing_repeat":
        reports[1]["outputs"]["json"].pop()
    elif change == "same_start":
        reports[1]["server_run_id"] = reports[0]["server_run_id"]
    elif change == "config":
        reports[1]["identity"] = {"different": True}
    else:
        for i, row in enumerate(reports[1]["performance"]["short-latency"]):
            row["mean_ttft_ms"] *= 1 + i * 0.5
    with pytest.raises(ValueError):
        nightly.finalize_calibration(reports, suite)


@pytest.mark.parametrize("metric", METRICS)
def test_ten_percent_boundary(metric):
    baseline = dict.fromkeys(METRICS, 100.0)
    actual = dict(baseline)
    actual[metric] = 90.0 if metric == "output_throughput" else 110.0
    compare_perf(actual, baseline)
    actual[metric] += -0.01 if metric == "output_throughput" else 0.01
    with pytest.raises(ValueError, match="regression"):
        compare_perf(actual, baseline)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 0, True])
def test_invalid_perf_metrics_fail(suite, value):
    workload = suite["workloads"]["short-latency"]
    result = sample(workload)
    result["mean_ttft_ms"] = value
    with pytest.raises(ValueError):
        validate_perf(result, workload)


def test_short_and_long_responses_cannot_cancel(suite):
    workload = suite["workloads"]["short-latency"]
    result = sample(workload)
    result["output_lens"][0] -= 1
    result["output_lens"][1] += 1
    with pytest.raises(ValueError, match="per-request"):
        validate_perf(result, workload)
    with pytest.raises(ValueError, match="rounds"):
        aggregate_perf([], workload, repeats=3, calibrating=False)


def test_checkpoint_identity_survives_rebuild_but_not_weight_change(dsa_checkpoint, tmp_path):
    source = load_source(str(dsa_checkpoint))
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=8)
    first, second = tmp_path / "first", tmp_path / "second"
    execute_plan(plan, source, str(first))
    execute_plan(plan, source, str(second))
    assert checkpoint_identity(first) == checkpoint_identity(second)
    manifest_path = second / "reduction_manifest.json"
    manifest = read_json(manifest_path)
    manifest["output"]["files"][0]["sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    assert checkpoint_identity(first) != checkpoint_identity(second)


@pytest.mark.parametrize("failure", ["output mismatch", "throughput regression", "baseline missing"])
def test_nightly_failure_reaches_summary_artifact_and_caller(monkeypatch, tmp_path, capsys, failure):
    monkeypatch.chdir(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    config = SimpleNamespace(name="glm-test", model="model", server_cmd=[], server_port=1, envs={})
    monkeypatch.setattr(nightly, "prepare_case", lambda c: {})

    def fail(*args):
        raise ValueError(failure)

    monkeypatch.setattr(nightly, "run_round", fail)
    if failure == "baseline missing":
        monkeypatch.setattr(nightly, "prepare_case", fail)
    with pytest.raises(ValueError, match=failure):
        nightly.run_nightly(config, lambda **kw: nullcontext(object()))
    result = read_json(tmp_path / "benchmark_results/glm-test.json")
    assert result["passed"] is False
    assert failure in result["error"]
    assert "FAIL" in summary.read_text()
    assert "::error" in capsys.readouterr().out


def test_nightly_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(name="glm-test", model="model", server_cmd=[], server_port=1, envs={})
    monkeypatch.setattr(nightly, "prepare_case", lambda c: {})
    monkeypatch.setattr(nightly, "run_round", lambda *a: {"passed": True, "errors": []})
    nightly.run_nightly(config, lambda **kw: nullcontext(object()))
    assert read_json(tmp_path / "benchmark_results/glm-test.json")["passed"]


def test_performance_runs_after_output_failure(monkeypatch, tmp_path, suite, reports):
    baseline = nightly.finalize_calibration(reports, suite)
    calls = []

    class Bench:
        def __init__(self, model, port, config, **kwargs):
            calls.append(config)
            self.result = sample(config)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "tools.vllm_bench", SimpleNamespace(VllmbenchRunner=Bench))

    def bad_completion(*args):
        raise ValueError("injected output error")

    monkeypatch.setattr(nightly, "curl_completion", bad_completion)
    state = {
        "identity": reports[0]["identity"],
        "runtime": {},
        "suite": suite,
        "baseline": baseline,
        "model": "reduced",
        "report_dir": tmp_path,
        "requests": [{"id": p["id"], "request": {}} for p in suite["prompts"]],
    }
    server = SimpleNamespace(port=123, url_for=lambda *args: "http://localhost/v1/completions")
    with pytest.raises(ValueError, match="injected output error"):
        nightly.run_round(state, server)
    assert len(calls) == 8  # Two workloads, each with a warmup and three measured rounds.
    result = read_json(tmp_path / "report.json")
    assert not result["passed"]
    assert all(len(samples) == 3 for samples in result["performance"].values())


def test_retrieval_token_lengths_and_suffix(suite):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode("utf-8"))

    requests = nightly.compile_requests(suite, Tokenizer())
    assert len(requests) == 8
    for entry, prompt in zip(requests[-3:], suite["prompts"][-3:]):
        tokens = entry["request"]["prompt"]
        assert len(tokens) == prompt["input_tokens"]
        suffix = list(prompt["suffix"].encode())
        assert tokens[-len(suffix) :] == suffix


def test_calibration_software_must_stay_fixed(suite, reports):
    reports[1]["runtime"]["vllm"] = "changed during calibration"
    with pytest.raises(ValueError, match="software changed"):
        nightly.finalize_calibration(reports, suite)


def test_unstable_calibration_still_collects_performance(monkeypatch, tmp_path, suite):
    calls = []

    class Bench:
        def __init__(self, model, port, config, **kwargs):
            calls.append(config)
            self.result = sample(config)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "tools.vllm_bench", SimpleNamespace(VllmbenchRunner=Bench))
    texts = iter(str(i) for i in range(40))
    monkeypatch.setattr(
        nightly,
        "curl_completion",
        lambda *a: {
            "choices": [{"index": 0, "text": next(texts), "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 64},
        },
    )
    state = {
        "identity": {},
        "runtime": {},
        "suite": suite,
        "model": "reduced",
        "report_dir": tmp_path,
        "requests": [{"id": p["id"], "request": {"prompt": [1, 2, 3], "max_tokens": 64}} for p in suite["prompts"]],
    }
    server = SimpleNamespace(port=123, url_for=lambda *a: "http://127.0.0.1/v1/completions")
    with pytest.raises(ValueError, match="Unstable output within one server"):
        nightly.run_round(state, server, calibrating=True)
    assert len(calls) == 12
    result = read_json(tmp_path / "report.json")
    assert not result["passed"]
    assert all(len(samples) == 5 for samples in result["performance"].values())


@pytest.mark.parametrize("fault", [None, "output", "performance"])
def test_real_gate_decision_propagates_to_nightly(monkeypatch, tmp_path, suite, reports, fault):
    """Inject only external response data; exercise the complete decision path."""
    monkeypatch.chdir(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    baseline = nightly.finalize_calibration(reports, suite)
    if fault == "output":
        baseline["outputs"]["chinese"] = "deliberately changed expected text"

    class Bench:
        def __init__(self, model, port, config, **kwargs):
            self.result = sample(config)
            if fault == "performance":
                self.result["output_throughput"] = 89.9

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "tools.vllm_bench", SimpleNamespace(VllmbenchRunner=Bench))
    monkeypatch.setattr(
        nightly,
        "curl_completion",
        lambda *a: {
            "choices": [{"index": 0, "text": "fixed result", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 64},
        },
    )
    state = {
        "identity": baseline["identity"],
        "runtime": {},
        "suite": suite,
        "baseline": baseline,
        "model": "reduced",
        "report_dir": tmp_path,
        "requests": [{"id": p["id"], "request": {"prompt": [1, 2, 3], "max_tokens": 64}} for p in suite["prompts"]],
    }
    monkeypatch.setattr(nightly, "prepare_case", lambda c: state)
    server = SimpleNamespace(port=123, url_for=lambda *a: "http://127.0.0.1/v1/completions")
    config = SimpleNamespace(name="glm-test", model="reduced", server_cmd=[], server_port=123, envs={})
    factory = lambda **kw: nullcontext(server)
    if fault:
        with pytest.raises(ValueError, match="regression"):
            nightly.run_nightly(config, factory)
    else:
        nightly.run_nightly(config, factory)
    assert read_json(tmp_path / "benchmark_results/glm-test.json")["passed"] is (fault is None)
    assert ("FAIL" if fault else "PASS") in summary.read_text()
