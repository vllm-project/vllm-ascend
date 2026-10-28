# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate test ordering and the precise CI scope without importing an NPU serving runtime."""

import ast
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise import AscendStoreLayerwiseWorkerExtension

REPO_ROOT = Path(__file__).resolve().parents[3]
LAYERWISE_TEST = "tests/e2e/pull_request/one_card/test_ascendstore_v1_layerwise.py"


@pytest.mark.parametrize("failure", [None, "production", "v1"])
def test_direct_layerwise_serving_reports_outputs_and_performance_without_probe(tmp_path, failure):
    source = ast.parse((REPO_ROOT / LAYERWISE_TEST).read_text())
    entry = next(node for node in source.body if isinstance(node, ast.FunctionDef))
    # Execute the actual orchestration with boundary doubles; do not import pytest's NPU E2E conftest.
    module = ast.Module(body=[entry], type_ignores=[])
    launches, reports = [], []
    connectors = (("production", "production"), ("v1", "v1"))

    def exercise(path, model, tokenizer, inputs, warmup, connector, observations, **kwargs):
        diagnostic = kwargs.pop("after_measurements")
        assert kwargs == {"use_layerwise": True, "worker_extension": "layerwise-extension"}
        assert path.is_dir()
        launches.append(connector[0])
        observations.append({"cold_store": {}, "external_load": {}})
        if connector[0] == failure:
            raise AssertionError("Injected serving failure")
        if connector[0] == "v1":
            assert diagnostic is not None
            diagnostic("server")
        else:
            assert diagnostic is None

    compare = Mock(return_value={"output_and_performance_comparison": True})

    namespace: dict[str, Any] = {
        "Path": Path,
        "Any": object,
        "partial": partial,
        "shutil": SimpleNamespace(which=lambda _: "master"),
        "MODEL": "model",
        "maybe_model_redirect": lambda model: model,
        "get_tokenizer": lambda _: None,
        "make_requests": lambda *args, **kwargs: [],
        "CONCURRENCY": 8,
        "make_report": lambda *args: {"runs": []},
        "CROSSOVER_RUNS": connectors + connectors[::-1],
        "LAYERWISE_WORKER_EXTENSION": "layerwise-extension",
        "_exercise_connector": exercise,
        "_diagnose_external_load": lambda server, **kwargs: kwargs["run"]["external_load_diagnostic"].update(done=True),
        "compare_runs": compare,
        "time": SimpleNamespace(monotonic=lambda: 1),
        "write_report": lambda *args: reports.append(args[-1]),
    }
    exec(compile(module, str(REPO_ROOT / LAYERWISE_TEST), "exec"), namespace)
    test = namespace[entry.name]
    if failure is None:
        test(tmp_path)
        assert launches == ["production", "v1", "v1", "production"]
        assert reports[0]["status"] == "measured"
        assert reports[0]["comparisons"] == compare.return_value
        assert [run.get("external_load_diagnostic") for run in reports[0]["runs"]] == [
            None,
            {"done": True},
            {"done": True},
            None,
        ]
        compare.assert_called_once_with(reports[0]["runs"])
    else:
        with pytest.raises(AssertionError, match="Injected serving failure"):
            test(tmp_path)
        assert launches == (["production"] if failure == "production" else ["production", "v1"])
        assert reports[0]["status"] == "incomplete"
        compare.assert_not_called()
    assert reports[0]["scenario"] == "layerwise"
    assert "validation" not in reports[0]
    assert all(run["measurements"] and "duration_seconds" in run for run in reports[0]["runs"])


@pytest.mark.parametrize("implementation", ["production", "v1"])
def test_worker_drains_the_layerwise_owner_not_the_bulk_sender(monkeypatch, implementation):
    extension = AscendStoreLayerwiseWorkerExtension()
    store = object()
    backend = SimpleNamespace(store=store, _store=store)
    worker = SimpleNamespace(_drain_attention_transfers=Mock(), fence_previous_store=Mock(), _pending_store_batch=None)
    monkeypatch.setattr(extension, "_layerwise_resources", lambda: (worker, backend, implementation))
    assert extension._drain_stores()[:2] == (backend, store)
    if implementation == "production":
        worker._drain_attention_transfers.assert_called_once_with(drain_recv=False, full=True)
        worker.fence_previous_store.assert_not_called()
    else:
        worker.fence_previous_store.assert_called_once_with()
        worker._drain_attention_transfers.assert_not_called()


def test_output_differences_are_reported_alongside_performance_results():
    benchmark = REPO_ROOT / "tests/e2e/pull_request/ascendstore_v1_benchmark.py"
    source = ast.parse(benchmark.read_text())
    compare = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "compare_runs")
    namespace: dict[str, Any] = {
        "Any": Any,
        "CONNECTORS": (("production", ""), ("v1", "")),
        "SUMMARY_METRICS": ("median_ttft_ms", "output_throughput"),
        "median": median,
    }
    exec(compile(ast.Module(body=[compare], type_ignores=[]), str(benchmark), "exec"), namespace)
    runs = []
    for run_index, implementation in enumerate(("production", "v1"), start=1):
        measurement = {
            "generated_texts": ["same", implementation],
            "median_ttft_ms": 100 / run_index,
            "output_throughput": 10 * run_index,
        }
        runs.append(
            {
                "run": run_index,
                "implementation": implementation,
                "measurements": [{"round": 1, "cold_store": measurement, "external_load": measurement}],
            }
        )
    results = namespace["compare_runs"](runs)
    for phase in ("cold_store", "external_load"):
        comparison = results[phase]
        assert comparison["text_comparisons"][0]["generated_texts_equal"] is False
        assert comparison["text_comparisons"][0]["different_request_indices"] == [1]
        assert comparison["median_ttft_ms"]["v1_over_production"] == 0.5
        assert comparison["output_throughput"]["v1_over_production"] == 2.0


def test_layerwise_byte_probe_and_instrumentation_entry_points_are_removed():
    assert not (REPO_ROOT / "tests/e2e/common/kv_pool/ascendstore_layerwise_probe.py").exists()
    worker = (REPO_ROOT / "tests/e2e/pull_request/one_card/ascendstore_v1_layerwise.py").read_text()
    assert "LayerwiseRangeProbe" not in worker
    assert "validation" not in worker
    assert "kv_caches" not in worker
    shared = ast.parse((REPO_ROOT / "tests/e2e/pull_request/one_card/test_ascendstore_v1.py").read_text())
    exercise = next(
        node for node in shared.body if isinstance(node, ast.FunctionDef) and node.name == "_exercise_connector"
    )
    assert "validate" not in [argument.arg for argument in exercise.args.kwonlyargs]


def test_demo_ci_selects_only_one_tp1_layerwise_e2e_and_has_real_timeouts(tmp_path):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/pr_test.yaml").read_text())
    step = next(step for step in workflow["jobs"]["select-tests"]["steps"] if step.get("id") == "scope-v1")
    script = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output_path = tmp_path / "github-output"
    environment = {**os.environ, "GITHUB_OUTPUT": str(output_path)}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    groups = json.loads(outputs["test_groups"])
    assert len(groups) == 1
    assert groups[0]["runner"] == "linux-aarch64-a2b3-1"
    assert groups[0]["num_npus"] == 1
    assert groups[0]["tests"].split() == [LAYERWISE_TEST]
    assert outputs["has_tests"] == "true"
    assert workflow["jobs"]["run-selected-tests"]["with"]["timeout-minutes"] == 120
    selected = yaml.safe_load((REPO_ROOT / ".github/workflows/_selected_tests.yaml").read_text())
    test_step = next(
        step
        for step in selected["jobs"]["selected-tests"]["steps"]
        if step.get("name") == "Run selected tests with device"
    )
    assert "test_ascendstore_v1_layerwise.py" in test_step["timeout-minutes"]
    assert "&& 90 || inputs.timeout-minutes" in test_step["timeout-minutes"]


def test_ci_installs_new_layerwise_sdk_and_runs_npu_tests_without_cpu_preflight():
    source = (REPO_ROOT / ".github/workflows/_selected_tests.yaml").read_text()
    workflow = yaml.safe_load(source)
    steps = workflow["jobs"]["selected-tests"]["steps"]
    install = next(step for step in steps if step.get("name") == "Install Mooncake wheel")
    build = next(step for step in steps if step.get("name") == "Install vllm-project/vllm-ascend with device")
    run = next(step for step in steps if step.get("name") == "Run selected tests with device")
    assert install["env"]["MOONCAKE_WHEEL"] == (
        "${{ contains(matrix.group.tests, '" + LAYERWISE_TEST + "') && "
        "'mooncake-transfer-engine-npu==0.3.13.post1' || 'mooncake-transfer-engine-npu==0.3.11.post1' }}"
    )
    assert '"$MOONCAKE_WHEEL"' in install["run"]
    assert steps.index(install) + 1 == steps.index(build) < steps.index(run)
    assert "MC_FORCE_TCP" not in source
    assert "check_mooncake_layerwise" not in source
    assert not (REPO_ROOT / ".github/workflows/scripts/check_mooncake_layerwise.py").exists()
    subprocess.run(["bash", "-n", "-c", install["run"]], check=True, capture_output=True)
