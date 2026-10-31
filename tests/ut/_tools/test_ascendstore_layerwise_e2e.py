# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate test ordering and the precise CI scope without importing an NPU serving runtime."""

import ast
import json
import os
import subprocess
import sys
from collections.abc import Callable
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


@pytest.mark.parametrize("failure", [None, "production", "v1", "byte-comparison"])
def test_layerwise_checks_accuracy_before_balanced_performance_and_keeps_failure_reports(tmp_path, failure):
    entry_source = ast.parse((REPO_ROOT / LAYERWISE_TEST).read_text())
    entry = next(node for node in entry_source.body if isinstance(node, ast.FunctionDef))
    shared_source = ast.parse((REPO_ROOT / "tests/e2e/pull_request/ascendstore_v1_benchmark.py").read_text())
    comparison = next(
        node for node in shared_source.body if isinstance(node, ast.FunctionDef) and node.name == "run_comparison"
    )
    # Execute real orchestration with server and byte-comparison boundaries replaced.
    module = ast.Module(body=[comparison, entry], type_ignores=[])
    launches, reports = [], []
    connectors = (("production", "production"), ("v1", "v1"))
    compare_bytes = Mock(
        side_effect=AssertionError("Injected serving failure") if failure == "byte-comparison" else None
    )
    compare = Mock(return_value={"output_and_performance_comparison": True})

    def exercise(path, connector, observations, accuracy_directory=None, **kwargs):
        assert kwargs == {
            "model_path": "model",
            "tokenizer": None,
            "inputs": [],
            "warmup": [],
            "use_layerwise": True,
            "worker_extension": "layerwise-extension",
        }
        assert path.is_dir()
        phase = "accuracy" if accuracy_directory is not None else "performance"
        launches.append((phase, connector[0]))
        if phase == "accuracy":
            observations.append({"stored_objects": {"key": "saved-bytes"}})
            if connector[0] == failure:
                raise AssertionError("Injected serving failure")
        else:
            assert compare_bytes.called, "Performance started before the production/v1 byte comparison"
            observations.append({"cold_store": {}, "external_load": {}})

    namespace: dict[str, Any] = {
        "Path": Path,
        "Any": Any,
        "Callable": Callable,
        "partial": partial,
        "shutil": SimpleNamespace(which=lambda _: "master"),
        "MODEL": "model",
        "maybe_model_redirect": lambda model: model,
        "get_tokenizer": lambda _: None,
        "make_requests": lambda *args, **kwargs: [],
        "CONCURRENCY": 8,
        "make_report": lambda *args: {"runs": [], "accuracy": []},
        "CONNECTORS": connectors,
        "CROSSOVER_RUNS": connectors + connectors[::-1],
        "LAYERWISE_WORKER_EXTENSION": "layerwise-extension",
        "_exercise_connector": exercise,
        "compare_saved_bytes": compare_bytes,
        "compare_runs": compare,
        "time": SimpleNamespace(monotonic=lambda: 1),
        "write_report": lambda *args: reports.append(args[-1]),
    }
    exec(compile(module, str(REPO_ROOT / LAYERWISE_TEST), "exec"), namespace)
    test = namespace[entry.name]
    if failure is None:
        test(tmp_path)
        assert launches == [
            ("accuracy", "production"),
            ("accuracy", "v1"),
            ("performance", "production"),
            ("performance", "v1"),
            ("performance", "v1"),
            ("performance", "production"),
        ]
        assert reports[0]["accuracy_status"] == "passed"
        assert reports[0]["status"] == "measured"
        assert reports[0]["comparisons"] == compare.return_value
        compare.assert_called_once_with(reports[0]["runs"])
        compare_bytes.assert_called_once()
        assert all(run["measurements"] and "duration_seconds" in run for run in reports[0]["runs"])
    else:
        with pytest.raises(AssertionError, match="Injected serving failure"):
            test(tmp_path)
        expected = [("accuracy", "production")]
        if failure in ("v1", "byte-comparison"):
            expected.append(("accuracy", "v1"))
        assert launches == expected
        assert reports[0]["status"] == "incomplete"
        assert not reports[0]["runs"]
        if failure == "byte-comparison":
            compare_bytes.assert_called_once()
        else:
            compare_bytes.assert_not_called()
        compare.assert_not_called()
    assert reports[0]["scenario"] == "layerwise"


@pytest.mark.parametrize("implementation", ["production", "v1"])
def test_worker_drains_the_layerwise_owner_not_the_bulk_sender(monkeypatch, implementation):
    extension = AscendStoreLayerwiseWorkerExtension()
    store = object()
    backend = SimpleNamespace(store=store, _store=store)
    worker = SimpleNamespace(
        _drain_attention_transfers=Mock(), fence_previous_store=Mock(), store=SimpleNamespace(pending=False)
    )
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


def test_demo_ci_selects_all_comparisons_with_distinct_artifacts_and_real_timeouts(tmp_path):
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
    expected = {
        "tests/e2e/pull_request/one_card/test_ascendstore_v1.py": ("bulk", "linux-aarch64-a2b3-1", 1),
        LAYERWISE_TEST: ("layerwise", "linux-aarch64-a2b3-1", 1),
        "tests/e2e/pull_request/four_card/test_ascendstore_v1_tp_mismatch.py": (
            "tp-mismatch",
            "linux-aarch64-a2b3-4",
            4,
        ),
    }
    assert len(groups) == len(expected)
    assert {group["tests"] for group in groups} == set(expected)
    artifact_ids = set()
    for group in groups:
        scenario, runner, cards = expected[group["tests"]]
        assert group["runner"] == runner and group["num_npus"] == cards
        assert group["npu_type"] == "a2"
        assert group["partition"].startswith(scenario + "-")
        assert group["image_tag"] and group["csrc_cache_target"] == "a2-arm64-ubuntu"
        artifact_ids.add((group["npu_type"], group["num_npus"], group["partition"]))
    assert len(artifact_ids) == len(groups)
    assert json.loads(outputs["csrc_cache_target_ids"]) == ["a2-arm64-ubuntu"]
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
    assert steps.index(install) < steps.index(build) < steps.index(run)
    assert "MC_FORCE_TCP" not in source
    assert "check_mooncake_layerwise" not in source
    assert not (REPO_ROOT / ".github/workflows/scripts/check_mooncake_layerwise.py").exists()
    subprocess.run(["bash", "-n", "-c", install["run"]], check=True, capture_output=True)
