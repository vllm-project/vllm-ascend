# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate test ordering and the precise CI scope without importing an NPU serving runtime."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LAYERWISE_TEST = "tests/e2e/pull_request/one_card/test_ascendstore_v1_layerwise.py"


@pytest.mark.parametrize("failure", [None, "production", "v1"])
def test_both_correctness_runs_gate_fresh_uninstrumented_performance_servers(tmp_path, failure):
    source = ast.parse((REPO_ROOT / LAYERWISE_TEST).read_text())
    entry = next(node for node in source.body if isinstance(node, ast.FunctionDef))
    # Execute the actual orchestration with boundary doubles; do not import pytest's NPU E2E conftest.
    module = ast.Module(body=[entry], type_ignores=[])
    launches, operations, reports = [], [], []
    connectors = (("production", "production", "module"), ("v1", "v1", "module"))

    def exercise(path, model, tokenizer, inputs, warmup, connector, observations, **kwargs):
        assert kwargs["use_layerwise"] is True
        assert kwargs["worker_extension"] == "layerwise-extension"
        assert path.is_dir()
        validator = kwargs.get("validate")
        launches.append((connector[0], "validation" if validator is not None else "performance"))
        if validator is not None:
            validator(SimpleNamespace(implementation=connector[0]), SimpleNamespace(master_address="127.0.0.1:1"))

    def rpc(server, method, **kwargs):
        operations.append((server.implementation, method))
        if method == "finish_layerwise_validation" and failure == server.implementation:
            raise AssertionError("Injected validation failure")
        return {"workers": [{"status": "passed"}]}

    namespace: dict[str, Any] = {
        "Path": Path,
        "Any": object,
        "shutil": SimpleNamespace(which=lambda _: "master"),
        "import_module": lambda _: SimpleNamespace(
            MOONCAKE_LAYERWISE_CLIENT_METHODS=("start",), MooncakeDistributedStore=SimpleNamespace(start=lambda: None)
        ),
        "MODEL": "model",
        "maybe_model_redirect": lambda model: model,
        "get_tokenizer": lambda _: None,
        "make_requests": lambda *args, **kwargs: [],
        "CONCURRENCY": 8,
        "VALIDATION_REQUEST_COUNT": 2,
        "make_report": lambda *args: {"runs": []},
        "CONNECTORS": connectors,
        "CROSSOVER_RUNS": connectors + connectors[::-1],
        "LAYERWISE_WORKER_EXTENSION": "layerwise-extension",
        "_exercise_connector": exercise,
        "call_worker_rpc": rpc,
        "reset_local_prefix_cache": lambda _: {},
        "run_benchmark": lambda *args: {},
        "assert_prompt_sources": Mock(),
        "compare_runs": lambda _: {},
        "time": SimpleNamespace(monotonic=lambda: 1),
        "write_report": lambda *args: reports.append(args[-1]),
    }
    exec(compile(module, str(REPO_ROOT / LAYERWISE_TEST), "exec"), namespace)
    test = namespace[entry.name]
    if failure is None:
        test(tmp_path)
        assert launches == [(name, "validation") for name in ("production", "v1")] + [
            (name, "performance") for name in ("production", "v1", "v1", "production")
        ]
        assert reports[0]["status"] == "validated_and_measured"
    else:
        with pytest.raises(AssertionError, match="Injected validation failure"):
            test(tmp_path)
        assert all(phase == "validation" for _, phase in launches)
        assert reports[0]["status"] == "incomplete"
    for implementation, _ in launches[:2]:
        methods = [method for name, method in operations if name == implementation]
        assert methods == [
            "reset_ascendstore_benchmark_pool",
            "start_layerwise_validation",
            "drain_ascendstore_benchmark_stores",
            "arm_layerwise_destination_poisoning",
            "finish_layerwise_validation",
            "remove_layerwise_validation",
        ]


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
