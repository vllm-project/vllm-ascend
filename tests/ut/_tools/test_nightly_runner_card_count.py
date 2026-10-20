# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = (
    "_e2e_nightly_single_node.yaml",
    "_e2e_nightly_single_node_models.yaml",
)
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="Bash is required for workflow shell steps")


def _run_max_jobs_step(
    workflow: str, runner_label: str, tmp_path: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    workflow_path = REPO_ROOT / ".github" / "workflows" / workflow
    config = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    step = next(
        item for item in config["jobs"]["e2e-nightly"]["steps"] if item["name"] == "Set MAX_JOBS based on runner"
    )
    script = step["run"].replace("${{ inputs.runner }}", runner_label)
    environment = os.environ.copy()
    environment.update(VLLM_CI_RUNNER=runner_label, GITHUB_ENV="github-env")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    return result, tmp_path / "github-env"


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize(
    ("runner_label", "expected_jobs"),
    [
        ("linux-aarch64-a2b3-1", 23),
        ("linux-aarch64-a2b3-4", 92),
        ("linux-aarch64-a2-4-binding", 92),
        ("linux-aarch64-a3-4-binding", 92),
        ("linux-aarch64-a3-800i-16", 368),
        ("linux-aarch64-310p-8", 184),
        ("linux-aarch64-a5-8", 184),
        ("linux-aarch64-a2-08-binding", 184),
    ],
)
def test_max_jobs_from_runner_label(workflow: str, runner_label: str, expected_jobs: int, tmp_path: Path):
    result, output = _run_max_jobs_step(workflow, runner_label, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == f"MAX_JOBS={expected_jobs}\n"


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize(
    "runner_label",
    ["", "linux-aarch64-a2b3", "linux-aarch64-a2-x-binding", "linux-aarch64-a2-4-binding-extra"],
)
def test_max_jobs_rejects_unparseable_runner_label(workflow: str, runner_label: str, tmp_path: Path):
    result, output = _run_max_jobs_step(workflow, runner_label, tmp_path)

    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert not output.exists()
