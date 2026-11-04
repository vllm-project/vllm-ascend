# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("demo_scope", ["true", "false"])
def test_source_snapshot_rebases_onto_upstream_for_every_scope(tmp_path: Path, demo_scope: str):
    checkout, base_sha, _ = _make_repositories(tmp_path)

    result = _run_source_snapshot(checkout, base_sha, demo_scope)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "github-output").read_text() == f"base_sha={base_sha}\n"
    assert _git(checkout, "merge-base", "HEAD", base_sha) == base_sha
    assert (checkout / ".github/vllm-main-verified.commit").read_text() == "new-vllm\n"
    assert (checkout / "model_runner.py").read_text() == "new interface\n"
    assert (checkout / "feature.py").read_text() == "PR change\n"


def test_source_snapshot_keeps_upstream_cache_workflow_guard(tmp_path: Path):
    checkout, base_sha, event_base_sha = _make_repositories(tmp_path, change_cache_workflow=True)
    original_head = _git(checkout, "rev-parse", "HEAD")

    result = _run_source_snapshot(checkout, event_base_sha, "true")

    assert result.returncode != 0
    assert "CSRC build workflows changed on main" in result.stdout
    assert _git(checkout, "rev-parse", "HEAD") == original_head
    assert not (tmp_path / "github-output").exists()
    assert _git(checkout, "rev-parse", "FETCH_HEAD") == base_sha


def _make_repositories(tmp_path: Path, change_cache_workflow: bool = False) -> tuple[Path, str, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--initial-branch=main")
    (upstream / ".github/workflows").mkdir(parents=True)
    (upstream / ".github/vllm-main-verified.commit").write_text("old-vllm\n")
    (upstream / "model_runner.py").write_text("old interface\n")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "Initial paired source and dependency")
    event_base_sha = _git(upstream, "rev-parse", "HEAD")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", str(upstream), str(checkout))
    _git(checkout, "checkout", "-b", "ascendstore-v1")
    (checkout / "feature.py").write_text("PR change\n")
    _git(checkout, "add", "feature.py")
    _git(checkout, "commit", "-m", "PR feature")

    (upstream / ".github/vllm-main-verified.commit").write_text("new-vllm\n")
    (upstream / "model_runner.py").write_text("new interface\n")
    if change_cache_workflow:
        (upstream / ".github/workflows/_ensure_csrc_cache.yaml").write_text("new cache workflow\n")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "Advance source and dependency together")
    return checkout, _git(upstream, "rev-parse", "HEAD"), event_base_sha


def _run_source_snapshot(checkout: Path, event_base_sha: str, demo_scope: str) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/pr_test.yaml").read_text())
    step = next(step for step in workflow["jobs"]["select-tests"]["steps"] if step.get("id") == "source-snapshot")
    environment = os.environ.copy()
    environment.update(
        BASE_REF="main",
        EVENT_BASE_SHA=event_base_sha,
        ASCENDSTORE_V1_ONLY=demo_scope,
        GITHUB_WORKSPACE=str(checkout),
        GITHUB_OUTPUT=str(checkout.parent / "github-output"),
        GIT_CONFIG_GLOBAL=str(checkout.parent / "gitconfig"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _git(repository: Path, *arguments: str) -> str:
    configuration = ["-c", "user.name=CI Test", "-c", "user.email=ci-test@example.com"]
    configuration += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    result = subprocess.run(
        ["git", "-C", str(repository), *configuration, *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()
