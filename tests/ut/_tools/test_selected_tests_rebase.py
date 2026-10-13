# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/_selected_tests.yaml"


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()


def _commit(repo, name, content):
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "--no-gpg-sign", "-qm", content)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    remote, repo = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "CI Rebase Test")
    _git(repo, "config", "user.email", "ci-rebase-test@example.invalid")
    _git(repo, "remote", "add", "origin", str(remote))
    common = _commit(repo, "policy.po", "shared translation\n")
    _git(repo, "checkout", "-b", "feature")
    feature = _commit(repo, "feature.txt", "package version feature\n")
    _git(repo, "checkout", "main")
    base = _commit(repo, "main.txt", "base snapshot update\n")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "feature")
    return SimpleNamespace(path=repo, common=common, feature=feature, base=base)


def _run_step(repo, fixed_sha=""):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    step = next(step for step in steps if step.get("name", "").startswith("Rebase on "))
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=repo,
        env={**os.environ, "BASE_REF": "main", "FIXED_BASE_SHA": fixed_sha},
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_resolved_merge_is_tested_without_replaying_conflicting_commits(repository, fixed):
    repo = repository.path
    _git(repo, "checkout", "main")
    base = _commit(repo, "policy.po", "shared translation\nmain release translation\n")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "feature")
    _commit(repo, "policy.po", "shared translation\nrelease branch version translation\n")
    merge = subprocess.run(["git", "merge", "--no-commit", "main"], cwd=repo, capture_output=True)
    assert merge.returncode == 1
    (repo / "policy.po").write_text(
        "shared translation\nrelease branch version translation\nmain release translation\n"
    )
    _git(repo, "add", "policy.po")
    _git(repo, "commit", "--no-gpg-sign", "-qm", "retain both translations")
    before = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    result = _run_step(repo, base if fixed else "")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repo, "rev-parse", "HEAD") == before
    assert _git(repo, "rev-parse", "HEAD^{tree}") == tree
    assert "main release translation" in (repo / "policy.po").read_text()
    assert "release branch version translation" in (repo / "policy.po").read_text()


def test_checkout_missing_base_is_still_rebased(repository):
    repo = repository.path
    result = _run_step(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repo, "rev-parse", "HEAD^") == repository.base
    assert _git(repo, "rev-parse", "HEAD") != repository.feature
    assert (repo / "feature.txt").read_text() == "package version feature\n"
    assert (repo / "main.txt").read_text() == "base snapshot update\n"


def test_fixed_snapshot_is_used_when_remote_main_advances(repository):
    repo = repository.path
    _git(repo, "checkout", "main")
    newer = _commit(repo, "newer.txt", "new main beyond requested snapshot\n")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "feature")
    result = _run_step(repo, repository.base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repo, "rev-parse", "HEAD^") == repository.base
    assert not (repo / "newer.txt").exists()
    assert _git(repo, "rev-parse", "origin/main") == newer


def test_invalid_snapshot_fails_before_changing_checkout(repository):
    repo = repository.path
    before = _git(repo, "rev-parse", "HEAD")
    result = _run_step(repo, "0" * 40)
    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD") == before


def test_real_rebase_conflict_is_not_hidden(repository):
    repo = repository.path
    _git(repo, "checkout", "main")
    _commit(repo, "policy.po", "shared translation\nmain release translation\n")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "feature")
    _commit(repo, "policy.po", "shared translation\nrelease branch version translation\n")
    result = _run_step(repo)
    assert result.returncode != 0
    assert "CONFLICT" in result.stdout
