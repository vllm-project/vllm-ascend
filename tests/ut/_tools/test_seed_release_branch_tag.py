# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/seed_release_branch_tag.py"


@pytest.fixture(scope="module")
def seed():
    spec = importlib.util.spec_from_file_location("seed_release_branch_tag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()


def _commit(repo, text):
    (repo / "source.txt").write_text(text)
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "--no-gpg-sign", "-qm", text)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, request):
    branch = getattr(request, "param", "releases/v0.26.0rc")
    remote, work = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(work))
    _git(work, "config", "user.name", "Release Tag Test")
    _git(work, "config", "user.email", "release-tag-test@example.invalid")
    _git(work, "remote", "add", "origin", str(remote))
    _commit(work, "old release")
    _git(work, "tag", "-a", "v0.19.1rc1", "-m", "old release")
    release_sha = _commit(work, "release branch point")
    _git(work, "branch", branch)
    _git(work, "push", "origin", "main", branch, "--tags")
    main_sha = _commit(work, "main advances independently")
    _git(work, "push", "origin", "main")
    return SimpleNamespace(work=work, remote=remote, branch=branch, release_sha=release_sha, main_sha=main_sha)


@pytest.mark.parametrize("repository", ["releases/v0.26.0rc", "releases/v0.25.1rc", "releases/v0.30.0"], indirect=True)
def test_new_release_gets_its_own_reachable_baseline(seed, repository):
    version = seed.release_version(repository.branch)
    tag = f"v{version}rc0"
    assert _git(repository.work, "describe", "--tags", "--abbrev=0", repository.release_sha) == "v0.19.1rc1"
    assert seed.seed_release_tag(repository.work, repository.branch) == f"Created {tag} at {repository.release_sha}"
    assert _git(repository.remote, "rev-parse", f"refs/tags/{tag}") == repository.release_sha
    assert repository.release_sha != repository.main_sha
    _git(repository.work, "fetch", "--tags", "origin")
    assert _git(repository.work, "describe", "--tags", "--abbrev=0", repository.release_sha) == tag


def test_delayed_create_event_tags_the_creation_commit(seed, repository):
    _git(repository.work, "checkout", repository.branch)
    newer_tip = _commit(repository.work, "release advances before job runs")
    _git(repository.work, "push", "origin", repository.branch)
    seed.seed_release_tag(repository.work, repository.branch, repository.release_sha)
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha
    assert repository.release_sha != newer_tip
    _git(repository.work, "fetch", "--tags", "origin")
    assert _git(repository.work, "describe", "--tags", "--long").startswith("v0.26.0rc0-1-g")


@pytest.mark.parametrize(
    "tag,annotated", [("v0.26.0rc0", False), ("v0.26.0rc1", True), ("v0.26.0", False), ("v0.26.0.post1", True)]
)
def test_existing_version_tags_are_preserved(seed, repository, tag, annotated):
    _git(repository.work, "checkout", repository.branch)
    if annotated:
        _git(repository.work, "tag", "-a", tag, "-m", "published version")
    else:
        _git(repository.work, "tag", tag)
    _git(repository.work, "push", "origin", f"refs/tags/{tag}")
    before = _git(repository.remote, "show-ref", "--tags")
    assert seed.seed_release_tag(repository.work, repository.branch) == f"Already versioned by {tag}"
    assert _git(repository.remote, "show-ref", "--tags") == before


def test_repeat_after_new_release_commit_does_not_move_rc0(seed, repository):
    seed.seed_release_tag(repository.work, repository.branch)
    _git(repository.work, "checkout", repository.branch)
    _commit(repository.work, "next nightly")
    _git(repository.work, "push", "origin", repository.branch)
    assert seed.seed_release_tag(repository.work, repository.branch) == "Already versioned by v0.26.0rc0"
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha


def test_unrelated_existing_version_tag_is_not_overwritten(seed, repository):
    _git(repository.remote, "update-ref", "refs/tags/v0.26.0rc0", repository.main_sha)
    with pytest.raises(ValueError, match="outside the release branch history"):
        seed.seed_release_tag(repository.work, repository.branch)
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.main_sha


def test_creation_commit_must_belong_to_the_target_branch(seed, repository):
    with pytest.raises(ValueError, match="not in the current release branch history"):
        seed.seed_release_tag(repository.work, repository.branch, repository.main_sha)
    assert _git(repository.remote, "tag", "--list", "v0.26.0rc0") == ""


@pytest.mark.parametrize(
    "branch", ["main", "releases/v1.2", "releases/v01.2.3", "releases/v1.2.3rc1", "releases/v1.2.3;echo unsafe"]
)
def test_invalid_branch_is_rejected_before_git_io(seed, tmp_path, monkeypatch, branch):
    def unexpected_git(*args):
        pytest.fail("invalid branch reached git")

    monkeypatch.setattr(seed, "git", unexpected_git)
    with pytest.raises(ValueError, match="Expected a release branch"):
        seed.seed_release_tag(tmp_path, branch)


@pytest.mark.parametrize("sha", ["abcd", "0" * 39, "z" * 40])
def test_invalid_creation_sha_is_rejected_before_git_io(seed, tmp_path, monkeypatch, sha):
    def unexpected_git(*args):
        pytest.fail("invalid creation SHA reached git")

    monkeypatch.setattr(seed, "git", unexpected_git)
    with pytest.raises(ValueError, match="full SHA-1"):
        seed.seed_release_tag(tmp_path, "releases/v0.26.0rc", sha)


def test_failed_push_can_be_retried_in_the_same_checkout(seed, repository):
    hook = repository.remote / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError):
        seed.seed_release_tag(repository.work, repository.branch)
    assert _git(repository.work, "tag", "--list", "v0.26.0rc0") == ""
    assert _git(repository.remote, "tag", "--list", "v0.26.0rc0") == ""
    hook.unlink()
    seed.seed_release_tag(repository.work, repository.branch)
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha


def test_concurrent_seed_is_an_idempotent_success(seed, repository, monkeypatch):
    original_git = seed.git

    def race(repo, *args):
        if args[0] == "push":
            _git(repository.remote, "update-ref", "refs/tags/v0.26.0rc0", repository.release_sha)
            raise subprocess.CalledProcessError(1, ["git", *args], stderr="tag already exists")
        return original_git(repo, *args)

    monkeypatch.setattr(seed, "git", race)
    assert seed.seed_release_tag(repository.work, repository.branch) == "Already versioned by v0.26.0rc0"
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha


def test_unpublished_local_tag_does_not_hide_missing_remote_baseline(seed, repository):
    _git(repository.work, "tag", "v0.26.0rc0", repository.main_sha)
    seed.seed_release_tag(repository.work, repository.branch)
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha
    assert _git(repository.work, "rev-parse", "refs/tags/v0.26.0rc0") == repository.main_sha


def test_cli_seeds_only_the_named_release_branch(repository):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--branch", repository.branch, "--creation-sha", repository.release_sha],
        cwd=repository.work,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"Created v0.26.0rc0 at {repository.release_sha}"
    assert _git(repository.remote, "rev-parse", "refs/tags/v0.26.0rc0") == repository.release_sha
