import subprocess
from pathlib import Path

import pytest

from tools.bisect import git_ops


def test_candidate_list_parses_first_parent_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sha_with_pr = "a" * 40
    sha_without_pr = "b" * 40
    raw_log = f"{sha_with_pr}\x1fFix nightly regression (#12345)\nignored\n{sha_without_pr}\x1fPlain commit"
    monkeypatch.setattr(git_ops, "_is_shallow", lambda repo: False)
    monkeypatch.setattr(git_ops, "is_ancestor", lambda repo, good, bad: True)
    monkeypatch.setattr(git_ops, "_git", lambda repo, *args, check=True: raw_log)

    candidates = git_ops.candidate_list(tmp_path, "good", "bad")

    assert [candidate.commit for candidate in candidates] == [sha_with_pr, sha_without_pr]
    assert candidates[0].pr_number == "12345"
    assert candidates[0].label == "PR-12345"
    assert candidates[1].pr_number is None
    assert candidates[1].label == f"commit-{sha_without_pr[:12]}"


def test_candidate_list_rejects_shallow_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(git_ops, "_is_shallow", lambda repo: True)

    with pytest.raises(git_ops.GitError, match="shallow clone"):
        git_ops.candidate_list(tmp_path, "good", "bad")


def test_candidate_list_rejects_invalid_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(git_ops, "_is_shallow", lambda repo: False)
    monkeypatch.setattr(git_ops, "is_ancestor", lambda repo, good, bad: False)

    with pytest.raises(git_ops.GitError, match="is not an ancestor"):
        git_ops.candidate_list(tmp_path, "good", "bad")


def test_matches_any_returns_files_matching_configured_globs():
    files = [
        "vllm_ascend/worker/model_runner.py",
        "csrc/kernels/attention.cpp",
        "requirements-dev.txt",
        "docs/README.md",
    ]

    assert git_ops.matches_any(files, ("csrc/**", "requirements*.txt")) == [
        "csrc/kernels/attention.cpp",
        "requirements-dev.txt",
    ]


@pytest.mark.parametrize("remote_state", ["deleted", "rewritten", "unchanged"])
def test_resolve_commit_refreshes_cached_pr_head(tmp_path: Path, remote_state: str):
    remote = tmp_path / "remote"
    repo = tmp_path / "repo"

    def git(path, *args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "commit", "--allow-empty", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "commit", "--allow-empty", "-m", "old PR head")
    old_head = git(repo, "rev-parse", "HEAD")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "HEAD:refs/pull/123/head")
    assert git_ops.resolve_commit(repo, "#123") == old_head

    if remote_state == "deleted":
        git(remote, "update-ref", "-d", "refs/pull/123/head")
        with pytest.raises(git_ops.GitError, match="Could not resolve ref"):
            git_ops.resolve_commit(repo, "#123")
    elif remote_state == "rewritten":
        git(repo, "checkout", "--detach", base)
        git(repo, "commit", "--allow-empty", "-m", "rewritten PR head")
        new_head = git(repo, "rev-parse", "HEAD")
        git(repo, "push", "--force", "origin", "HEAD:refs/pull/123/head")
        assert git_ops.resolve_commit(repo, "#123") == new_head
    else:
        assert git_ops.resolve_commit(repo, "#123") == old_head
