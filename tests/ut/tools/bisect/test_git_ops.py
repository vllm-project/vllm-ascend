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
