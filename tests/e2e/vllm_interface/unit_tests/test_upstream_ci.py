from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.vllm_interface.vllm_interface_contracts.upstream_ci import (
    build_analysis_command,
    resolve_vllm_range,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    (root / f"{message}.txt").write_text(message, encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Interface Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def test_resolve_vllm_range_uses_feature_merge_base(tmp_path: Path) -> None:
    root = tmp_path / "vllm"
    root.mkdir()
    _git(root, "init", "-b", "main")
    base_sha = _commit(root, "base")
    _git(root, "switch", "-c", "feature")
    head_sha = _commit(root, "feature")

    old_sha, new_sha = resolve_vllm_range(root, upstream_url=str(root), upstream_branch="main")

    assert old_sha == base_sha
    assert new_sha == head_sha


def test_resolve_vllm_range_requires_git_metadata(tmp_path: Path) -> None:
    root = tmp_path / "vllm"
    root.mkdir()

    with pytest.raises(ValueError, match="no Git metadata"):
        resolve_vllm_range(root)


def test_build_analysis_command_uses_fixed_upstream_pr_scope(tmp_path: Path) -> None:
    command = build_analysis_command(
        vllm_root=tmp_path / "vllm",
        ascend_root=tmp_path / "vllm-ascend",
        old_sha="old",
        new_sha="new",
        ascend_sha="ascend",
        index_workers=4,
    )

    assert command[1:4] == [
        "-m",
        "tests.e2e.vllm_interface.vllm_interface_contracts",
        "analyze-range",
    ]
    assert "--scenario" not in command
    assert command[command.index("--fail-on") + 1] == "introduced"
    assert "--output-dir" not in command
    assert "--external-root" not in command
    assert "--expect-external-sha" not in command
    assert command[command.index("--analysis-workers") + 1] == "3"
    assert command[command.index("--index-workers") + 1] == "4"
