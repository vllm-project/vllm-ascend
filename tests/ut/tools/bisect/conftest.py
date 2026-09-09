import importlib.util
import subprocess
import sys
import types
from unittest.mock import MagicMock

import pytest

if importlib.util.find_spec("psutil") is None:
    psutil = types.ModuleType("psutil")
    psutil.__spec__ = importlib.util.spec_from_loader("psutil", loader=None)
    psutil.Error = RuntimeError  # type: ignore[attr-defined]
    psutil.Process = MagicMock()  # type: ignore[attr-defined]
    psutil.process_iter = MagicMock(return_value=[])  # type: ignore[attr-defined]
    sys.modules["psutil"] = psutil


def run_git(cwd, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


@pytest.fixture
def repo_with_commits(tmp_path):
    """Factory: a real git repo with ``n`` commits; returns (path, [shas])."""

    def _make(n: int, name: str = "src"):
        repo = tmp_path / name
        repo.mkdir(parents=True)
        run_git(repo, "init", "-b", "main")
        run_git(repo, "config", "user.email", "bisect-ut@example.com")
        run_git(repo, "config", "user.name", "bisect-ut")
        shas = []
        for i in range(1, n + 1):
            (repo / "a.py").write_text(f"value = {i}\n", encoding="utf-8")
            run_git(repo, "add", ".")
            run_git(repo, "commit", "-m", f"c{i}")
            shas.append(run_git(repo, "rev-parse", "HEAD"))
        return repo, shas

    return _make


@pytest.fixture
def shallow_clone(repo_with_commits):
    """Factory: depth-1 clone of ``src`` at ``dst`` -- the CI worker's repo state.

    The nightly pods clone with ``--depth 1``, so the clone tip is the only
    local commit; every other commit must be recovered from origin.
    """

    def _make(src, dst):
        run_git(
            dst.parent,
            "clone",
            "--quiet",
            "--depth",
            "1",
            "file:///" + src.resolve().as_posix(),
            str(dst),
        )
        return dst

    return _make
