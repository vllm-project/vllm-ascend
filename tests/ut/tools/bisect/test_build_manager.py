from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.build_manager import BuildManager
from tools.bisect.config import BisectOptions


def _build_manager(tmp_path: Path, *, native_check: str = "per-commit") -> BuildManager:
    return BuildManager(
        BisectOptions(
            repo_dir=tmp_path,
            assume_built_head=False,
            native_check=native_check,
        )
    )


def test_decide_rebuilds_without_established_baseline(tmp_path: Path):
    manager = _build_manager(tmp_path)

    decision = manager.decide("a" * 40)

    assert decision.rebuild is True
    assert decision.reinstall_reqs is False
    assert "no established build baseline" in decision.reason


def test_decide_skips_when_target_is_already_built(tmp_path: Path):
    target = "a" * 40
    manager = _build_manager(tmp_path)
    manager.last_built_commit = target

    decision = manager.decide(target)

    assert decision.rebuild is False
    assert decision.reinstall_reqs is False
    assert decision.reason == "already built this exact commit"


@pytest.mark.parametrize(
    ("changed_files", "expected_rebuild", "expected_reinstall"),
    [
        (["vllm_ascend/worker/model_runner.py"], False, False),
        (["requirements-dev.txt"], False, True),
        (["csrc/attention/op.cpp"], True, False),
        (["csrc/attention/op.cpp", "requirements-dev.txt"], True, True),
        (["pyproject.toml"], True, False),
    ],
)
def test_decide_per_commit_file_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_files: list[str],
    expected_rebuild: bool,
    expected_reinstall: bool,
):
    manager = _build_manager(tmp_path)
    manager.last_built_commit = "a" * 40
    monkeypatch.setattr(git_ops, "commit_changed_files", lambda repo, commit: changed_files)

    decision = manager.decide("b" * 40)

    assert decision.rebuild is expected_rebuild
    assert decision.reinstall_reqs is expected_reinstall


def test_decide_since_build_uses_delta_from_last_built_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    base = "a" * 40
    target = "b" * 40
    calls = []
    manager = _build_manager(tmp_path, native_check="since-build")
    manager.last_built_commit = base

    def fake_changed_files(repo: Path, old: str, new: str) -> list[str]:
        calls.append((repo, old, new))
        return ["setup.py"]

    monkeypatch.setattr(git_ops, "changed_files", fake_changed_files)

    decision = manager.decide(target)

    assert calls == [(tmp_path, base, target)]
    assert decision.rebuild is True
    assert "since last build" in decision.reason


def test_prepare_checkout_only_does_not_run_install_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    base = "a" * 40
    target = "b" * 40
    checkouts = []
    manager = _build_manager(tmp_path)
    manager.last_built_commit = base
    monkeypatch.setattr(git_ops, "commit_changed_files", lambda repo, commit: ["tests/test_only.py"])
    monkeypatch.setattr(git_ops, "checkout", lambda repo, commit: checkouts.append((repo, commit)))
    monkeypatch.setattr(manager, "_run", pytest.fail)

    decision = manager.prepare(target)

    assert checkouts == [(tmp_path, target)]
    assert decision.rebuild is False
    assert decision.reinstall_reqs is False
    assert manager.last_built_commit == base
