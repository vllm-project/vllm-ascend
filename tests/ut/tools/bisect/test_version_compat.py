import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.config import BisectOptions
from tools.bisect.version_compat import (
    TORCH_NPU_REQUIREMENTS_FILE,
    VLLM_PACKAGE,
    VLLM_TAG_FILE,
    PackageVersions,
    VersionAdapter,
    VersionPolicy,
    expected_versions,
    installed_package_version,
    versions_equal,
)

VLLM_SOURCE_DIR = Path("/vllm-workspace/vllm")
VLLM_ROLLBACK_TAG = "v0.25.1"


def test_expected_versions_read_nightly_source_files(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / VLLM_TAG_FILE).write_text("v0.25.1\n", encoding="utf-8")
    (tmp_path / TORCH_NPU_REQUIREMENTS_FILE).write_text(
        "torch-npu==2.10.0.post2\n",
        encoding="utf-8",
    )

    assert expected_versions(tmp_path) == PackageVersions(
        vllm="v0.25.1",
        torch_npu="2.10.0.post2",
    )


def test_expected_versions_falls_back_to_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "torch-npu==2.10.0.post3"]\n',
        encoding="utf-8",
    )

    assert expected_versions(tmp_path).torch_npu == "2.10.0.post3"


def test_expected_versions_at_commit_reads_all_files_without_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    contents = {
        VLLM_TAG_FILE: "v0.24.0\n",
        TORCH_NPU_REQUIREMENTS_FILE: "torch-npu==2.10.0.post1\n",
    }
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, path: contents.get(path))

    assert expected_versions(tmp_path, "a" * 40) == PackageVersions(
        vllm="v0.24.0",
        torch_npu="2.10.0.post1",
    )


def test_version_policy_only_checks_switchable_endpoint_changes():
    policy = VersionPolicy.between(
        PackageVersions(vllm="v0.24.0", torch_npu="2.10.0.post1"),
        PackageVersions(vllm="v0.25.1", torch_npu="2.10.0.post1"),
    )

    assert policy.checked_packages == ("vllm",)


def test_version_policy_disables_checks_when_endpoints_match():
    versions = PackageVersions(vllm="v0.25.1", torch_npu="2.10.0.post2")

    assert VersionPolicy.between(versions, versions).enabled is False


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        ("v0.9.1", "0.9.1+local", True),
        ("0.9.2", "0.9.1", False),
        (None, None, True),
        ("not-a-version", "vnot-a-version", True),
    ],
)
def test_versions_equal_ignores_local_suffix(left: str | None, right: str | None, equal: bool):
    assert versions_equal(left, right) is equal


def test_adapter_switches_mismatched_vllm_and_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    commands: list[list[str]] = []
    monkeypatch.setattr("tools.bisect.version_compat.installed_vllm_version", lambda: "0.24.0")
    monkeypatch.setattr("tools.bisect.version_compat.installed_torch_npu_version", lambda: "2.10.0.post1")
    monkeypatch.setattr(
        VersionAdapter,
        "_run",
        staticmethod(lambda command, log_file, label, env=None: commands.append(command)),
    )
    options = BisectOptions(repo_dir=tmp_path, vllm_dir=tmp_path / "missing-vllm")
    adapter = VersionAdapter(options)

    adapter.ensure_targets(
        {"vllm": "v0.25.1", "torch-npu": "2.10.0.post2"},
        ("vllm", "torch-npu"),
    )

    assert commands == [
        [
            "pip",
            "install",
            "vllm==0.25.1",
            "--no-input",
            "--disable-pip-version-check",
        ],
        [
            "pip",
            "install",
            "torch-npu==2.10.0.post2",
            "--force-reinstall",
            "--no-deps",
            "--no-input",
            "--disable-pip-version-check",
        ],
    ]


def test_adapter_sets_empty_target_device_for_editable_vllm_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    runs: list[tuple[list[str], dict[str, str] | None]] = []
    vllm_dir = tmp_path / "vllm"
    vllm_dir.mkdir()
    monkeypatch.setattr("tools.bisect.version_compat.installed_vllm_version", lambda: "0.24.0")
    monkeypatch.setattr(git_ops, "resolve_commit", lambda repo, ref: "a" * 40)
    monkeypatch.setattr(git_ops, "checkout", lambda repo, commit: None)
    monkeypatch.setattr(
        VersionAdapter,
        "_run",
        staticmethod(lambda command, log_file, label, env=None: runs.append((command, env))),
    )
    adapter = VersionAdapter(BisectOptions(repo_dir=tmp_path, vllm_dir=vllm_dir))

    adapter.ensure_targets({"vllm": "v0.25.1"}, ("vllm",))

    command, env = runs[0]
    assert command[:4] == ["pip", "install", "-e", str(vllm_dir)]
    assert env is not None
    assert env["VLLM_TARGET_DEVICE"] == "empty"


def _is_git_worktree_clean(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return not result.stdout.strip()


def _installed_vllm_version_from_test_interpreter() -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import version; print(version('vllm'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not VLLM_SOURCE_DIR.is_dir(), reason="requires the nightly vLLM source checkout")
def test_adapter_switches_real_vllm_release_and_restores_source(tmp_path: Path):
    if not _is_git_worktree_clean(VLLM_SOURCE_DIR):
        pytest.skip("vLLM source checkout has uncommitted changes")

    original_commit = git_ops.current_commit(VLLM_SOURCE_DIR)
    original_version = installed_package_version(VLLM_PACKAGE)
    original_vllm_version_env = os.environ.get("VLLM_VERSION")
    target_commit = git_ops.resolve_commit(VLLM_SOURCE_DIR, VLLM_ROLLBACK_TAG)
    if target_commit == original_commit:
        pytest.skip(f"vLLM source is already at {VLLM_ROLLBACK_TAG}")

    adapter = VersionAdapter(BisectOptions(repo_dir=tmp_path, vllm_dir=VLLM_SOURCE_DIR))
    log_file = tmp_path / "vllm_switch.log"
    try:
        adapter._switch_vllm(VLLM_ROLLBACK_TAG, log_file)
        assert git_ops.current_commit(VLLM_SOURCE_DIR) == target_commit
        assert versions_equal(
            _installed_vllm_version_from_test_interpreter(),
            VLLM_ROLLBACK_TAG,
        )
    finally:
        try:
            adapter._switch_vllm(original_commit, log_file)
        finally:
            if original_vllm_version_env is None:
                os.environ.pop("VLLM_VERSION", None)
            else:
                os.environ["VLLM_VERSION"] = original_vllm_version_env

    assert git_ops.current_commit(VLLM_SOURCE_DIR) == original_commit
    if original_version is not None:
        assert versions_equal(_installed_vllm_version_from_test_interpreter(), original_version)
