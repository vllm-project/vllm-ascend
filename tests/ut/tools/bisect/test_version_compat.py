from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.config import BisectOptions
from tools.bisect.version_compat import (
    CANN_VERSION_FILE,
    TORCH_NPU_REQUIREMENTS_FILE,
    VLLM_TAG_FILE,
    PackageVersions,
    VersionAdapter,
    VersionPolicy,
    expected_cann_version,
    expected_torch_npu_version,
    expected_versions_at,
    expected_vllm_version,
)


def test_expected_versions_read_nightly_source_files(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / VLLM_TAG_FILE).write_text("v0.25.1\n", encoding="utf-8")
    (tmp_path / TORCH_NPU_REQUIREMENTS_FILE).write_text(
        "torch-npu==2.10.0.post2\n",
        encoding="utf-8",
    )
    (tmp_path / "csrc").mkdir()
    (tmp_path / CANN_VERSION_FILE).write_text("Version=9.0.1\n", encoding="utf-8")

    assert expected_vllm_version(tmp_path) == "v0.25.1"
    assert expected_torch_npu_version(tmp_path) == "2.10.0.post2"
    assert expected_cann_version(tmp_path) == "9.0.1"


def test_expected_torch_npu_version_falls_back_to_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "torch-npu==2.10.0.post3"]\n',
        encoding="utf-8",
    )

    assert expected_torch_npu_version(tmp_path) == "2.10.0.post3"


def test_expected_versions_at_reads_all_files_without_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    contents = {
        VLLM_TAG_FILE: "v0.24.0\n",
        TORCH_NPU_REQUIREMENTS_FILE: "torch-npu==2.10.0.post1\n",
        CANN_VERSION_FILE: "Version=9.0.0\n",
    }
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, path: contents[path])

    assert expected_versions_at(tmp_path, "a" * 40) == PackageVersions(
        vllm="v0.24.0",
        torch_npu="2.10.0.post1",
        cann="9.0.0",
    )


def test_version_policy_only_checks_switchable_endpoint_changes():
    policy = VersionPolicy.between(
        PackageVersions(vllm="v0.24.0", torch_npu="2.10.0.post1", cann="9.0.0"),
        PackageVersions(vllm="v0.25.1", torch_npu="2.10.0.post1", cann="9.1.0"),
    )

    assert policy.checked_packages == ("vllm",)
    assert policy.checks("cann") is False


def test_version_policy_disables_checks_when_endpoints_match():
    versions = PackageVersions(vllm="v0.25.1", torch_npu="2.10.0.post2", cann="9.0.1")

    assert VersionPolicy.between(versions, versions).enabled is False


def test_adapter_switches_mismatched_vllm_and_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    commands: list[list[str]] = []
    monkeypatch.setattr("tools.bisect.version_compat.installed_vllm_version", lambda: "0.24.0")
    monkeypatch.setattr("tools.bisect.version_compat.installed_torch_npu_version", lambda: "2.10.0.post1")
    monkeypatch.setattr(
        VersionAdapter,
        "_run",
        staticmethod(lambda command, log_file, label: commands.append(command)),
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
