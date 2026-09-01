from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.config import BisectOptions
from tools.bisect.version_compat import (
    TORCH_NPU_REQUIREMENTS_FILE,
    VLLM_TAG_FILE,
    PackageVersions,
    VersionAdapter,
    VersionPolicy,
    expected_versions,
    versions_equal,
)


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
