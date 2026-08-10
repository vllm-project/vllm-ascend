from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.config import Candidate
from tools.bisect.version_history import (
    ExternalVersionManager,
    VersionHistory,
    VersionProfile,
    _extract_torch_npu,
    extract_profile_at,
    infer_branch_from_good_table,
)


def _candidate(commit: str) -> Candidate:
    return Candidate(commit=commit, pr_number=None, subject=commit)


def test_extract_profile_reads_versions_from_designated_files(monkeypatch, tmp_path: Path):
    files = {
        ("c1", ".github/vllm-release-tag.commit"): "v0.26.0\n",
        ("c1", "requirements.txt"): "torch-npu==2.10.0.post2\n",
        ("c1", "pyproject.toml"): "",
        ("c1", "Dockerfile.a3"): 'ARG CANN_VERSION="9.0.1"\n',
    }
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, rel_path: files.get((commit, rel_path)))

    profile = extract_profile_at(tmp_path, "c1", "main", "a3")

    assert profile == VersionProfile(
        branch="main",
        target="a3",
        commit="c1",
        vllm_release_tag="v0.26.0",
        torch_npu_version="2.10.0.post2",
        cann_version="9.0.1",
    )


def test_extract_torch_npu_falls_back_to_pyproject():
    assert _extract_torch_npu("", 'dependencies = ["torch-npu==2.10.0.post2"]') == "2.10.0.post2"


def test_extract_torch_npu_parses_toml_requirements():
    pyproject = """
[build-system]
requires = ["setuptools", "torch-npu==2.10.0.post2"]
"""

    assert _extract_torch_npu("", pyproject) == "2.10.0.post2"


def test_infer_branch_from_good_table_cache_path():
    path = "/root/.cache/vllm-ascend/releases-v0.26/nightly/good_table.csv"

    assert infer_branch_from_good_table(path) == "releases-v0.26"


def test_record_range_writes_only_change_points(monkeypatch, tmp_path: Path):
    table = tmp_path / "version_history.csv"
    profiles = {
        "good": VersionProfile("main", "a2", "good", "v0.25.0", "2.9.0", "9.0.0"),
        "mid": VersionProfile("main", "a2", "mid", "v0.26.0", "2.10.0.post2", "9.0.1"),
        "bad": VersionProfile("main", "a2", "bad", "v0.26.0", "2.10.0.post2", "9.0.1"),
    }
    monkeypatch.setattr(
        "tools.bisect.version_history.extract_profile_at",
        lambda repo, commit, branch, target: profiles[commit],
    )

    history = VersionHistory(str(table), tmp_path, "main", "a2")
    active = history.record_range("good", [_candidate("mid"), _candidate("bad")])

    rows = table.read_text(encoding="utf-8").splitlines()
    assert active is True
    assert len(rows) == 3
    assert rows[1].startswith("main,a2,good,v0.25.0")
    assert rows[2].startswith("main,a2,mid,v0.26.0")


def test_record_range_disables_sync_when_endpoints_match(monkeypatch, tmp_path: Path):
    table = tmp_path / "version_history.csv"
    profile = VersionProfile("main", "a2", "ignored", "v0.26.0", "2.10.0.post2", "9.0.1")
    cann_versions = {"good": "9.0.0", "mid": "9.0.1", "bad": "9.0.1"}
    monkeypatch.setattr(
        "tools.bisect.version_history.extract_profile_at",
        lambda repo, commit, branch, target: VersionProfile(
            profile.branch,
            profile.target,
            commit,
            profile.vllm_release_tag,
            profile.torch_npu_version,
            cann_versions[commit],
        ),
    )

    history = VersionHistory(str(table), tmp_path, "main", "a2")
    active = history.record_range("good", [_candidate("mid"), _candidate("bad")])

    assert active is False
    rows = table.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert rows[2].startswith("main,a2,mid,v0.26.0,2.10.0.post2,9.0.1")


def test_sync_profile_ignores_cann(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    manager = ExternalVersionManager(
        VersionHistory(str(tmp_path / "version_history.csv"), tmp_path, "main", "a2"),
        sync_enabled=True,
    )
    monkeypatch.setattr(manager, "_sync_vllm", lambda profile, log_file: calls.append("vllm"))
    monkeypatch.setattr(manager, "_sync_torch_npu", lambda profile, log_file: calls.append("torch-npu"))

    manager.sync_profile(VersionProfile("main", "a2", "commit", "v0.26.0", "2.10.0.post2", "missing-cann"))

    assert calls == ["vllm", "torch-npu"]


def test_lookup_uses_latest_reachable_change_point(monkeypatch, tmp_path: Path):
    table = tmp_path / "version_history.csv"
    table.write_text(
        "branch,target,commit,vllm_release_tag,torch_npu_version,cann_version\n"
        "main,a2,good,v0.25.0,2.9.0,9.0.0\n"
        "main,a2,mid,v0.26.0,2.10.0.post2,9.0.1\n",
        encoding="utf-8",
    )

    def is_ancestor(repo, ancestor, descendant):
        order = {"good": 0, "mid": 1, "target": 2}
        return order[ancestor] <= order[descendant]

    monkeypatch.setattr(git_ops, "is_ancestor", is_ancestor)

    profile = VersionHistory(str(table), tmp_path, "main", "a2").lookup("target")

    assert profile is not None
    assert profile.commit == "target"
    assert profile.vllm_release_tag == "v0.26.0"
