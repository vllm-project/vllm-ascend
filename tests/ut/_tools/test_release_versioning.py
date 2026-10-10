# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version
from setuptools_scm import get_version

POLICY = Path(__file__).resolve().parents[3] / "versioning.py"


@pytest.fixture(scope="module")
def package_version():
    # The same behavioral cases can run on main as a before/after control.
    resolver = None
    if POLICY.exists():
        spec = importlib.util.spec_from_file_location("release_versioning", POLICY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolver = module.get_package_version

    def version(repo, **kwargs):
        if resolver is not None:
            return resolver(root=str(repo), **kwargs)
        return get_version(root=str(repo), fallback_root=str(repo), **kwargs)

    return version


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()


def _commit(repo, text):
    (repo / "source.txt").write_text(text)
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "--no-gpg-sign", "-qm", text)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "source"
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Release Version Test")
    _git(repo, "config", "user.email", "release-version-test@example.invalid")
    _commit(repo, "old release")
    _git(repo, "tag", "-a", "v0.19.1rc1", "-m", "old release")
    creation = _commit(repo, "release branch point")
    _git(repo, "checkout", "-b", "releases/v0.26.0rc")
    return SimpleNamespace(path=repo, creation=creation)


@pytest.mark.parametrize("branch", ["releases/v0.26.0rc", "releases/v0.25.1rc", "releases/v0.30.0"])
def test_untagged_release_uses_branch_family_and_native_traceability(package_version, repository, branch):
    repo = repository.path
    _git(repo, "checkout", "-B", branch)
    native = Version(get_version(root=str(repo)))
    actual = Version(package_version(repo))
    assert native.base_version == "0.19.1"
    assert actual.base_version == branch.removeprefix("releases/v").removesuffix("rc")
    assert actual.pre == ("rc", 1)
    assert actual.dev == native.dev == 1
    assert actual.local == native.local


def test_old_annotated_tag_at_creation_cannot_override_new_family(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "-a", "v0.25.1", "-m", "old family at creation")
    assert package_version(repo) == "0.26.0rc0"
    _commit(repo, "first nightly change")
    actual = Version(package_version(repo))
    assert actual.base_version == "0.26.0"
    assert actual.pre == ("rc", 1) and actual.dev == 1
    assert actual.local.startswith("g")


@pytest.mark.parametrize("with_baselines", [False, True])
def test_shared_creation_commit_does_not_mix_release_families(package_version, repository, with_baselines):
    repo = repository.path
    if with_baselines:
        _git(repo, "tag", "v0.26.0rc0")
        _git(repo, "tag", "v0.27.1rc0")
    expected_suffix = "rc0" if with_baselines else "rc1.dev1"
    for branch, release in [("releases/v0.26.0rc", "0.26.0"), ("releases/v0.27.1rc", "0.27.1")]:
        _git(repo, "checkout", "-B", branch, repository.creation)
        actual = Version(package_version(repo))
        assert actual.public == release + expected_suffix


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("release_tag", ["v0.26.0rc1", "v0.26.0", "v0.26.0.post1"])
def test_real_release_at_baseline_commit_supersedes_rc0(package_version, repository, annotated, release_tag):
    repo = repository.path
    _git(repo, "tag", "-a", "v0.26.0rc0", "-m", "baseline")
    if annotated:
        _git(repo, "tag", "-a", release_tag, "-m", "real release")
    else:
        _git(repo, "tag", release_tag)
    assert package_version(repo) == release_tag.removeprefix("v")


def test_nested_annotated_release_supersedes_baseline(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "-a", "v0.26.0rc0", "-m", "baseline")
    _git(repo, "tag", "-a", "v0.26.0rc1", "v0.26.0rc0", "-m", "nested real release")
    assert package_version(repo) == "0.26.0rc1"


@pytest.mark.parametrize("tag", ["v0.26.0a1", "v0.26.0b1", "v0.26.0rc1", "v0.26.0", "v0.26.0.post1", "v0.26.0.dev0"])
def test_existing_family_tag_retains_native_development_scheme(package_version, repository, tag):
    repo = repository.path
    _git(repo, "tag", tag)
    assert package_version(repo) == tag.removeprefix("v")
    _commit(repo, "next nightly")
    assert package_version(repo) == get_version(root=str(repo))


def test_newer_unrelated_family_is_ignored(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "v0.26.0rc1")
    _commit(repo, "next release change")
    _git(repo, "tag", "-a", "v0.27.1rc1", "-m", "other family")
    actual = Version(package_version(repo))
    assert actual.base_version == "0.26.0" and actual.pre == ("rc", 2) and actual.dev == 1


def test_closest_commit_is_used_before_semantic_tag_tiebreak(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "v0.26.0.post9")
    _commit(repo, "closer tagged commit")
    _git(repo, "tag", "v0.26.0rc1")
    assert package_version(repo) == "0.26.0rc1"


def test_unreachable_and_invalid_tags_are_ignored(package_version, repository):
    repo = repository.path
    _git(repo, "checkout", "main")
    _commit(repo, "unrelated main change")
    _git(repo, "tag", "v0.26.0rc9")
    _git(repo, "checkout", "releases/v0.26.0rc")
    _git(repo, "tag", "v0.26.0invalid")
    _git(repo, "tag", "v0.26.0rc2+private")
    assert Version(package_version(repo)).public == "0.26.0rc1.dev1"


@pytest.mark.parametrize("dirty", [False, True])
def test_no_tags_still_retains_commit_count_and_hash(package_version, repository, dirty):
    repo = repository.path
    _git(repo, "tag", "-d", "v0.19.1rc1")
    if dirty:
        (repo / "source.txt").write_text("uncommitted source")
    native = Version(get_version(root=str(repo)))
    actual = Version(package_version(repo))
    assert actual.public == "0.26.0rc1.dev2"
    assert actual.local == native.local


def test_dirty_tagged_release_retains_native_dirty_marker(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "v0.26.0rc1")
    (repo / "source.txt").write_text("uncommitted source")
    assert package_version(repo) == get_version(root=str(repo))


@pytest.mark.parametrize(
    "branch", ["main", "feature/versioning", "releases/v1.2", "releases/v01.2.3", "releases/v1.2.3rc1"]
)
def test_other_branches_retain_native_version(package_version, repository, branch):
    repo = repository.path
    _git(repo, "checkout", "-B", branch)
    assert package_version(repo) == get_version(root=str(repo))


def test_detached_checkout_retains_native_version(package_version, repository):
    repo = repository.path
    _git(repo, "tag", "v0.26.0rc1")
    _git(repo, "checkout", "--detach", "v0.26.0rc1")
    assert package_version(repo) == get_version(root=str(repo)) == "0.26.0rc1"


def test_sdist_pkg_info_fallback_still_works(package_version, tmp_path):
    (tmp_path / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: vllm_ascend\nVersion: 0.26.0rc1\n")
    assert package_version(tmp_path) == "0.26.0rc1"


def test_pretend_version_is_honored(package_version, repository, monkeypatch):
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.26.0rc7")
    assert package_version(repository.path) == "0.26.0rc7"


def test_version_file_is_written_without_modifying_git_refs(package_version, repository):
    repo = repository.path
    before = _git(repo, "show-ref")
    result = package_version(repo, write_to="_version.py")
    values: dict[str, object] = {}
    exec((repo / "_version.py").read_text(), values)
    assert values["__version__"] == result
    assert _git(repo, "show-ref") == before
