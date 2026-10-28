import os
import subprocess
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

# Docs dependencies are optional in the runtime/lint CI environment.
PluginError = pytest.importorskip("mkdocs.exceptions").PluginError
mkdocs_files = pytest.importorskip("mkdocs.structure.files")
pytest.importorskip("babel")
revision_date = import_module("docs.hooks.revision_date")


def _document(name, docs_dir, *, excluded=False, generated=False):
    file = mkdocs_files.File(name, str(docs_dir), str(docs_dir / "site"), use_directory_urls=False)
    file.inclusion = mkdocs_files.InclusionLevel.EXCLUDED if excluded else mkdocs_files.InclusionLevel.INCLUDED
    if generated:
        file.generated_by = "test"
    return file


def _git(repo, *args, env=None, check=True):
    return subprocess.run(["git", *args], cwd=repo, env=env, check=check, capture_output=True, text=True).stdout.strip()


def _commit(repo, day):
    date = f"2024-01-{day:02d}T00:00:00+00:00"
    _git(repo, "add", ".")
    _git(
        repo,
        "commit",
        "-qm",
        "Update docs",
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_DATE": "2024-02-01T00:00:00+00:00",
        },
    )
    return int(datetime.fromisoformat(date).timestamp())


@pytest.fixture
def english_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(revision_date, "ENGLISH_DOCS_DIR", tmp_path)
    for name, content in {
        "main.md": '{% include "first.inc.md" %}',
        "first.inc.md": '{% include "second.inc.md" %}',
        "second.inc.md": "Included content",
    }.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def git_docs(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    for key, value in {
        "user.name": "Docs Test",
        "user.email": "docs@example.invalid",
        "commit.gpgsign": "false",
    }.items():
        _git(tmp_path, "config", key, value)
    docs_dir = tmp_path / "docs" / "source"
    docs_dir.mkdir(parents=True)
    (docs_dir / "original.md").write_text("Hello world\n", encoding="utf-8")
    created = _commit(tmp_path, 1)
    monkeypatch.setattr(revision_date, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(revision_date, "ENGLISH_DOCS_DIR", docs_dir)
    return tmp_path, docs_dir, created


def test_recursive_includes_handle_shared_sources_and_cycles(english_sources):
    (english_sources / "second.inc.md").write_text('{% include "first.inc.md" %}', encoding="utf-8")
    assert revision_date._english_dependencies(["main.md", "first.inc.md"]) == {
        "main.md": {"first.inc.md", "main.md", "second.inc.md"},
        "first.inc.md": {"first.inc.md", "second.inc.md"},
    }


@pytest.mark.parametrize(
    "language, expected_dates",
    [
        ("en", ("January 2, 1970", "January 3, 1970")),
        ("zh", ("1970年1月2日", "1970年1月3日")),
    ],
)
def test_page_dates_follow_english_includes_across_rebuilds(english_sources, monkeypatch, language, expected_dates):
    (english_sources / "other.md").write_text('{% include "first.inc.md" %}', encoding="utf-8")
    (english_sources / "new.inc.md").write_text("New content", encoding="utf-8")
    timestamps = {"main.md": 0, "other.md": 1, "first.inc.md": 3600, "second.inc.md": 86399, "new.inc.md": 172799}
    monkeypatch.setattr(revision_date, "_check_git_history", lambda: None)
    monkeypatch.setattr(
        revision_date, "_english_revision_timestamps", lambda sources: {s: timestamps[s] for s in sources}
    )
    docs_dir = english_sources / "zh" if language == "zh" else english_sources
    documents = [_document(name, docs_dir) for name in ("main.md", "other.md")]
    files = mkdocs_files.Files(
        documents
        + [
            _document("first.inc.md", docs_dir, excluded=True),
            _document("unused.inc.md", docs_dir, excluded=True),
            _document("generated.md", docs_dir, generated=True),
        ]
    )
    config = SimpleNamespace(extra={"docs_lang": language})

    for include, expected in zip(("second.inc.md", "new.inc.md"), expected_dates, strict=True):
        (english_sources / "first.inc.md").write_text(f'{{% include "{include}" %}}', encoding="utf-8")
        revision_date.on_files(files, config)
        for document in documents:
            page = SimpleNamespace(file=document, meta={})
            context: dict[str, object] = {}
            assert revision_date.on_page_context(context, page, config) is context
            assert page.meta["git_revision_date_localized"] == expected


@pytest.mark.parametrize(
    "content, message",
    [
        ('{% include "missing.inc.md" %}', "include not found"),
        ('{% include "../outside.md" %}', "include not found"),
        ("{% include template_name %}", "Dynamic Jinja include"),
        ("{% include %}", "Unable to parse Jinja includes"),
    ],
)
def test_invalid_includes_fail_explicitly(english_sources, content, message):
    (english_sources / "main.md").write_text(content, encoding="utf-8")
    with pytest.raises(PluginError, match=message):
        revision_date._english_dependencies(["main.md"])


def test_dates_can_be_disabled_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr(revision_date, "REPO_ROOT", tmp_path)
    config = SimpleNamespace(extra={"show_revision_date": False})
    document = _document("untracked.md", tmp_path)
    page = SimpleNamespace(file=document, meta={})
    revision_date.on_files(mkdocs_files.Files([document]), config)
    revision_date.on_page_context({}, page, config)
    assert "git_revision_date_localized" not in page.meta


@pytest.mark.parametrize("change", ["rename", "rename-and-edit", "whitespace", "recreate"])
def test_file_changes_update_author_date(git_docs, change):
    repo, docs_dir, _ = git_docs
    path = docs_dir / "original.md"
    if change.startswith("rename"):
        path = path.rename(docs_dir / "renamed.md")
        if change == "rename-and-edit":
            path.write_text("Hello world\nNew content\n", encoding="utf-8")
    elif change == "whitespace":
        path.write_text("Hello   world\n\n", encoding="utf-8")
    else:
        path.unlink()
        _commit(repo, 2)
        path.write_text("Recreated page\n", encoding="utf-8")
    updated = _commit(repo, 3)

    assert revision_date._english_revision_timestamps({path.name}) == {path.name: updated}


def test_batch_query_preserves_paths_and_uses_history_order(git_docs):
    repo, docs_dir, created = git_docs
    name = "nested/中文 space\tand\nnewline.md"
    path = docs_dir / name
    path.parent.mkdir()
    path.write_text("First version\n", encoding="utf-8")
    _commit(repo, 4)
    path.write_text("Latest version with an older author date\n", encoding="utf-8")
    updated = _commit(repo, 2)

    with patch.object(revision_date.subprocess, "run", wraps=subprocess.run) as run:
        dates = revision_date._english_revision_timestamps({"original.md", name})
    assert dates == {"original.md": created, name: updated}
    assert run.call_count == 1


@pytest.mark.parametrize("merge_kind", ["incoming", "conflict", "discarded"])
def test_merge_dates_follow_changes_to_current_branch(git_docs, merge_kind):
    repo, docs_dir, created = git_docs
    _git(repo, "checkout", "-qb", "topic")
    (docs_dir / "original.md").write_text("Topic content\n", encoding="utf-8")
    _commit(repo, 2)
    _git(repo, "checkout", "main")
    (docs_dir / "other.md").write_text("Main branch content\n", encoding="utf-8")
    if merge_kind == "conflict":
        (docs_dir / "original.md").write_text("Conflicting content\n", encoding="utf-8")
    main_date = _commit(repo, 3)
    options = ["-s", "ours"] if merge_kind == "discarded" else []
    _git(repo, "merge", "--no-ff", "--no-commit", *options, "topic", check=merge_kind != "conflict")
    if merge_kind == "conflict":
        (docs_dir / "original.md").write_text("Resolved content\n", encoding="utf-8")
    merge_date = _commit(repo, 4)

    assert revision_date._english_revision_timestamps({"original.md", "other.md"}) == {
        "original.md": created if merge_kind == "discarded" else merge_date,
        "other.md": main_date,
    }


def test_untracked_sources_do_not_use_build_time(git_docs):
    _, docs_dir, _ = git_docs
    (docs_dir / "new.md").write_text("Uncommitted content", encoding="utf-8")
    with pytest.raises(PluginError, match="have no Git history: new.md"):
        revision_date._english_revision_timestamps({"new.md", "original.md"})


def test_git_is_required_only_for_nonempty_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(revision_date, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(revision_date, "ENGLISH_DOCS_DIR", tmp_path / "docs" / "source")
    assert revision_date._english_revision_timestamps(set()) == {}
    with pytest.raises(PluginError, match="require a Git checkout"):
        revision_date._check_git_history()
    with pytest.raises(PluginError, match="Unable to read Git history"):
        revision_date._english_revision_timestamps({"original.md"})


@pytest.mark.parametrize("version_type", ["branch", "tag"])
def test_rtd_completes_history_without_downloading_old_contents(git_docs, tmp_path, monkeypatch, version_type):
    repo, docs_dir, _ = git_docs
    _git(repo, "config", "uploadpack.allowFilter", "true")
    historical_blob = _git(repo, "rev-parse", "HEAD:docs/source/original.md")
    _git(repo, "checkout", "-qb", "docs-preview")
    (docs_dir / "original.md").write_text("Preview content\n", encoding="utf-8")
    updated = _commit(repo, 2)
    ref = "docs-preview"
    if version_type == "tag":
        ref = "docs-release"
        _git(repo, "tag", ref)
    clone = tmp_path / "checkout"
    _git(repo, "clone", "--quiet", "--depth=1", "--branch", ref, repo.as_uri(), str(clone))
    expected_commit = _git(clone, "rev-parse", "HEAD")
    monkeypatch.setattr(revision_date, "REPO_ROOT", clone)
    monkeypatch.setattr(revision_date, "ENGLISH_DOCS_DIR", clone / "docs" / "source")
    with pytest.raises(PluginError, match="git fetch --unshallow"):
        revision_date._check_git_history()

    config_path = Path(__file__).resolve().parents[3] / ".readthedocs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for command in config["build"]["jobs"]["post_checkout"]:
        subprocess.run(["bash", "-e", "-c", command], cwd=clone, check=True, capture_output=True)
    revision_date._check_git_history()
    assert _git(clone, "rev-parse", "HEAD") == expected_commit
    assert _git(clone, "rev-list", "--count", "HEAD") == "2"
    assert f"?{historical_blob}" in _git(clone, "rev-list", "--objects", "--all", "--missing=print")
    # Dates must still resolve offline when historical file contents are absent.
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "unavailable-origin"))
    assert revision_date._english_revision_timestamps({"original.md"}) == {"original.md": updated}
    assert f"?{historical_blob}" in _git(clone, "rev-list", "--objects", "--all", "--missing=print")
