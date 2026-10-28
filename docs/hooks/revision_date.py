"""Resolve English source revision dates once per build for both languages."""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from babel.dates import format_date
from jinja2 import Environment, TemplateError, meta
from mkdocs.exceptions import PluginError

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGLISH_DOCS_DIR = REPO_ROOT / "docs" / "source"
REVISION_DATE_ATTRIBUTE = "_english_git_revision_timestamp"
REVISION_DATE_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _english_dependencies(src_uris: list[str]) -> dict[str, set[str]]:
    """Collect each page's English sources, parsing shared includes once per build."""
    docs_dir = ENGLISH_DOCS_DIR.resolve()
    environment = Environment()
    references: dict[str, list[str]] = {}

    def visit(template_name: str, visited: set[str]) -> None:
        source_path = (docs_dir / template_name).resolve()
        if not source_path.is_relative_to(docs_dir) or not source_path.is_file():
            raise PluginError(f"English document include not found: {template_name}")
        src_uri = source_path.relative_to(docs_dir).as_posix()
        if src_uri in visited:
            return
        visited.add(src_uri)

        if src_uri not in references:
            source = source_path.read_text(encoding="utf-8")
            included_names = []
            if "{%" in source and "include" in source:
                try:
                    for included_name in meta.find_referenced_templates(environment.parse(source)):
                        if included_name is None:
                            raise PluginError(f"Dynamic Jinja include in {source_path} cannot be tracked")
                        included_names.append(included_name)
                except TemplateError as error:
                    raise PluginError(f"Unable to parse Jinja includes in {source_path}") from error
            references[src_uri] = included_names

        for included_name in references[src_uri]:
            visit(included_name, visited)

    page_dependencies: dict[str, set[str]] = {}
    for src_uri in src_uris:
        visited: set[str] = set()
        visit(src_uri, visited)
        page_dependencies[src_uri] = visited
    return page_dependencies


def _check_git_history() -> None:
    """Reject shallow history, which can silently report an incorrect date."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PluginError("Revision dates require a Git checkout with complete history") from error
    if result.stdout.strip() == "true":
        raise PluginError("Revision dates require complete Git history; run 'git fetch --unshallow' first")


def _english_revision_timestamps(source_uris: set[str]) -> dict[str, int]:
    """Read path changes in one tree-only traversal of this branch's history.

    First-parent diffs include changes introduced by merges, including conflict
    resolutions, without counting discarded changes from merged branches.
    Renames and formatting changes count as updates; no blob contents are read.
    """
    if not source_uris:
        return {}
    docs_prefix = ENGLISH_DOCS_DIR.relative_to(REPO_ROOT).as_posix() + "/"
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--first-parent",
                "--diff-merges=first-parent",
                "--no-renames",
                "--no-show-signature",
                "--no-color",
                "--format=%at",
                "--name-only",
                "-z",
                "--",
                docs_prefix,
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PluginError("Unable to read Git history for English sources") from error

    timestamps: dict[str, int] = {}
    timestamp = None
    # Git separates names and commit headers with NULs. Only the first name
    # after a header has an extra newline; preserve whitespace inside paths.
    for field in result.stdout.split("\0"):
        if not field:
            continue
        field = field.removeprefix("\n")
        if not field.startswith(docs_prefix):
            timestamp = int(field)
            continue
        src_uri = field[len(docs_prefix) :]
        if src_uri in source_uris and src_uri not in timestamps:
            if timestamp is None:
                raise PluginError("Missing revision timestamp in Git history")
            # Select the latest change in history, not the largest author date:
            # authors can backdate commits and rebases preserve author dates.
            timestamps[src_uri] = timestamp
            if len(timestamps) == len(source_uris):
                break
    missing = source_uris - timestamps.keys()
    if missing:
        raise PluginError(f"English sources have no Git history: {', '.join(sorted(missing))}")
    return timestamps


def on_files(files, config, **kwargs) -> None:
    """Resolve published pages and shared English dependencies in one query."""
    if not config.extra.get("show_revision_date", True):
        return
    documents = [file for file in files.documentation_pages() if not getattr(file, "generated_by", None)]
    if not documents:
        return
    start = time.perf_counter()
    _check_git_history()
    page_dependencies = _english_dependencies([file.src_uri for file in documents])
    source_uris = {src_uri for dependencies in page_dependencies.values() for src_uri in dependencies}
    timestamps = _english_revision_timestamps(source_uris)

    # Keep dates on File: MkDocs loads and replaces page.meta later in the build.
    for file in documents:
        dependencies = page_dependencies[file.src_uri]
        setattr(file, REVISION_DATE_ATTRIBUTE, max(timestamps[src_uri] for src_uri in dependencies))
    logging.getLogger("mkdocs").info(
        "Resolved revision dates for %d English sources in %.2f seconds", len(source_uris), time.perf_counter() - start
    )


def on_page_context(context, page, config, **kwargs):
    """Render only the date used by Material's page footer."""
    if not config.extra.get("show_revision_date", True):
        return context
    timestamp = getattr(page.file, REVISION_DATE_ATTRIBUTE, None)
    if timestamp is not None:
        page.meta["git_revision_date_localized"] = format_date(
            datetime.fromtimestamp(timestamp, REVISION_DATE_TIMEZONE),
            format="long",
            locale=config.extra.get("docs_lang", "en"),
        )
    return context
