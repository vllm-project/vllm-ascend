#!/usr/bin/env python3
"""
Collect release note fragments into a draft and audit for missing ones.

Fragments live in release_notes/fragments/<PR number>.yaml. See release_notes/README.md
for when a fragment is needed and the field rules. The default --fragments directory is
resolved from the current directory first, then from the repository that contains this
script, so it works from the repo root and from the skill directory.

Usage:
    python collect_release_notes.py \
        --version v0.31.0 \
        --raw-commits output/v0.31.0/0-current-raw-commits.md \
        --output output/v0.31.0/1-fragments-draft.md \
        [--archive]
"""

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_REPO = "vllm-project/vllm-ascend"

# Maps each release notes section to the fragment categories that feed it, in the
# order the sections appear in docs/source/user_guide/release_notes.md.
SECTION_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Features", ["feature"]),
    ("Hardware and Operator Support", ["hardware"]),
    ("Performance", ["performance"]),
    ("Dependencies", ["dependency"]),
    ("Deprecation and Configuration Changes", ["breaking", "config"]),
    ("Documentation", ["documentation"]),
    ("Stability and Bug Fixes", ["bugfix"]),
    ("Other Changes", ["other"]),
]

VALID_CATEGORIES = [category for _, categories in SECTION_CATEGORIES for category in categories]

PR_LINK_IN_TEXT_RE = re.compile(r"\(#\d+\)|pull/\d+")
PR_REF_RE = re.compile(r"\(#(\d+)\)|pull/(\d+)")
FRAGMENT_NAME_RE = re.compile(r"^(\d+)(?:[-._].+)?$")


@dataclass
class Entry:
    pr: int
    category: str
    text: str
    highlight: bool = False
    model: str = ""
    hardware: list[str] = field(default_factory=list)


@dataclass
class MergedEntry:
    category: str
    text: str
    prs: list[int] = field(default_factory=list)
    highlight: bool = False
    model: str = ""
    hardware: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Collapse whitespace so equivalent texts merge correctly."""
    return " ".join(str(text).split())


def resolve_default_fragments() -> str:
    """Resolve release_notes/fragments from the current directory or the script's repository."""
    default = Path("release_notes/fragments")
    if default.is_dir():
        return str(default)
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except IndexError:
        return str(default)
    candidate = repo_root / default
    return str(candidate if candidate.is_dir() else default)


def parse_fragment(path: Path, errors: list[str]) -> list[Entry]:
    """Parse and validate a single fragment file."""
    match = FRAGMENT_NAME_RE.match(path.stem)
    if not match:
        errors.append(f"{path}: file name must be <PR number>.yaml or <PR number>-<slug>.yaml")
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: cannot read: {exc}")
        return []

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        errors.append(f"{path}: invalid YAML: {exc}")
        return []

    if not isinstance(data, dict):
        errors.append(f"{path}: top level must be a mapping with 'pr' and 'entries'")
        return []

    pr = data.get("pr")
    if isinstance(pr, bool):
        pr = None
    if isinstance(pr, str) and pr.isdigit():
        pr = int(pr)
    if not isinstance(pr, int) or pr <= 0:
        errors.append(f"{path}: 'pr' is required and must be the PR number")
        return []
    if int(match.group(1)) != pr:
        errors.append(f"{path}: file name number {match.group(1)} does not match 'pr: {pr}'")

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        errors.append(f"{path}: 'entries' is required and must be a non-empty list")
        return []

    entries = []
    for i, raw in enumerate(raw_entries):
        where = f"{path} entries[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{where}: must be a mapping")
            continue

        category = raw.get("category")
        if category not in VALID_CATEGORIES:
            errors.append(f"{where}: 'category' must be one of {', '.join(VALID_CATEGORIES)} (got {category!r})")
            continue

        raw_text = raw.get("text")
        if not isinstance(raw_text, str):
            errors.append(f"{where}: 'text' is required and must be a string")
            continue
        text = normalize_text(raw_text)
        if not text:
            errors.append(f"{where}: 'text' is required")
            continue
        if PR_LINK_IN_TEXT_RE.search(text):
            errors.append(f"{where}: do not append the PR link in 'text', the tooling adds it")

        highlight = raw.get("highlight", False)
        if not isinstance(highlight, bool):
            errors.append(f"{where}: 'highlight' must be true or false")
            continue

        model = raw.get("model", "")
        if model is None:
            model = ""
        if not isinstance(model, str):
            errors.append(f"{where}: 'model' must be a string")
            continue
        model = normalize_text(model)

        hardware = raw.get("hardware", [])
        if isinstance(hardware, str):
            hardware = [hardware]
        if not isinstance(hardware, list) or not all(isinstance(h, str) for h in hardware):
            errors.append(f"{where}: 'hardware' must be a string or a list of strings")
            continue
        hardware = [normalize_text(h) for h in hardware]
        if any(not h for h in hardware):
            errors.append(f"{where}: 'hardware' entries must not be empty")
            continue

        entries.append(Entry(pr=pr, category=category, text=text, highlight=highlight, model=model, hardware=hardware))

    return entries


def fragment_sort_key(path: Path) -> int:
    """Sort fragment files by their leading PR number."""
    match = FRAGMENT_NAME_RE.match(path.stem)
    return int(match.group(1)) if match else 0


def load_fragments(fragments_dir: Path) -> tuple[list[Entry], list[Path], list[str]]:
    """Load all fragments in the directory, returning entries, files, and errors."""
    if not fragments_dir.is_dir():
        return [], [], [f"{fragments_dir}: fragments directory not found"]

    files = sorted(
        (
            p
            for p in fragments_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml") and p.stem != "TEMPLATE"
        ),
        key=fragment_sort_key,
    )
    entries: list[Entry] = []
    errors: list[str] = []
    for path in files:
        entries.extend(parse_fragment(path, errors))
    return entries, files, errors


def merge_entries(entries: list[Entry]) -> list[MergedEntry]:
    """Merge entries with identical text (for example cherry-picks) into one bullet."""
    merged: dict[tuple[str, str], MergedEntry] = {}
    for entry in entries:
        key = (entry.category, entry.text.lower())
        if key not in merged:
            merged[key] = MergedEntry(category=entry.category, text=entry.text)
        target = merged[key]
        if entry.pr not in target.prs:
            target.prs.append(entry.pr)
        target.highlight = target.highlight or entry.highlight
        if not target.model and entry.model:
            target.model = entry.model
        for hardware in entry.hardware:
            if hardware not in target.hardware:
                target.hardware.append(hardware)
    return list(merged.values())


def format_bullet(entry: MergedEntry, repo: str) -> str:
    links = " ".join(f"[#{pr}](https://github.com/{repo}/pull/{pr})" for pr in entry.prs)
    return f"- {entry.text} {links}"


def escape_table_cell(value: str) -> str:
    """Escape '|' so metadata values do not break the markdown table."""
    return value.replace("|", "\\|")


def audit_raw_commits(raw_commits: Path, fragment_prs: set[int]) -> tuple[list[str], list[str], list[int]]:
    """Audit the raw commit list against the fragment PRs.

    Returns lines with a PR reference but no fragment, lines without a PR reference, and
    fragment PRs that never appear in the commit list (for example reverted PRs).
    """
    missing: list[str] = []
    no_pr: list[str] = []
    seen_prs: set[int] = set()
    for raw_line in raw_commits.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prs = {int(a or b) for a, b in PR_REF_RE.findall(line)}
        seen_prs |= prs
        if not prs:
            no_pr.append(line)
        elif prs - fragment_prs:
            missing.append(line)
    unmatched_prs = sorted(fragment_prs - seen_prs)
    return missing, no_pr, unmatched_prs


def render_draft(
    version: str,
    merged: list[MergedEntry],
    repo: str,
    fragment_count: int,
    raw_commits: Path | None,
    audit_result: tuple[list[str], list[str], list[int]] | None = None,
) -> str:
    lines = [f"# Release Note Fragments Draft: {version}", ""]

    highlights = [entry for entry in merged if entry.highlight]
    lines.append(f"> {fragment_count} fragment(s), {len(merged)} entries, {len(highlights)} highlight candidate(s).")
    lines.append(">")
    lines.append(
        "> This is raw material: curate the wording and order, write the Highlights, then copy"
        " the result into `docs/source/user_guide/release_notes.md`."
    )
    if audit_result is not None and audit_result[2]:
        lines.append(">")
        lines.append(
            f"> Warning: {len(audit_result[2])} fragment PR(s) are not in the commit list, likely leftovers from an"
            " earlier release. Review the Audit section before using this draft."
        )
    lines.append("")

    lines.append("## Highlights (candidates)")
    lines.append("")
    if highlights:
        lines.extend(format_bullet(entry, repo) for entry in highlights)
    else:
        lines.append("_No entry is marked with `highlight: true`._")
    lines.append("")

    for section, categories in SECTION_CATEGORIES:
        lines.append(f"## {section}")
        lines.append("")
        section_entries = [entry for entry in merged if entry.category in categories]
        if section_entries:
            lines.extend(format_bullet(entry, repo) for entry in section_entries)
        else:
            lines.append("_(empty)_")
        lines.append("")

    matrix_entries = [entry for entry in merged if entry.model or entry.hardware]
    lines.append("## Support Matrix Hints")
    lines.append("")
    lines.append("| Model | Hardware | PRs |")
    lines.append("|-------|----------|-----|")
    if matrix_entries:
        for entry in matrix_entries:
            prs = ", ".join(f"#{pr}" for pr in entry.prs)
            model = escape_table_cell(entry.model) or "-"
            hardware = escape_table_cell(", ".join(entry.hardware)) or "-"
            lines.append(f"| {model} | {hardware} | {prs} |")
    else:
        lines.append("| - | - | - |")
    lines.append("")

    if raw_commits is not None and audit_result is not None:
        missing, no_pr, unmatched_prs = audit_result
        lines.append("## Audit: Code Changes Without a Fragment")
        lines.append("")
        lines.append(f"Raw commits source: `{raw_commits}`")
        lines.append("")
        fragment_prs = {pr for entry in merged for pr in entry.prs}
        lines.append(f"Covered by fragments: {len(fragment_prs)} PR(s), missing: {len(missing)} line(s).")
        lines.append("")
        lines.append("### Missing (review required)")
        lines.append("")
        if missing:
            lines.extend(f"- {line}" for line in missing)
        else:
            lines.append("_Every commit with a PR reference has a fragment._")
        lines.append("")
        lines.append("### Commits Without a PR Reference")
        lines.append("")
        if no_pr:
            lines.extend(f"- {line}" for line in no_pr)
        else:
            lines.append("_None._")
        lines.append("")
        lines.append("### Fragment PRs Not Found in the Commit List")
        lines.append("")
        if unmatched_prs:
            lines.append(
                "These fragments do not match any commit in this window. They are usually leftovers from"
                " an earlier release or PRs that were never merged, so archive or drop them before publishing:"
            )
            lines.extend(f"- #{pr}" for pr in unmatched_prs)
        else:
            lines.append("_Every fragment PR appears in the commit list._")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="Release version, e.g. v0.31.0")
    parser.add_argument(
        "--fragments",
        default=resolve_default_fragments(),
        help="Directory containing fragment YAML files (default: release_notes/fragments)",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo used to build PR links (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--raw-commits",
        help="Raw commits markdown file from fetch_commits.py, used for the missing-fragment audit",
    )
    parser.add_argument("--output", required=True, help="Output file for the aggregated draft")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Move the fragment files to <archive-dir>/<version>/ after a successful run "
        "(use when preparing the release PR)",
    )
    parser.add_argument(
        "--archive-dir",
        help="Root directory for archived fragments (default: <fragments parent>/archive)",
    )
    args = parser.parse_args()

    if args.archive_dir and not args.archive:
        parser.error("--archive-dir requires --archive")

    if "/" in args.version or "\\" in args.version or args.version.startswith("."):
        parser.error("--version must be a plain version string like v0.31.0 (it is used as a directory name)")

    fragments_dir = Path(args.fragments)
    raw_commits = Path(args.raw_commits) if args.raw_commits else None
    if raw_commits is not None and not raw_commits.is_file():
        print(f"Error: raw commits file not found: {raw_commits}", file=sys.stderr)
        return 1

    entries, files, errors = load_fragments(fragments_dir)
    if errors:
        print("Fragment validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    archive_dir = None
    if args.archive:
        archive_root = Path(args.archive_dir) if args.archive_dir else fragments_dir.parent / "archive"
        archive_dir = archive_root / args.version
        conflicts = sorted(f.name for f in files if (archive_dir / f.name).exists())
        if conflicts:
            print(f"Error: {archive_dir} already contains: {', '.join(conflicts)}", file=sys.stderr)
            return 1

    print(f"Found {len(files)} fragment file(s) and {len(entries)} entries in {fragments_dir}")
    if not files:
        print("  Warning: no fragments found. Did the contributors forget, or is the path wrong?")

    merged = merge_entries(entries)

    audit_result = None
    if raw_commits is not None:
        fragment_prs = {pr for entry in merged for pr in entry.prs}
        audit_result = audit_raw_commits(raw_commits, fragment_prs)

    draft = render_draft(args.version, merged, args.repo, len(files), raw_commits, audit_result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draft, encoding="utf-8")
    print(f"Draft written to {output}")

    if audit_result is not None:
        missing, no_pr, unmatched_prs = audit_result
        print(
            f"Audit: {len(missing)} commit(s) with a PR reference and no fragment, "
            f"{len(no_pr)} without a PR reference, {len(unmatched_prs)} fragment PR(s) outside the commit list"
        )

    if archive_dir is not None:
        if not files:
            print("No fragments to archive.")
        else:
            if audit_result is not None and audit_result[2]:
                print(
                    f"Warning: {len(audit_result[2])} fragment PR(s) are not in the commit list and will be"
                    f" archived under {args.version}. Confirm they belong to this release.",
                    file=sys.stderr,
                )
            archive_dir.mkdir(parents=True, exist_ok=True)
            try:
                for path in files:
                    shutil.move(str(path), str(archive_dir / path.name))
            except OSError as exc:
                print(f"Error: failed to archive: {exc}", file=sys.stderr)
                print("Re-run with --archive after fixing the problem; moved files are safe", file=sys.stderr)
                return 1
            print(f"Archived {len(files)} fragment file(s) to {archive_dir}")
            print("Commit the move in the release PR; fragments are archived, never deleted")
    else:
        print("Fragments kept. Re-run with --archive when preparing the release PR")

    return 0


if __name__ == "__main__":
    sys.exit(main())
