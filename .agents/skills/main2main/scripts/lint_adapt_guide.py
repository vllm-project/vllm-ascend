#!/usr/bin/env python3
"""Lint the lookup tables in reference/adapt-guide.md.

The adapt-guide carries three lookup regions that route per-step adapt work:
  - key-areas     : upstream vLLM subsystems with adaptation hints
  - file-locations: where Ascend-specific code lives
  - file-mapping  : upstream path -> vllm-ascend path

These regions reference concrete paths that drift as vllm-ascend evolves.
This script reports drift; it does not edit the guide. The agent reads the
report (plus the current step patches) and updates the AUTO-MAINTAINED
regions itself, so the human-written prose around the tables stays untouched.

Usage:
    python3 lint_adapt_guide.py \
      --ascend-path <path> \
      [--patches-dir /tmp/main2main/steps] \
      [--output /tmp/main2main/adapt-guide-refresh/check_report.md]

Output:
    A Markdown report at --output with three sections:
      1. Invalidated paths     : paths the guide cites that no longer exist
      2. Uncovered directories : vllm_ascend/ subdirs that no region mentions
      3. Untouched upstream    : vllm/ paths changed this run but missing from
                                 the file-mapping region (only if patches found)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REGION_NAMES = ("key-areas", "file-locations", "file-mapping")

REGION_RE = {
    name: re.compile(
        r"<!-- BEGIN AUTO-MAINTAINED: " + re.escape(name) + r" -->\n"
        r"(?P<body>.*?)\n"
        r"<!-- END AUTO-MAINTAINED: " + re.escape(name) + r" -->",
        re.DOTALL,
    )
    for name in REGION_NAMES
}

# Match backtick-quoted paths like `vllm_ascend/foo/bar.py` or `vllm/v1/...`.
# Trailing slash is treated as a directory reference.
PATH_RE = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_./*-]*)`")

# Path prefixes we consider "concrete enough" to lint.
ASCEND_PREFIX = "vllm_ascend/"
UPSTREAM_PREFIX = "vllm/"

# Directories under vllm_ascend/ that don't represent adaptation surface area
# and are intentionally not listed in the guide.
ASCEND_IGNORED_DIRS = {
    "__pycache__",
    "tests",
    "test",
}


def _extract_region(text: str, name: str) -> str | None:
    match = REGION_RE[name].search(text)
    if not match:
        return None
    return match.group("body")


def _paths_in(region_body: str, prefix: str) -> list[str]:
    """Pull backtick-quoted paths that start with prefix out of region body.

    Glob-like entries (containing `*`) are skipped — they're patterns, not
    concrete paths.
    """
    found: list[str] = []
    for match in PATH_RE.finditer(region_body):
        raw = match.group(1)
        if not raw.startswith(prefix):
            continue
        if "*" in raw:
            continue
        found.append(raw)
    # Preserve order but dedupe.
    seen: set[str] = set()
    deduped: list[str] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _path_exists(repo: Path, rel: str) -> bool:
    """Check existence treating trailing-slash entries as directories."""
    candidate = repo / rel.rstrip("/")
    if rel.endswith("/"):
        return candidate.is_dir()
    return candidate.exists()


def _check_invalidated(repo: Path, regions: dict[str, str | None]) -> list[dict]:
    """List ascend paths referenced in any region but missing from the repo."""
    invalid: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for name in REGION_NAMES:
        body = regions[name]
        if body is None:
            continue
        for path in _paths_in(body, ASCEND_PREFIX):
            if _path_exists(repo, path):
                continue
            key = (name, path)
            if key in seen:
                continue
            seen.add(key)
            invalid.append({"region": name, "path": path})
    return invalid


def _ascend_top_level_dirs(repo: Path) -> list[str]:
    """Return top-level subdirectories under vllm_ascend/ as `vllm_ascend/<x>/`."""
    base = repo / "vllm_ascend"
    if not base.is_dir():
        return []
    dirs: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            # Skip dotfiles and Python private modules like __pycache__.
            if child.name not in ASCEND_IGNORED_DIRS and not child.name.startswith("__"):
                dirs.append(f"vllm_ascend/{child.name}/")
            continue
        if child.name in ASCEND_IGNORED_DIRS:
            continue
        dirs.append(f"vllm_ascend/{child.name}/")
    return dirs


def _check_uncovered_dirs(repo: Path, regions: dict[str, str | None]) -> list[str]:
    """List vllm_ascend/<dir>/ that no region mentions (even as prefix)."""
    actual = _ascend_top_level_dirs(repo)
    mentioned: set[str] = set()
    for name in REGION_NAMES:
        body = regions[name]
        if body is None:
            continue
        # Any backtick path beginning with vllm_ascend/<x>/ counts as coverage.
        for path in _paths_in(body, ASCEND_PREFIX):
            parts = path.split("/")
            if len(parts) >= 2 and parts[0] == "vllm_ascend":
                mentioned.add(f"vllm_ascend/{parts[1]}/")
    return [d for d in actual if d not in mentioned]


def _collect_changed_files(patches_dir: Path) -> list[str]:
    """Read every changed-files.txt under patches_dir/<step-id>/, dedupe.

    Falls back to parsing `diff --git` lines from upstream.patch when
    changed-files.txt is absent.
    """
    if not patches_dir.is_dir():
        return []

    files: set[str] = set()
    for step_dir in sorted(patches_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        changed = step_dir / "changed-files.txt"
        if changed.is_file():
            for line in changed.read_text().splitlines():
                line = line.strip()
                if line:
                    files.add(line)
            continue
        patch = step_dir / "upstream.patch"
        if patch.is_file():
            for line in patch.read_text(errors="replace").splitlines():
                if line.startswith("diff --git a/"):
                    parts = line.split()
                    if len(parts) >= 3:
                        files.add(parts[2][2:])  # strip "a/"
    return sorted(files)


def _check_untouched_upstream(regions: dict[str, str | None], changed_files: list[str]) -> list[str]:
    """List vllm/ paths touched this run that file-mapping doesn't cover.

    Coverage rule: a row covers a changed file when the row's upstream cell
    is a prefix of the changed file. Comparison is path-prefix only; this
    deliberately overflags rather than missing real gaps — the agent decides
    what to add.
    """
    body = regions["file-mapping"]
    if body is None:
        return []
    covered_prefixes = [p for p in _paths_in(body, UPSTREAM_PREFIX)]

    untouched: list[str] = []
    for changed in changed_files:
        if not changed.startswith(UPSTREAM_PREFIX):
            continue
        if any(
            changed.startswith(prefix.rstrip("/") + "/") or changed == prefix.rstrip("/") or changed.startswith(prefix)
            for prefix in covered_prefixes
        ):
            continue
        untouched.append(changed)
    return untouched


def _render_report(
    invalid: list[dict], uncovered: list[str], untouched: list[str], patches_dir: Path, patches_seen: bool
) -> str:
    lines: list[str] = ["# Adapt Guide Lint Report", ""]

    lines.append("## 1. Invalidated paths")
    lines.append("")
    lines.append("Paths the guide cites that no longer exist in vllm_ascend/.")
    lines.append("")
    if invalid:
        for entry in invalid:
            lines.append(f"- [{entry['region']}] `{entry['path']}`")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## 2. Uncovered directories")
    lines.append("")
    lines.append("vllm_ascend/ subdirectories that no AUTO-MAINTAINED region currently references.")
    lines.append("")
    if uncovered:
        for path in uncovered:
            lines.append(f"- `{path}`")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## 3. Untouched upstream paths (this run)")
    lines.append("")
    if not patches_seen:
        lines.append(f"_No step patches found under `{patches_dir}` — section skipped._")
    else:
        lines.append("vLLM paths changed in this run's step patches but not covered by any file-mapping row.")
        lines.append("")
        if untouched:
            for path in untouched:
                lines.append(f"- `{path}`")
        else:
            lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint the AUTO-MAINTAINED tables in adapt-guide.md.",
    )
    parser.add_argument("--ascend-path", type=Path, required=True, help="Path to the vllm-ascend repository.")
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=Path("/tmp/main2main/steps"),
        help="Directory containing per-step patch outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/main2main/adapt-guide-refresh/check_report.md"),
        help="Where to write the Markdown report.",
    )
    args = parser.parse_args()

    guide = args.ascend_path / ".agents/skills/main2main/reference/adapt-guide.md"
    if not guide.is_file():
        print(f"error: adapt-guide.md not found at {guide}", file=sys.stderr)
        return 1

    text = guide.read_text()
    regions = {name: _extract_region(text, name) for name in REGION_NAMES}
    missing = [name for name, body in regions.items() if body is None]
    if missing:
        print(
            "error: missing AUTO-MAINTAINED region(s) in adapt-guide.md: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    invalid = _check_invalidated(args.ascend_path, regions)
    uncovered = _check_uncovered_dirs(args.ascend_path, regions)
    changed = _collect_changed_files(args.patches_dir)
    untouched = _check_untouched_upstream(regions, changed)

    report = _render_report(
        invalid=invalid,
        uncovered=uncovered,
        untouched=untouched,
        patches_dir=args.patches_dir,
        patches_seen=bool(changed),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")
    print(f"  invalidated paths   : {len(invalid)}")
    print(f"  uncovered dirs      : {len(uncovered)}")
    print(f"  untouched upstream  : {len(untouched)}" + ("" if changed else " (no patches found)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
