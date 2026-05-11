#!/usr/bin/env python3
"""Deterministic step planner for the main2main upgrade pipeline.

Splits a range of upstream vLLM commits into ordered steps based on changed
lines in vllm/ source files.

Algorithm:
  1. git rev-list --reverse base..target → ordered commit list
  2. For each commit, git diff --numstat → changed files + lines
  3. Classify files: vllm/ source → "vllm", requirements* → "requirements",
     everything else → "ignored"
  4. Requirements commits get their own step (dependency changes are isolated)
  5. Commits accumulate into a step until vllm_changed_lines exceeds 500
  6. A single commit with vllm_changed_lines > 500 becomes its own step
  7. "ignored" files (docs, tests, CI) can be batched into any step but don't
     count toward the line budget

Output:
  - /tmp/main2main/steps.json  — machine-readable step plan
  - /tmp/main2main/steps.md    — human-readable plan summary
  - stdout: JSON summary

Usage:
    python3 plan_steps.py \\
      --vllm-path <path> \\
      --base-commit <sha> \\
      --target-commit <sha>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

LINE_BUDGET = 500


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _list_commits(repo: Path, base: str, target: str) -> list[dict[str, str]]:
    """Return commits in chronological order (oldest first)."""
    log_output = _run_git(
        repo, "log", "--reverse", "--format=%H%x1f%s", f"{base}..{target}",
    )
    commits: list[dict[str, str]] = []
    for line in log_output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f", 1)
        commits.append({
            "sha": parts[0].strip(),
            "subject": parts[1].strip() if len(parts) > 1 else "",
        })
    return commits


def _numstat(repo: Path, sha: str) -> list[dict[str, Any]]:
    """Return per-file change stats for a single commit."""
    output = _run_git(repo, "diff-tree", "--no-commit-id", "-r", "--numstat", sha)
    files: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = int(parts[0]) if parts[0] != "-" else 0
        deleted = int(parts[1]) if parts[1] != "-" else 0
        filepath = parts[2]
        files.append({
            "path": filepath,
            "added": added,
            "deleted": deleted,
            "lines": added + deleted,
        })
    return files


def _classify_file(filepath: str) -> str:
    """Classify a file as 'vllm', 'requirements', or 'ignored'."""
    if filepath.startswith("requirements") or filepath == "pyproject.toml":
        return "requirements"
    if filepath.startswith("vllm/"):
        return "vllm"
    return "ignored"


def _commit_stats(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregated stats for a commit's changed files."""
    vllm_lines = 0
    total_lines = 0
    categories: set[str] = set()
    has_requirements = False

    for f in files:
        cat = _classify_file(f["path"])
        categories.add(cat)
        total_lines += f["lines"]
        if cat == "vllm":
            vllm_lines += f["lines"]
        if cat == "requirements":
            has_requirements = True

    return {
        "vllm_changed_lines": vllm_lines,
        "total_changed_lines": total_lines,
        "categories": sorted(categories),
        "has_requirements": has_requirements,
        "files": [f["path"] for f in files],
    }


def plan_steps(
    commits: list[dict[str, str]],
    stats_per_commit: dict[str, dict[str, Any]],
    base_commit: str,
) -> list[dict[str, Any]]:
    """Group commits into steps respecting line budget."""
    steps: list[dict[str, Any]] = []

    current_commits: list[dict[str, str]] = []
    current_vllm_lines = 0
    current_total_lines = 0
    current_cats: set[str] = set()
    current_files: list[str] = []

    def _flush(start: str) -> None:
        nonlocal current_commits, current_vllm_lines, current_total_lines
        nonlocal current_cats, current_files
        if not current_commits:
            return
        steps.append({
            "index": len(steps) + 1,
            "id": f"step-{len(steps) + 1}",
            "commits": list(current_commits),
            "start_commit": start,
            "end_commit": current_commits[-1]["sha"],
            "categories": sorted(current_cats),
            "vllm_changed_lines": current_vllm_lines,
            "total_changed_lines": current_total_lines,
            "line_budget": LINE_BUDGET,
            "files_changed": sorted(set(current_files)),
        })
        current_commits = []
        current_vllm_lines = 0
        current_total_lines = 0
        current_cats = set()
        current_files = []

    prev_end = base_commit
    for commit in commits:
        st = stats_per_commit.get(commit["sha"], {})
        vllm_lines = st.get("vllm_changed_lines", 0)
        total_lines = st.get("total_changed_lines", 0)
        cats = set(st.get("categories", []))
        files = st.get("files", [])
        has_req = st.get("has_requirements", False)

        # Requirements commits get their own step
        if has_req:
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            current_commits = [commit]
            current_vllm_lines = vllm_lines
            current_total_lines = total_lines
            current_cats = cats
            current_files = list(files)
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            continue

        # Single commit exceeding budget → solo step
        if vllm_lines > LINE_BUDGET:
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            current_commits = [commit]
            current_vllm_lines = vllm_lines
            current_total_lines = total_lines
            current_cats = cats
            current_files = list(files)
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            continue

        # Would exceed budget? → flush first
        if current_vllm_lines + vllm_lines > LINE_BUDGET:
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit

        current_commits.append(commit)
        current_vllm_lines += vllm_lines
        current_total_lines += total_lines
        current_cats.update(cats)
        current_files.extend(files)

    # Flush remaining
    _flush(prev_end)

    return steps


def _render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# main2main Step Plan",
        "",
        f"**Base:** `{plan['base_commit']}`",
        f"**Target:** `{plan['target_commit']}`",
        f"**Commits:** {plan['total_commits']}  |  **Steps:** {len(plan['steps'])}",
        "",
    ]
    for step in plan["steps"]:
        lines.append(f"## {step['id']} (vllm: {step['vllm_changed_lines']} lines, "
                      f"total: {step['total_changed_lines']} lines)")
        lines.append("")
        lines.append(f"- Categories: {', '.join(step['categories'])}")
        lines.append(f"- Range: `{step['start_commit'][:8]}..{step['end_commit'][:8]}`")
        lines.append("")
        for c in step["commits"]:
            lines.append(f"  - `{c['sha'][:8]}` {c['subject']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan upgrade steps for main2main pipeline.",
    )
    parser.add_argument("--vllm-path", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--target-commit", required=True)
    args = parser.parse_args()

    commits = _list_commits(args.vllm_path, args.base_commit, args.target_commit)
    if not commits:
        plan = {
            "base_commit": args.base_commit,
            "target_commit": args.target_commit,
            "total_commits": 0,
            "steps": [],
        }
    else:
        stats_per_commit: dict[str, dict[str, Any]] = {}
        for c in commits:
            files = _numstat(args.vllm_path, c["sha"])
            stats_per_commit[c["sha"]] = _commit_stats(files)

        steps = plan_steps(commits, stats_per_commit, args.base_commit)

        plan = {
            "base_commit": args.base_commit,
            "target_commit": args.target_commit,
            "total_commits": len(commits),
            "steps": steps,
        }

    # Write outputs
    output_dir = Path("/tmp/main2main")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "steps.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "steps.md").write_text(
        _render_markdown(plan), encoding="utf-8",
    )

    for step in plan["steps"]:
        step_dir = output_dir / "steps" / step["id"]
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "ci").mkdir(exist_ok=True)

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
