#!/usr/bin/env python3
"""Deterministic step planner for the main2main upgrade pipeline.

Splits a range of upstream vLLM commits into ordered steps using fixed
requirements and vLLM package line-count rules.

Algorithm:
  1. git rev-list --reverse base..target → ordered commit list.
  2. For each commit, git diff --name-only/--numstat → changed files/lines.
  3. Requirements/dependency commits become their own step.
  4. vLLM package changes are batched within a 500 line budget.
  5. Non-vLLM, non-requirements paths merge into adjacent steps without
     increasing vllm_changed_lines.

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
from pathlib import Path
from typing import Any

REQUIREMENTS_PREFIXES = ("requirements", "constraints")
REQUIREMENTS_FILES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "uv.lock",
    "poetry.lock",
}
VLLM_CODE_PREFIX = "vllm/"
VLLM_LINE_BUDGET = 500


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
        commits.append({"sha": parts[0].strip(), "subject": parts[1].strip() if len(parts) > 1 else ""})
    return commits


def _changed_files(repo: Path, sha: str) -> list[str]:
    """Return the list of files changed in a single commit."""
    output = _run_git(repo, "diff-tree", "--no-commit-id", "-r", "--name-only", sha)
    return [f for f in output.strip().splitlines() if f.strip()]


def _changed_line_count(
    repo: Path,
    sha: str,
    prefixes: tuple[str, ...] | None = (VLLM_CODE_PREFIX,),
) -> int:
    """Return added+deleted lines for selected paths in a single commit."""
    output = _run_git(repo, "diff-tree", "--no-commit-id", "-r", "--numstat", sha)
    total = 0
    for line in output.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        filepath = parts[2]
        if prefixes is not None and not filepath.startswith(prefixes):
            continue
        added, deleted = parts[0], parts[1]
        if added.isdigit():
            total += int(added)
        if deleted.isdigit():
            total += int(deleted)
    return total


def _is_requirements_file(filepath: str) -> bool:
    return filepath in REQUIREMENTS_FILES or filepath.startswith(REQUIREMENTS_PREFIXES)


def _commit_categories(files: list[str]) -> list[str]:
    """Return fixed categories for a commit's changed files."""
    cats: set[str] = set()
    for f in files:
        if _is_requirements_file(f):
            cats.add("requirements")
        elif f.startswith(VLLM_CODE_PREFIX):
            cats.add("vllm")
        else:
            cats.add("ignored")
    return sorted(cats)


def plan_steps(
    commits: list[dict[str, str]],
    files_per_commit: dict[str, list[str]],
    base_commit: str,
    vllm_line_counts_per_commit: dict[str, int] | None = None,
    total_line_counts_per_commit: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Group commits into steps respecting budget constraints."""
    steps: list[dict[str, Any]] = []
    current_commits: list[dict[str, str]] = []
    current_vllm_lines = 0
    current_total_lines = 0
    current_line_budget = 0
    current_cats: set[str] = set()
    current_files: list[str] = []

    def _flush(start: str) -> None:
        nonlocal current_commits, current_vllm_lines, current_total_lines
        nonlocal current_line_budget
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
            "line_budget": current_line_budget,
            "files_changed": sorted(set(current_files)),
        })
        current_commits = []
        current_vllm_lines = 0
        current_total_lines = 0
        current_line_budget = 0
        current_cats = set()
        current_files = []

    def _line_budget_for(vllm_changed_lines: int) -> int:
        return VLLM_LINE_BUDGET if vllm_changed_lines > 0 else 0

    prev_end = base_commit
    for commit in commits:
        files = files_per_commit.get(commit["sha"], [])
        cats = _commit_categories(files)
        vllm_changed_lines = 0
        if vllm_line_counts_per_commit is not None:
            vllm_changed_lines = vllm_line_counts_per_commit.get(commit["sha"], 0)
        total_changed_lines = vllm_changed_lines
        if total_line_counts_per_commit is not None:
            total_changed_lines = total_line_counts_per_commit.get(commit["sha"], 0)
        line_budget = _line_budget_for(vllm_changed_lines)

        # Requirements/dependency commit → flush current batch, then solo step.
        if "requirements" in cats:
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            current_commits.append(commit)
            current_vllm_lines = vllm_changed_lines
            current_total_lines = total_changed_lines
            current_line_budget = line_budget
            current_cats.update(cats)
            current_files.extend(files)
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            continue

        # A single large vLLM package commit gets its own step, but can absorb
        # adjacent ignored-only commits because they do not affect the line budget.
        if vllm_changed_lines > VLLM_LINE_BUDGET:
            if current_commits and current_vllm_lines > 0:
                _flush(prev_end)
                prev_end = steps[-1]["end_commit"] if steps else base_commit
            current_commits.append(commit)
            current_vllm_lines += vllm_changed_lines
            current_total_lines += total_changed_lines
            current_line_budget = VLLM_LINE_BUDGET
            current_cats.update(cats)
            current_files.extend(files)
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            continue

        # Would exceed budget? → flush first
        next_line_budget = current_line_budget
        if line_budget:
            next_line_budget = min(current_line_budget, line_budget) if current_line_budget else line_budget
        if (next_line_budget and current_commits and
                current_vllm_lines + vllm_changed_lines > next_line_budget):
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            next_line_budget = line_budget

        current_commits.append(commit)
        current_vllm_lines += vllm_changed_lines
        current_total_lines += total_changed_lines
        current_line_budget = next_line_budget
        current_cats.update(cats)
        current_files.extend(files)

    # Flush remaining
    _flush(prev_end)

    return steps


def _render_markdown(plan: dict[str, Any]) -> str:
    """Render a human-readable markdown summary of the step plan."""
    lines = [
        f"# main2main Step Plan",
        f"",
        f"**Base commit:** `{plan['base_commit']}`",
        f"**Target commit:** `{plan['target_commit']}`",
        f"**Total commits:** {plan['total_commits']}",
        f"**Steps:** {len(plan['steps'])}",
        f"",
    ]
    for step in plan["steps"]:
        lines.append(f"## Step {step['index']}: {step['id']}")
        lines.append(f"")
        lines.append(f"- **vLLM changed lines:** {step.get('vllm_changed_lines', 0)}")
        lines.append(f"- **Total changed lines:** {step.get('total_changed_lines', 0)}")
        lines.append(f"- **Line budget:** {step.get('line_budget', 0)}")
        lines.append(f"- **Categories:** {', '.join(step['categories'])}")
        lines.append(f"- **Commits:** {len(step['commits'])}")
        lines.append(f"- **Range:** `{step['start_commit'][:8]}..{step['end_commit'][:8]}`")
        lines.append(f"")
        for c in step["commits"]:
            lines.append(f"  - `{c['sha'][:8]}` {c['subject']}")
        lines.append(f"")
        if step["files_changed"]:
            lines.append(f"**Changed files:**")
            for f in step["files_changed"][:20]:  # cap at 20
                lines.append(f"  - {f}")
            if len(step["files_changed"]) > 20:
                lines.append(f"  - ... and {len(step['files_changed']) - 20} more")
            lines.append(f"")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan upgrade steps for main2main pipeline.",
    )
    parser.add_argument("--vllm-path", type=Path, required=True,
                        help="Path to the local vLLM repository")
    parser.add_argument("--base-commit", required=True,
                        help="Starting commit (current pinned commit)")
    parser.add_argument("--target-commit", required=True,
                        help="Target commit (vLLM HEAD)")
    args = parser.parse_args()

    # Enumerate commits
    commits = _list_commits(args.vllm_path, args.base_commit, args.target_commit)
    if not commits:
        plan = {
            "base_commit": args.base_commit,
            "target_commit": args.target_commit,
            "total_commits": 0,
            "steps": [],
        }
    else:
        # Get changed files per commit
        files_per_commit: dict[str, list[str]] = {}
        vllm_line_counts_per_commit: dict[str, int] = {}
        total_line_counts_per_commit: dict[str, int] = {}
        for c in commits:
            files_per_commit[c["sha"]] = _changed_files(args.vllm_path, c["sha"])
            vllm_line_counts_per_commit[c["sha"]] = _changed_line_count(args.vllm_path, c["sha"])
            total_line_counts_per_commit[c["sha"]] = _changed_line_count(
                args.vllm_path, c["sha"], prefixes=None,
            )

        steps = plan_steps(
            commits,
            files_per_commit,
            args.base_commit,
            vllm_line_counts_per_commit,
            total_line_counts_per_commit,
        )

        plan = {
            "base_commit": args.base_commit,
            "target_commit": args.target_commit,
            "total_commits": len(commits),
            "steps": steps,
        }

    # Write outputs
    output_dir = Path("/tmp/main2main")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "steps.json"
    json_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    md_path = output_dir / "steps.md"
    md_path.write_text(_render_markdown(plan), encoding="utf-8")

    # Create step directories
    for step in plan["steps"]:
        step_dir = output_dir / "steps" / step["id"]
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "ci").mkdir(exist_ok=True)

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
