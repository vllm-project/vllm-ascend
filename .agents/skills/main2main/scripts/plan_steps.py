#!/usr/bin/env python3
"""Deterministic step planner for the main2main upgrade pipeline.

Splits a range of upstream vLLM commits into ordered steps based on file
categories and risk scores defined in step-planner.yaml.

Algorithm:
  1. git rev-list --reverse base..target → ordered commit list.
  2. For each commit, git diff --name-only → changed files.
  3. Classify files using step-planner.yaml patterns → risk score per commit.
  4. High-risk commits (>= force_solo_threshold) become their own step.
  5. Remaining commits are batched greedily within risk_budget and
     commit_count_budget limits.

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
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml


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


def _classify_file(filepath: str, categories: dict[str, Any]) -> tuple[str, int]:
    """Return (category_name, risk_score) for a file path."""
    for cat_name, cat_cfg in categories.items():
        for pattern in cat_cfg["patterns"]:
            if pattern == "*":
                continue  # skip fallback during first pass
            if filepath.startswith(pattern) or fnmatch(filepath, pattern):
                return cat_name, cat_cfg["risk_score"]
    # Fallback to misc
    if "misc" in categories:
        return "misc", categories["misc"]["risk_score"]
    return "misc", 3


def _commit_risk(files: list[str], categories: dict[str, Any]) -> tuple[int, list[str]]:
    """Return (max_risk_score, list_of_categories) for a commit's changed files."""
    cats: set[str] = set()
    max_risk = 0
    for f in files:
        cat, risk = _classify_file(f, categories)
        cats.add(cat)
        max_risk = max(max_risk, risk)
    return max_risk, sorted(cats)


def plan_steps(
    commits: list[dict[str, str]],
    files_per_commit: dict[str, list[str]],
    categories: dict[str, Any],
    budgets: dict[str, int],
    base_commit: str,
) -> list[dict[str, Any]]:
    """Group commits into steps respecting budget constraints."""
    risk_budget = budgets.get("risk_budget", 12)
    count_budget = budgets.get("commit_count_budget", 5)
    solo_threshold = budgets.get("force_solo_threshold", 8)

    steps: list[dict[str, Any]] = []
    current_commits: list[dict[str, str]] = []
    current_risk = 0
    current_cats: set[str] = set()
    current_files: list[str] = []

    def _flush(start: str) -> None:
        nonlocal current_commits, current_risk, current_cats, current_files
        if not current_commits:
            return
        steps.append({
            "index": len(steps) + 1,
            "id": f"step-{len(steps) + 1}",
            "commits": list(current_commits),
            "start_commit": start,
            "end_commit": current_commits[-1]["sha"],
            "categories": sorted(current_cats),
            "risk_score": current_risk,
            "files_changed": sorted(set(current_files)),
        })
        current_commits = []
        current_risk = 0
        current_cats = set()
        current_files = []

    prev_end = base_commit
    for commit in commits:
        files = files_per_commit.get(commit["sha"], [])
        risk, cats = _commit_risk(files, categories)

        # High-risk commit → flush current batch, then solo step
        if risk >= solo_threshold:
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            current_commits = [commit]
            current_risk = risk
            current_cats = set(cats)
            current_files = list(files)
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit
            continue

        # Would exceed budget? → flush first
        if (current_risk + risk > risk_budget or
                len(current_commits) + 1 > count_budget):
            _flush(prev_end)
            prev_end = steps[-1]["end_commit"] if steps else base_commit

        current_commits.append(commit)
        current_risk = max(current_risk, risk)
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
        lines.append(f"- **Risk score:** {step['risk_score']}")
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
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to step-planner.yaml (default: same dir as this script)")
    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "step-planner.yaml")
    if not config_path.exists():
        print(f"Error: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    categories = config["categories"]
    budgets = config["budgets"]

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
        for c in commits:
            files_per_commit[c["sha"]] = _changed_files(args.vllm_path, c["sha"])

        steps = plan_steps(commits, files_per_commit, categories, budgets, args.base_commit)

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
