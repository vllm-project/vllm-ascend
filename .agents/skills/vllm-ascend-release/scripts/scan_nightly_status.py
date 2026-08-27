#!/usr/bin/env python3
"""
Scan Nightly CI status for release readiness.

This script:
1. Gets the latest Nightly-A3 and Nightly-A2 GitHub Action runs
2. Calls extract_and_analyze.py to analyze failures
3. Generates a markdown summary table for the release checklist

Usage:
    python scan_nightly_status.py \
        --repo vllm-project/vllm-ascend \
        --output nightly-status.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = "vllm-project/vllm-ascend"

# Nightly workflow files
NIGHTLY_WORKFLOWS = {
    "Nightly-A3": "nightly_test_vllm_main_a3.yaml",
    "Nightly-A2": "nightly_test_vllm_main.yaml",
}

# Path to extract_and_analyze.py (relative to this script)
EXTRACT_SCRIPT = Path(__file__).parent.parent.parent / "main2main-error-analysis" / "scripts" / "extract_and_analyze.py"


@dataclass
class NightlyRun:
    """A nightly workflow run."""

    workflow_name: str
    run_id: int
    run_url: str
    conclusion: str
    created_at: str
    total_jobs: int = 0
    failed_jobs: int = 0
    failed_tests: list[str] = field(default_factory=list)
    code_bugs: list[dict] = field(default_factory=list)
    env_flakes: list[dict] = field(default_factory=list)


def gh_api_json(endpoint: str, **params) -> dict:
    """Call `gh api` and return parsed JSON."""
    url = endpoint
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{endpoint}?{qs}"
    try:
        r = subprocess.run(
            ["gh", "api", url],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: 'gh' CLI not found. Install it or run 'gh auth login'.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gh api {url} failed: {e.stderr.strip()}", file=sys.stderr)
        return {}
    return json.loads(r.stdout)


def get_latest_workflow_run(repo: str, workflow_file: str) -> dict | None:
    """Get the latest run of a workflow."""
    data = gh_api_json(
        f"/repos/{repo}/actions/workflows/{workflow_file}/runs",
        per_page="1",
    )
    runs = data.get("workflow_runs", [])
    return runs[0] if runs else None


def run_extract_and_analyze(run_id: int) -> dict | None:
    """Call extract_and_analyze.py script and get analysis result."""
    if not EXTRACT_SCRIPT.exists():
        print(f"ERROR: extract_and_analyze.py not found at {EXTRACT_SCRIPT}", file=sys.stderr)
        return None

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--run-id",
                str(run_id),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: extract_and_analyze.py failed: {e.stderr}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse extract_and_analyze.py output: {e}", file=sys.stderr)
        return None


def analyze_workflow_run(repo: str, workflow_name: str, run: dict) -> NightlyRun:
    """Analyze a workflow run using extract_and_analyze.py."""
    run_id = run["id"]
    result = NightlyRun(
        workflow_name=workflow_name,
        run_id=run_id,
        run_url=run.get("html_url", ""),
        conclusion=run.get("conclusion", "unknown"),
        created_at=run.get("created_at", "")[:10],
    )

    # If success, no need to analyze
    if run.get("conclusion") == "success":
        # Get job count anyway
        jobs_data = gh_api_json(f"/repos/{repo}/actions/runs/{run_id}/jobs", per_page="100")
        result.total_jobs = len(jobs_data.get("jobs", []))
        return result

    # Call extract_and_analyze.py for detailed analysis
    print(f"  Analyzing run {run_id} with extract_and_analyze.py...", file=sys.stderr)
    analysis = run_extract_and_analyze(run_id)

    if analysis:
        result.total_jobs = analysis.get("total_jobs", 0)
        result.failed_jobs = analysis.get("failed_jobs_count", 0)
        result.failed_tests = analysis.get("failed_tests", [])
        result.code_bugs = analysis.get("code_bugs", [])
        result.env_flakes = analysis.get("env_flakes", [])

    return result


def generate_report(runs: list[NightlyRun], repo: str) -> str:
    """Generate a markdown report of nightly status."""
    lines = [
        "## Nightly CI Status",
        "",
        f"Repository: {repo}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Summary",
        "",
        "| Workflow | Status | Jobs | Failed Tests | Code Bugs | Env Flakes | Run |",
        "|----------|--------|------|--------------|-----------|------------|-----|",
    ]

    for run in runs:
        status_icon = "✅" if run.conclusion == "success" else "❌"
        status = f"{status_icon} {run.conclusion}"
        if run.total_jobs > 0:
            jobs = f"{run.failed_jobs}/{run.total_jobs} failed"
        else:
            jobs = "-"
        tests = str(len(run.failed_tests))
        bugs = str(len(run.code_bugs))
        flakes = str(len(run.env_flakes))
        link = f"[#{run.run_id}]({run.run_url})"
        lines.append(f"| {run.workflow_name} | {status} | {jobs} | {tests} | {bugs} | {flakes} | {link} |")

    lines.append("")

    # Detail section for each failed workflow
    for run in runs:
        if run.conclusion == "success":
            continue

        lines.append(f"### {run.workflow_name} Details")
        lines.append("")
        lines.append(f"**Run**: [{run.run_id}]({run.run_url})")
        lines.append(f"**Date**: {run.created_at}")
        lines.append("")

        if run.code_bugs:
            lines.append("#### Code Bugs (Need Fix)")
            lines.append("")
            lines.append("| Error Type | Message |")
            lines.append("|------------|---------|")
            for bug in run.code_bugs[:10]:
                msg = bug.get("error_message", "").replace("|", "\\|")
                if len(msg) > 80:
                    msg = msg[:77] + "..."
                lines.append(f"| {bug.get('error_type', 'Unknown')} | {msg} |")
            if len(run.code_bugs) > 10:
                lines.append(f"| ... | {len(run.code_bugs) - 10} more |")
            lines.append("")

        if run.env_flakes:
            lines.append("#### Environment Flakes")
            lines.append("")
            lines.append("| Error Type | Message |")
            lines.append("|------------|---------|")
            for flake in run.env_flakes[:5]:
                msg = flake.get("error_message", "").replace("|", "\\|")
                if len(msg) > 80:
                    msg = msg[:77] + "..."
                lines.append(f"| {flake.get('error_type', 'Unknown')} | {msg} |")
            if len(run.env_flakes) > 5:
                lines.append(f"| ... | {len(run.env_flakes) - 5} more |")
            lines.append("")

        if run.failed_tests:
            lines.append("#### Failed Tests")
            lines.append("")
            for test in run.failed_tests[:15]:
                lines.append(f"- `{test}`")
            if len(run.failed_tests) > 15:
                lines.append(f"- ... and {len(run.failed_tests) - 15} more")
            lines.append("")

    # Summary for checklist
    lines.append("### Summary for Release Checklist")
    lines.append("")
    lines.append("Copy the following to the 'Nightly Status' section:")
    lines.append("")
    lines.append("```markdown")
    all_passed = all(r.conclusion == "success" for r in runs)
    if all_passed:
        lines.append("- [x] All nightly tests passing")
    else:
        for run in runs:
            if run.conclusion == "success":
                lines.append(f"- [x] {run.workflow_name}: Passing")
            else:
                lines.append(
                    f"- [ ] {run.workflow_name}: {len(run.code_bugs)} code bugs, {len(run.failed_tests)} failed tests"
                )
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan Nightly CI status for release readiness")
    parser.add_argument("--repo", default=REPO, help="Repository")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    runs = []

    for workflow_name, workflow_file in NIGHTLY_WORKFLOWS.items():
        print(f"Fetching latest {workflow_name} run...", file=sys.stderr)

        run = get_latest_workflow_run(args.repo, workflow_file)
        if not run:
            print(f"  No runs found for {workflow_name}", file=sys.stderr)
            continue

        print(
            f"  Found run {run['id']}: {run.get('conclusion', 'unknown')}",
            file=sys.stderr,
        )

        result = analyze_workflow_run(args.repo, workflow_name, run)
        runs.append(result)

    if not runs:
        print("No nightly runs found.", file=sys.stderr)
        return 1

    # Generate report
    print("Generating report...", file=sys.stderr)
    report = generate_report(runs, args.repo)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}", file=sys.stderr)

    # Print summary
    print("\nSummary:", file=sys.stderr)
    for run in runs:
        status = "✅" if run.conclusion == "success" else "❌"
        print(
            f"  {status} {run.workflow_name}: {run.failed_jobs}/{run.total_jobs} jobs failed, "
            f"{len(run.code_bugs)} code bugs, {len(run.env_flakes)} env flakes",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    exit(main())
