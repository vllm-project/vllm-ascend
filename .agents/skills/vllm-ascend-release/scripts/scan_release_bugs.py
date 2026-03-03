#!/usr/bin/env python3
"""
Scan GitHub issues to identify release-blocking bugs.

This script analyzes open bugs and prioritizes them based on:
- Severity indicators (crash, data loss, security, regression)
- User impact (reactions, comments, linked issues)
- Recency (bugs in current release cycle)
- Labels (priority:high, regression, blocker)

Usage:
    python scan_release_bugs.py \
        --repo vllm-project/vllm-ascend \
        --days 30 \
        --output bug-analysis.md
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Bug:
    number: int
    title: str
    url: str
    created_at: str
    labels: list[str]
    reactions: int
    comments: int
    priority: str = "P3"
    reason: str = ""


# Keywords that indicate high severity
BLOCKER_KEYWORDS = [
    "crash",
    "hang",
    "freeze",
    "data loss",
    "security",
    "vulnerability",
    "memory leak",
    "oom",
    "out of memory",
    "deadlock",
    "infinite loop",
]

CRITICAL_KEYWORDS = [
    "regression",
    "broken",
    "fail",
    "error",
    "exception",
    "not working",
    "cannot",
    "unable",
    "block",
]

# Labels that indicate priority
BLOCKER_LABELS = ["blocker", "priority:critical", "priority:p0", "security"]
CRITICAL_LABELS = ["priority:high", "priority:p1", "regression"]
IMPORTANT_LABELS = ["priority:medium", "priority:p2"]


def run_gh_command(args: list[str]) -> str:
    """Run a GitHub CLI command and return the output."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"gh command failed: {result.stderr}")
    return result.stdout


def fetch_bugs(repo: str, days: int) -> list[dict]:
    """Fetch open bugs from the repository."""
    # Note: days parameter reserved for future filtering by date
    _ = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Fetch bugs with the 'bug' label
    output = run_gh_command(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--label",
            "bug",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,url,createdAt,labels,reactionGroups,comments",
        ]
    )

    issues = json.loads(output)
    return issues


def calculate_reactions(reaction_groups: list[dict]) -> int:
    """Calculate total reactions from reaction groups."""
    total = 0
    for group in reaction_groups:
        total += group.get("totalCount", 0)
    return total


def prioritize_bug(issue: dict) -> Bug:
    """Analyze a bug and assign priority."""
    title_lower = issue["title"].lower()
    labels = [label["name"].lower() for label in issue.get("labels", [])]
    reactions = calculate_reactions(issue.get("reactionGroups", []))
    comments = len(issue.get("comments", []))

    bug = Bug(
        number=issue["number"],
        title=issue["title"],
        url=issue["url"],
        created_at=issue["createdAt"],
        labels=labels,
        reactions=reactions,
        comments=comments,
    )

    # Check for blocker indicators
    for keyword in BLOCKER_KEYWORDS:
        if keyword in title_lower:
            bug.priority = "P0"
            bug.reason = f"Contains blocker keyword: '{keyword}'"
            return bug

    for label in BLOCKER_LABELS:
        if label in labels:
            bug.priority = "P0"
            bug.reason = f"Has blocker label: '{label}'"
            return bug

    # Check for critical indicators
    for keyword in CRITICAL_KEYWORDS:
        if keyword in title_lower:
            bug.priority = "P1"
            bug.reason = f"Contains critical keyword: '{keyword}'"
            return bug

    for label in CRITICAL_LABELS:
        if label in labels:
            bug.priority = "P1"
            bug.reason = f"Has critical label: '{label}'"
            return bug

    # Check for important indicators
    for label in IMPORTANT_LABELS:
        if label in labels:
            bug.priority = "P2"
            bug.reason = f"Has important label: '{label}'"
            return bug

    # Check user engagement
    if reactions >= 10 or comments >= 10:
        bug.priority = "P2"
        bug.reason = f"High engagement: {reactions} reactions, {comments} comments"
        return bug

    if reactions >= 5 or comments >= 5:
        bug.priority = "P3"
        bug.reason = f"Moderate engagement: {reactions} reactions, {comments} comments"
        return bug

    bug.priority = "P3"
    bug.reason = "Standard bug"
    return bug


def generate_report(bugs: list[Bug], repo: str) -> str:
    """Generate a markdown report of prioritized bugs."""
    lines = [
        "## Release Bug Analysis",
        "",
        f"Repository: {repo}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total bugs analyzed: {len(bugs)}",
        "",
    ]

    # Group by priority
    by_priority = {"P0": [], "P1": [], "P2": [], "P3": []}
    for bug in bugs:
        by_priority[bug.priority].append(bug)

    # P0 - Blockers
    lines.append("### P0 - Blockers (Must Fix)")
    lines.append("")
    if by_priority["P0"]:
        for bug in by_priority["P0"]:
            lines.append(f"- [ ] https://github.com/{repo}/issues/{bug.number}")
            lines.append(f"  - **{bug.title}**")
            lines.append(f"  - Reason: {bug.reason}")
            lines.append("")
    else:
        lines.append("No blocker bugs identified.")
        lines.append("")

    # P1 - Critical
    lines.append("### P1 - Critical (Should Fix)")
    lines.append("")
    if by_priority["P1"]:
        for bug in by_priority["P1"]:
            lines.append(f"- [ ] https://github.com/{repo}/issues/{bug.number}")
            lines.append(f"  - **{bug.title}**")
            lines.append(f"  - Reason: {bug.reason}")
            lines.append("")
    else:
        lines.append("No critical bugs identified.")
        lines.append("")

    # P2 - Important
    lines.append("### P2 - Important (Fix If Possible)")
    lines.append("")
    if by_priority["P2"]:
        for bug in by_priority["P2"][:10]:  # Limit to top 10
            lines.append(f"- [ ] https://github.com/{repo}/issues/{bug.number}")
            lines.append(f"  - {bug.title}")
            lines.append("")
        if len(by_priority["P2"]) > 10:
            lines.append(f"... and {len(by_priority['P2']) - 10} more P2 bugs")
            lines.append("")
    else:
        lines.append("No important bugs identified.")
        lines.append("")

    # Summary for checklist
    lines.append("### Summary for Release Checklist")
    lines.append("")
    lines.append("Copy the following to the 'Bug need Solve' section:")
    lines.append("")
    lines.append("```markdown")
    for priority in ["P0", "P1"]:
        for bug in by_priority[priority]:
            lines.append(f"- [ ] https://github.com/{repo}/issues/{bug.number}")
    # Add top P2 bugs
    for bug in by_priority["P2"][:5]:
        lines.append(f"- [ ] https://github.com/{repo}/issues/{bug.number}")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan GitHub issues for release-blocking bugs")
    parser.add_argument("--repo", default="vllm-project/vllm-ascend", help="Repository")
    parser.add_argument("--days", type=int, default=30, help="Look back period in days")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    print(f"Fetching bugs from {args.repo}...")
    issues = fetch_bugs(args.repo, args.days)

    print(f"Analyzing {len(issues)} bugs...")
    bugs = [prioritize_bug(issue) for issue in issues]

    # Sort by priority
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    bugs.sort(key=lambda b: (priority_order[b.priority], -b.reactions, -b.comments))

    print("Generating report...")
    report = generate_report(bugs, args.repo)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Bug analysis saved to {output_path}")

    # Print summary
    by_priority = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    for bug in bugs:
        by_priority[bug.priority] += 1

    print("\nSummary:")
    print(f"  P0 (Blockers): {by_priority['P0']}")
    print(f"  P1 (Critical): {by_priority['P1']}")
    print(f"  P2 (Important): {by_priority['P2']}")
    print(f"  P3 (Normal): {by_priority['P3']}")

    return 0


if __name__ == "__main__":
    exit(main())
