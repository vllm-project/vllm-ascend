#!/usr/bin/env python3
"""
Scan GitHub issues since the last release to identify bugs and problems.

This script:
1. Gets the release date of the previous version (including rc versions)
2. Fetches all issues created since that date
3. Lists issues with titles for quick browsing
4. Allows marking issues that need detailed review

The output is designed for human review - browse titles quickly and
flag issues that need more investigation.

Usage:
    python scan_release_bugs.py \
        --repo vllm-project/vllm-ascend \
        --since-tag v0.15.0rc1 \
        --output issue-scan.md
"""

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Issue:
    """An issue from the repository."""

    number: int
    title: str
    url: str
    state: str
    created_at: str
    author: str
    labels: list[str] = field(default_factory=list)
    comments_count: int = 0
    reactions_count: int = 0
    body_preview: str = ""
    needs_review: bool = False
    review_reason: str = ""


def run_gh_command(args: list[str]) -> str:
    """Run a GitHub CLI command and return the output."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"gh command failed: {result.stderr}")
    return result.stdout


def get_release_date(repo: str, tag: str) -> str:
    """Get the release date for a tag."""
    try:
        output = run_gh_command(
            [
                "release",
                "view",
                tag,
                "--repo",
                repo,
                "--json",
                "publishedAt",
            ]
        )
        data = json.loads(output)
        return data.get("publishedAt", "")[:10]  # Just the date part
    except Exception:
        # Fallback: try to get tag creation date
        try:
            output = run_gh_command(
                [
                    "api",
                    f"repos/{repo}/git/refs/tags/{tag}",
                ]
            )
            data = json.loads(output)
            sha = data.get("object", {}).get("sha", "")
            if sha:
                commit_output = run_gh_command(
                    [
                        "api",
                        f"repos/{repo}/commits/{sha}",
                    ]
                )
                commit_data = json.loads(commit_output)
                date = commit_data.get("commit", {}).get("committer", {}).get("date", "")
                return date[:10] if date else ""
        except Exception:
            pass
        return ""


def fetch_issues_since(repo: str, since_date: str, state: str = "all") -> list[dict]:
    """Fetch issues created since a given date."""
    output = run_gh_command(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            state,
            "--search",
            f"created:>{since_date}",
            "--limit",
            "300",
            "--json",
            "number,title,url,state,createdAt,author,labels,comments,reactionGroups,body",
        ]
    )
    return json.loads(output)


def calculate_reactions(reaction_groups: list[dict]) -> int:
    """Calculate total reactions from reaction groups."""
    total = 0
    for group in reaction_groups:
        total += group.get("totalCount", 0)
    return total


def should_flag_for_review(issue: dict) -> tuple[bool, str]:
    """Determine if an issue should be flagged for detailed review."""
    title_lower = issue["title"].lower()
    labels = [label["name"].lower() for label in issue.get("labels", [])]
    reactions = calculate_reactions(issue.get("reactionGroups", []))
    comments = len(issue.get("comments", []))

    reasons = []

    # High engagement indicates important issues
    if reactions >= 5:
        reasons.append(f"high reactions ({reactions})")
    if comments >= 5:
        reasons.append(f"many comments ({comments})")

    # Important labels
    important_labels = ["bug", "regression", "blocker", "priority:high", "critical"]
    for label in important_labels:
        if label in labels:
            reasons.append(f"has label '{label}'")

    # Keywords in title that suggest important issues
    important_keywords = [
        "crash",
        "hang",
        "freeze",
        "oom",
        "memory",
        "leak",
        "error",
        "fail",
        "broken",
        "regression",
        "block",
        "urgent",
        "critical",
    ]
    for keyword in important_keywords:
        if keyword in title_lower:
            reasons.append(f"title contains '{keyword}'")
            break

    return len(reasons) > 0, ", ".join(reasons)


def parse_issue(issue: dict) -> Issue:
    """Parse raw issue data into Issue object."""
    should_review, reason = should_flag_for_review(issue)

    body = issue.get("body", "") or ""
    body_preview = body[:300].replace("\n", " ").strip()
    if len(body) > 300:
        body_preview += "..."

    return Issue(
        number=issue["number"],
        title=issue["title"],
        url=issue["url"],
        state=issue["state"],
        created_at=issue.get("createdAt", "")[:10],
        author=issue.get("author", {}).get("login", "unknown"),
        labels=[label["name"] for label in issue.get("labels", [])],
        comments_count=len(issue.get("comments", [])),
        reactions_count=calculate_reactions(issue.get("reactionGroups", [])),
        body_preview=body_preview,
        needs_review=should_review,
        review_reason=reason,
    )


def generate_report(issues: list[Issue], repo: str, since_tag: str, since_date: str) -> str:
    """Generate a markdown report for human review."""
    lines = [
        "## Issue Scan Report",
        "",
        f"Repository: {repo}",
        f"Since: {since_tag} ({since_date})",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total issues: {len(issues)}",
        "",
    ]

    # Separate open and closed issues
    open_issues = [i for i in issues if i.state == "OPEN"]
    closed_issues = [i for i in issues if i.state == "CLOSED"]

    # Section 1: Issues flagged for review (high priority)
    flagged = [i for i in issues if i.needs_review]
    lines.append("### Flagged for Review")
    lines.append("")
    lines.append("These issues have been automatically flagged based on engagement or keywords.")
    lines.append("Please review the titles and check details if needed.")
    lines.append("")

    if flagged:
        for issue in flagged:
            state_icon = "🔴" if issue.state == "OPEN" else "🟢"
            lines.append(f"#### {state_icon} #{issue.number}: {issue.title}")
            lines.append("")
            lines.append(f"- **State**: {issue.state}")
            lines.append(f"- **Reason flagged**: {issue.review_reason}")
            lines.append(f"- **Labels**: {', '.join(issue.labels) if issue.labels else 'none'}")
            lines.append(f"- **Author**: @{issue.author}")
            lines.append(f"- **Created**: {issue.created_at}")
            lines.append(f"- **Engagement**: {issue.reactions_count} reactions, {issue.comments_count} comments")
            lines.append(f"- **Link**: {issue.url}")
            if issue.body_preview:
                lines.append(f"- **Preview**: {issue.body_preview}")
            lines.append("")
    else:
        lines.append("No issues flagged for review.")
        lines.append("")

    # Section 2: All open issues (quick browse)
    lines.append("### All Open Issues")
    lines.append("")
    lines.append("Quick browse - check titles and flag any that look important:")
    lines.append("")

    if open_issues:
        lines.append("| # | Title | Labels | Date | Engagement |")
        lines.append("|---|-------|--------|------|------------|")
        for issue in open_issues:
            labels_str = ", ".join(issue.labels[:3]) if issue.labels else "-"
            engagement = f"{issue.reactions_count}👍 {issue.comments_count}💬"
            # Escape pipe characters in title
            safe_title = issue.title.replace("|", "\\|")
            if len(safe_title) > 60:
                safe_title = safe_title[:57] + "..."
            lines.append(
                f"| [#{issue.number}]({issue.url}) | {safe_title} | {labels_str} | {issue.created_at} | {engagement} |"
            )
        lines.append("")
    else:
        lines.append("No open issues found.")
        lines.append("")

    # Section 3: Recently closed issues (might be relevant)
    lines.append("### Recently Closed Issues")
    lines.append("")
    lines.append("These issues were closed but might still be relevant for release notes:")
    lines.append("")

    if closed_issues:
        # Sort by engagement and show top ones
        closed_issues.sort(key=lambda i: -(i.reactions_count + i.comments_count))
        top_closed = closed_issues[:20]

        lines.append("| # | Title | Labels | Date | Engagement |")
        lines.append("|---|-------|--------|------|------------|")
        for issue in top_closed:
            labels_str = ", ".join(issue.labels[:3]) if issue.labels else "-"
            engagement = f"{issue.reactions_count}👍 {issue.comments_count}💬"
            safe_title = issue.title.replace("|", "\\|")
            if len(safe_title) > 60:
                safe_title = safe_title[:57] + "..."
            lines.append(
                f"| [#{issue.number}]({issue.url}) | {safe_title} | {labels_str} | {issue.created_at} | {engagement} |"
            )

        if len(closed_issues) > 20:
            lines.append("")
            lines.append(f"*... and {len(closed_issues) - 20} more closed issues*")
        lines.append("")
    else:
        lines.append("No closed issues found.")
        lines.append("")

    # Summary for checklist
    lines.append("### Summary for Release Checklist")
    lines.append("")
    lines.append("After reviewing the issues above, add important ones to the checklist:")
    lines.append("")
    lines.append("```markdown")
    lines.append("### Bug need Solve")
    lines.append("")
    # Pre-fill with flagged open issues
    flagged_open = [i for i in flagged if i.state == "OPEN"]
    if flagged_open:
        for issue in flagged_open[:10]:
            lines.append(f"- [ ] #{issue.number} - {issue.title[:50]}")
    else:
        lines.append("- [ ] (Add issues after review)")
    lines.append("```")
    lines.append("")

    # Statistics
    lines.append("### Statistics")
    lines.append("")
    lines.append(f"- Total issues scanned: {len(issues)}")
    lines.append(f"- Open issues: {len(open_issues)}")
    lines.append(f"- Closed issues: {len(closed_issues)}")
    lines.append(f"- Flagged for review: {len(flagged)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan GitHub issues since the last release")
    parser.add_argument("--repo", default="vllm-project/vllm-ascend", help="Repository")
    parser.add_argument("--since-tag", required=True, help="Previous release tag (including rc)")
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"], help="Issue state filter")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    # Get release date
    print(f"Getting release date for {args.since_tag}...")
    since_date = get_release_date(args.repo, args.since_tag)

    if not since_date:
        print(f"Warning: Could not get release date for {args.since_tag}")
        print("Please enter the date manually (YYYY-MM-DD):")
        since_date = input().strip()

    print(f"Scanning issues since {since_date}...")

    # Fetch issues
    print(f"Fetching issues from {args.repo}...")
    raw_issues = fetch_issues_since(args.repo, since_date, args.state)
    print(f"Found {len(raw_issues)} issues")

    # Parse issues
    issues = [parse_issue(issue) for issue in raw_issues]

    # Sort: flagged first, then by date descending
    issues.sort(key=lambda i: (not i.needs_review, i.created_at), reverse=True)

    # Generate report
    print("Generating report...")
    report = generate_report(issues, args.repo, args.since_tag, since_date)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")

    # Print summary
    flagged = [i for i in issues if i.needs_review]
    open_issues = [i for i in issues if i.state == "OPEN"]

    print("\nSummary:")
    print(f"  Total issues: {len(issues)}")
    print(f"  Open issues: {len(open_issues)}")
    print(f"  Flagged for review: {len(flagged)}")
    print("\nPlease review the report and identify issues that need to be fixed before release.")

    return 0


if __name__ == "__main__":
    exit(main())
