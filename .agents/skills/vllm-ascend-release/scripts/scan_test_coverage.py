#!/usr/bin/env python3
"""
Scan for features and new models that need manual testing.

This script identifies:
1. New features/models merged since the last release that may lack CI test coverage
2. Outstanding issues from the previous release's feedback issue

These are typically features that:
- Don't have corresponding test cases in CI
- Were merged without tests due to environment constraints
- New model support that needs validation

Usage:
    python scan_test_coverage.py \
        --repo vllm-project/vllm-ascend \
        --since-tag v0.15.0rc1 \
        --feedback-issue 1234 \
        --output test-coverage-analysis.md
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FeatureItem:
    """A feature or model that may need testing."""

    pr_number: int
    title: str
    url: str
    merged_at: str
    author: str
    labels: list[str] = field(default_factory=list)
    category: str = ""  # "model", "feature", "optimization", etc.
    has_tests: bool = False
    test_files: list[str] = field(default_factory=list)
    reason: str = ""  # Why it needs testing


@dataclass
class FeedbackItem:
    """An issue from the feedback thread."""

    comment_id: int
    author: str
    body: str
    created_at: str
    linked_issue: int | None = None  # Linked issue number if any
    status: str = "unknown"  # "resolved", "unresolved", "unclear"
    resolution_note: str = ""


# Keywords that indicate new model support
MODEL_KEYWORDS = [
    "support",
    "add",
    "enable",
    "model",
    "llm",
    "vlm",
    "moe",
    "mtp",
    "qwen",
    "deepseek",
    "llama",
    "glm",
    "kimi",
    "mixtral",
    "phi",
    "gemma",
    "internlm",
    "baichuan",
    "yi",
    "chatglm",
]

# Keywords that indicate features needing testing
FEATURE_KEYWORDS = [
    "graph mode",
    "expert parallel",
    "tensor parallel",
    "pipeline parallel",
    "multimodal",
    "speculative",
    "chunked prefill",
    "prefix caching",
    "quantization",
    "awq",
    "gptq",
    "fp8",
    "w8a8",
    "performance",
    "optimization",
]

# Labels that indicate new features
FEATURE_LABELS = [
    "enhancement",
    "feature",
    "new model",
    "new-model",
    "model support",
]


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
        return data.get("publishedAt", "")
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
            # Get the commit date
            sha = data.get("object", {}).get("sha", "")
            if sha:
                commit_output = run_gh_command(
                    [
                        "api",
                        f"repos/{repo}/commits/{sha}",
                    ]
                )
                commit_data = json.loads(commit_output)
                return commit_data.get("commit", {}).get("committer", {}).get("date", "")
        except Exception:
            pass
        return ""


def fetch_merged_prs(repo: str, since_date: str) -> list[dict]:
    """Fetch PRs merged since a given date."""
    output = run_gh_command(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--search",
            f"merged:>{since_date}",
            "--limit",
            "200",
            "--json",
            "number,title,url,mergedAt,author,labels,files",
        ]
    )
    return json.loads(output)


def check_pr_has_tests(pr: dict) -> tuple[bool, list[str]]:
    """Check if a PR includes test files."""
    test_files = []
    files = pr.get("files", [])

    for file in files:
        path = file.get("path", "")
        if "test" in path.lower() and path.endswith(".py"):
            test_files.append(path)

    return len(test_files) > 0, test_files


def categorize_pr(pr: dict) -> str:
    """Categorize a PR based on its content."""
    title_lower = pr["title"].lower()
    labels = [label["name"].lower() for label in pr.get("labels", [])]

    # Check for model support
    for keyword in ["model", "support", "add"]:
        if keyword in title_lower:
            for model in [
                "qwen",
                "deepseek",
                "llama",
                "glm",
                "kimi",
                "mixtral",
                "phi",
                "gemma",
                "internlm",
                "baichuan",
                "yi",
                "chatglm",
            ]:
                if model in title_lower:
                    return "model"

    # Check labels
    for label in labels:
        if "model" in label:
            return "model"
        if "feature" in label or "enhancement" in label:
            return "feature"

    # Check for specific features
    for feature in [
        "graph",
        "parallel",
        "multimodal",
        "speculative",
        "chunked",
        "prefix",
        "quantization",
        "performance",
    ]:
        if feature in title_lower:
            return "feature"

    if "fix" in title_lower or "bug" in title_lower:
        return "bugfix"

    if "doc" in title_lower or "readme" in title_lower:
        return "docs"

    if "ci" in title_lower or "test" in title_lower:
        return "ci"

    return "other"


def analyze_pr(pr: dict) -> FeatureItem | None:
    """Analyze a PR to determine if it needs testing."""
    category = categorize_pr(pr)

    # Skip docs, CI changes, and bug fixes (they should be tested via CI)
    if category in ["docs", "ci", "bugfix", "other"]:
        return None

    has_tests, test_files = check_pr_has_tests(pr)

    # If it's a feature/model and has tests, it's likely covered by CI
    # But we still report it for visibility
    item = FeatureItem(
        pr_number=pr["number"],
        title=pr["title"],
        url=pr["url"],
        merged_at=pr.get("mergedAt", ""),
        author=pr.get("author", {}).get("login", "unknown"),
        labels=[label["name"] for label in pr.get("labels", [])],
        category=category,
        has_tests=has_tests,
        test_files=test_files,
    )

    if not has_tests:
        item.reason = "No test files included in PR"
    else:
        item.reason = f"Has tests: {', '.join(test_files[:3])}"

    return item


def fetch_feedback_comments(repo: str, issue_number: int) -> list[dict]:
    """Fetch comments from a feedback issue."""
    output = run_gh_command(
        [
            "issue",
            "view",
            str(issue_number),
            "--repo",
            repo,
            "--json",
            "comments",
        ]
    )
    data = json.loads(output)
    return data.get("comments", [])


def analyze_feedback_comment(comment: dict, repo: str) -> FeedbackItem | None:
    """Analyze a feedback comment to extract issues."""
    body = comment.get("body", "")

    # Skip bot comments and simple reactions
    author = comment.get("author", {}).get("login", "")
    if "bot" in author.lower():
        return None

    # Skip very short comments (likely reactions)
    if len(body.strip()) < 50:
        return None

    # Try to extract linked issue numbers
    issue_matches = re.findall(r"#(\d+)", body)
    linked_issue = int(issue_matches[0]) if issue_matches else None

    item = FeedbackItem(
        comment_id=comment.get("id", 0),
        author=author,
        body=body[:500] + "..." if len(body) > 500 else body,
        created_at=comment.get("createdAt", ""),
        linked_issue=linked_issue,
    )

    # Check if there's a linked issue and if it's resolved
    if linked_issue:
        try:
            issue_output = run_gh_command(
                [
                    "issue",
                    "view",
                    str(linked_issue),
                    "--repo",
                    repo,
                    "--json",
                    "state,title",
                ]
            )
            issue_data = json.loads(issue_output)
            if issue_data.get("state") == "CLOSED":
                item.status = "resolved"
                item.resolution_note = f"Issue #{linked_issue} is closed"
            else:
                item.status = "unresolved"
                item.resolution_note = f"Issue #{linked_issue} is still open"
        except Exception:
            item.status = "unknown"
            item.resolution_note = f"Could not fetch issue #{linked_issue}"
    else:
        # No linked issue, mark as needing review
        item.status = "needs_review"
        item.resolution_note = "No linked issue found, needs manual review"

    return item


def generate_report(
    features: list[FeatureItem],
    feedback: list[FeedbackItem],
    repo: str,
    since_tag: str,
) -> str:
    """Generate a markdown report."""
    lines = [
        "## Test Coverage Analysis",
        "",
        f"Repository: {repo}",
        f"Since: {since_tag}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Section 1: Features/Models needing tests
    lines.append("### Features/Models Needing Manual Testing")
    lines.append("")
    lines.append("The following features/models were merged without test coverage in CI:")
    lines.append("")

    # Filter to only those without tests
    needs_testing = [f for f in features if not f.has_tests]

    if needs_testing:
        # Group by category
        models = [f for f in needs_testing if f.category == "model"]
        other_features = [f for f in needs_testing if f.category == "feature"]

        if models:
            lines.append("#### New Models")
            lines.append("")
            for f in models:
                lines.append(f"- [ ] **{f.title}**")
                lines.append(f"  - PR: #{f.pr_number}")
                lines.append(f"  - Author: @{f.author}")
                lines.append(f"  - Reason: {f.reason}")
                lines.append("")

        if other_features:
            lines.append("#### New Features")
            lines.append("")
            for f in other_features:
                lines.append(f"- [ ] **{f.title}**")
                lines.append(f"  - PR: #{f.pr_number}")
                lines.append(f"  - Author: @{f.author}")
                lines.append(f"  - Reason: {f.reason}")
                lines.append("")
    else:
        lines.append("All merged features/models appear to have test coverage.")
        lines.append("")

    # Section 2: Features with tests (for reference)
    has_tests = [f for f in features if f.has_tests]
    if has_tests:
        lines.append("### Features With Test Coverage (Reference)")
        lines.append("")
        lines.append("These features have tests and should be covered by CI:")
        lines.append("")
        for f in has_tests[:10]:  # Limit to 10
            lines.append(f"- {f.title} (#{f.pr_number})")
        if len(has_tests) > 10:
            lines.append(f"- ... and {len(has_tests) - 10} more")
        lines.append("")

    # Section 3: Feedback issue status
    lines.append("### Previous Release Feedback Status")
    lines.append("")

    if feedback:
        unresolved = [f for f in feedback if f.status == "unresolved"]
        needs_review = [f for f in feedback if f.status == "needs_review"]
        resolved = [f for f in feedback if f.status == "resolved"]

        if unresolved:
            lines.append("#### Unresolved Issues")
            lines.append("")
            for f in unresolved:
                lines.append(f"- [ ] Comment by @{f.author}")
                if f.linked_issue:
                    lines.append(f"  - Linked issue: #{f.linked_issue} (still open)")
                lines.append(f"  - Preview: {f.body[:200]}...")
                lines.append("")

        if needs_review:
            lines.append("#### Needs Manual Review")
            lines.append("")
            for f in needs_review:
                lines.append(f"- [ ] Comment by @{f.author}")
                lines.append(f"  - {f.resolution_note}")
                lines.append(f"  - Preview: {f.body[:200]}...")
                lines.append("")

        if resolved:
            lines.append("#### Resolved")
            lines.append("")
            for f in resolved[:5]:
                lines.append(f"- [x] Comment by @{f.author} - {f.resolution_note}")
            if len(resolved) > 5:
                lines.append(f"- ... and {len(resolved) - 5} more resolved items")
            lines.append("")
    else:
        lines.append("No feedback comments found or no feedback issue specified.")
        lines.append("")

    # Summary for checklist
    lines.append("### Summary for Release Checklist")
    lines.append("")
    lines.append("Copy the following to the 'Functional Test' section:")
    lines.append("")
    lines.append("```markdown")
    lines.append("#### Manual Testing Required")
    lines.append("")
    for f in needs_testing[:10]:
        lines.append(f"- [ ] {f.title} (#{f.pr_number})")
    if len(needs_testing) > 10:
        lines.append(f"- [ ] ... and {len(needs_testing) - 10} more items")
    lines.append("")
    lines.append("#### Feedback Issues to Address")
    lines.append("")
    unresolved_count = len([f for f in feedback if f.status in ["unresolved", "needs_review"]])
    if unresolved_count > 0:
        lines.append(f"- [ ] {unresolved_count} unresolved feedback items")
    else:
        lines.append("- [x] All feedback items addressed")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan for features/models needing manual testing")
    parser.add_argument("--repo", default="vllm-project/vllm-ascend", help="Repository")
    parser.add_argument("--since-tag", required=True, help="Previous release tag")
    parser.add_argument("--feedback-issue", type=int, help="Previous release feedback issue number")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    # Get release date
    print(f"Getting release date for {args.since_tag}...")
    release_date = get_release_date(args.repo, args.since_tag)
    if not release_date:
        print(f"Warning: Could not get release date for {args.since_tag}")
        # Use a default lookback
        release_date = "2024-01-01"
    else:
        # Extract just the date part
        release_date = release_date[:10]

    print(f"Fetching PRs merged since {release_date}...")
    prs = fetch_merged_prs(args.repo, release_date)
    print(f"Found {len(prs)} merged PRs")

    # Analyze PRs
    print("Analyzing PRs for test coverage...")
    features = []
    for pr in prs:
        item = analyze_pr(pr)
        if item:
            features.append(item)

    print(f"Found {len(features)} features/models")
    print(f"  - With tests: {len([f for f in features if f.has_tests])}")
    print(f"  - Without tests: {len([f for f in features if not f.has_tests])}")

    # Fetch feedback if provided
    feedback = []
    if args.feedback_issue:
        print(f"Fetching feedback from issue #{args.feedback_issue}...")
        comments = fetch_feedback_comments(args.repo, args.feedback_issue)
        print(f"Found {len(comments)} comments")

        for comment in comments:
            item = analyze_feedback_comment(comment, args.repo)
            if item:
                feedback.append(item)

        print(f"Analyzed {len(feedback)} feedback items")

    # Generate report
    print("Generating report...")
    report = generate_report(features, feedback, args.repo, args.since_tag)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")

    # Print summary
    needs_testing = len([f for f in features if not f.has_tests])
    unresolved_feedback = len([f for f in feedback if f.status in ["unresolved", "needs_review"]])

    print("\nSummary:")
    print(f"  Features/models needing manual testing: {needs_testing}")
    print(f"  Unresolved feedback items: {unresolved_feedback}")

    return 0


if __name__ == "__main__":
    exit(main())
