#!/usr/bin/env python3
"""Collect issues from vllm-project/vllm-ascend and save to CSV.

Uses state-based stratified sampling to ensure a balanced mix of open, solved,
and closed issues. Fetches pages until enough of each state are collected, then
evenly samples within each group.

Usage:
    python collect_issues.py --output issues.csv
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

import requests

REPO = "vllm-project/vllm-ascend"
PER_PAGE = 100
MAX_PAGES = 20

# Target count per state
STATE_TARGETS = {"open": 35, "solved": 35, "closed": 30}


def fetch_issues(page: int) -> list[dict]:
    url = f"https://api.github.com/repos/{REPO}/issues"
    params = {"state": "all", "per_page": PER_PAGE, "page": page,
              "sort": "created", "direction": "desc"}
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code == 422:
        return []
    resp.raise_for_status()
    return resp.json()


def extract_prefix(title: str) -> str:
    """Extract the issue type prefix like [Bug], [Doc], etc."""
    known = ["[Bug]", "[Installation]", "[Usage]", "[Doc]", "[Misc]",
             "[Feature]", "[RFC]", "[CI]", "[Performance]", "[BugFix]",
             "[Test]", "[Ops]", "[RoPE]", "[Ascend950]", "[WIP]",
             "[Community]", "[0.21.0]"]
    for prefix in known:
        if title.startswith(prefix):
            return prefix
    m = re.match(r'^\[([^\]]+)\]', title)
    if m:
        return f"[{m.group(1)}]"
    return "[Misc]"


def issue_state(issue: dict) -> str:
    """Return state: open, solved, or closed."""
    if issue.get("state") == "open":
        return "open"
    if issue.get("state_reason") == "completed":
        return "solved"
    return "closed"


def average_sample(items: list[dict], count: int) -> list[dict]:
    """Pick `count` items evenly spaced across the list."""
    if count >= len(items):
        return items[:]
    step = (len(items) - 1) / (count - 1) if count > 1 else 0
    return [items[round(i * step)] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect issues to CSV")
    parser.add_argument("--output", default="issues.csv", help="Output CSV file")
    args = parser.parse_args()

    # Fetch pages until we have enough of each state
    all_issues: list[dict] = []
    by_state: dict[str, list[dict]] = {"open": [], "solved": [], "closed": []}
    page = 1
    while page <= MAX_PAGES:
        print(f"Fetching page {page}...")
        raw = fetch_issues(page)
        if not raw:
            break
        issues = [i for i in raw if "pull_request" not in i]
        all_issues.extend(issues)
        for iss in issues:
            by_state[issue_state(iss)].append(iss)
        counts = {s: len(by_state[s]) for s in by_state}
        print(f"  page {page}: {len(issues)} issues, "
              f"open={counts['open']} solved={counts['solved']} closed={counts['closed']}")
        # Stop if we have at least the target for each state
        if all(counts[s] >= STATE_TARGETS[s] for s in STATE_TARGETS):
            break
        page += 1
        if len(raw) < PER_PAGE:
            break
        time.sleep(0.5)

    print(f"Fetched {len(all_issues)} issues total "
          f"(open={len(by_state['open'])} solved={len(by_state['solved'])} closed={len(by_state['closed'])})")

    # State-based stratified sampling
    selected: list[dict] = []
    for state, target in STATE_TARGETS.items():
        pool = by_state[state]
        count = min(target, len(pool))
        picked = average_sample(pool, count)
        selected.extend(picked)
        print(f"  State '{state}': pool={len(pool)}, picked={len(picked)}")

    print(f"Total selected: {len(selected)}")

    # Distribution summary
    by_prefix: dict[str, int] = {}
    final_by_state: dict[str, int] = {}
    for issue in selected:
        prefix = extract_prefix(issue["title"])
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        s = issue_state(issue)
        final_by_state[s] = final_by_state.get(s, 0) + 1
    print("Distribution by type:")
    for prefix, n in sorted(by_prefix.items()):
        print(f"  {prefix}: {n}")
    print("Distribution by state:")
    for state, n in sorted(final_by_state.items()):
        print(f"  {state}: {n}")

    # Write CSV
    fieldnames = ["issue_number", "title", "prefix", "state", "labels",
                  "created_at", "body", "deepseek_v4_flash_output",
                  "expected_ok_dsv4"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for issue in selected:
            body = (issue.get("body") or "").replace("\r\n", "\n")
            labels = ",".join(label["name"] for label in issue.get("labels", []))
            writer.writerow({
                "issue_number": issue["number"],
                "title": issue["title"],
                "prefix": extract_prefix(issue["title"]),
                "state": issue_state(issue),
                "labels": labels,
                "created_at": issue["created_at"],
                "body": body,
                "deepseek_v4_flash_output": "",
                "expected_ok_dsv4": "",
            })

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
