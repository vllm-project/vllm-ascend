#!/usr/bin/env python3
"""Collect issues from vllm-project/vllm-ascend and save to CSV.

Stratified sampling across time ranges:
  - Fetch 400 issues (pages 1-4, newest first)
  - Stratum 1 (indices 0-99):   select 80  (heavy weight on recent)
  - Stratum 2 (indices 100-199): select 10
  - Stratum 3 (indices 200-399): select 10
  - Total: 100 issues

Within each stratum, issues are picked at regular intervals (average/even
sampling) to avoid bias toward any contiguous block.

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

# Sampling config: (start_index, end_index, count_to_pick)
STRATA = [
    (0,   100, 80),   # newest 100 → pick 80
    (100, 200, 10),   # next 100   → pick 10
    (200, 400, 10),   # older 200  → pick 10
]


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
        return []  # GitHub API limit: max 1000 results
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

    # Fetch until we have enough actual issues (API mixes PRs with issues)
    total_needed = max(end for _, end, _ in STRATA)
    all_issues = []
    page = 1
    while len(all_issues) < total_needed:
        print(f"Fetching page {page}...")
        raw = fetch_issues(page)
        if not raw:
            break
        filtered = [i for i in raw if "pull_request" not in i]
        all_issues.extend(filtered)
        print(f"  page {page}: {len(raw)} total, {len(filtered)} issues, accumulated {len(all_issues)}/{total_needed}")
        page += 1
        if len(raw) < PER_PAGE:
            break
        time.sleep(0.5)

    print(f"Fetched {len(all_issues)} issues total")

    # Stratified average sampling
    selected: list[dict] = []
    for start, end, count in STRATA:
        pool = all_issues[start:min(end, len(all_issues))]
        picked = average_sample(pool, count)
        selected.extend(picked)
        print(f"  Stratum [{start}:{end}]: pool={len(pool)}, picked={len(picked)}")

    print(f"Total selected: {len(selected)}")

    # Distribution summary
    by_prefix: dict[str, int] = {}
    for issue in selected:
        prefix = extract_prefix(issue["title"])
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
    print("Distribution by type:")
    for prefix, n in sorted(by_prefix.items()):
        print(f"  {prefix}: {n}")

    # Write CSV
    fieldnames = ["issue_number", "title", "prefix", "state", "labels",
                  "created_at", "body", "deepseek_v4_flash_output"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for issue in selected:
            body = (issue.get("body") or "").replace("\r\n", "\n")
            labels = ",".join(l["name"] for l in issue.get("labels", []))
            writer.writerow({
                "issue_number": issue["number"],
                "title": issue["title"],
                "prefix": extract_prefix(issue["title"]),
                "state": issue["state"],
                "labels": labels,
                "created_at": issue["created_at"],
                "body": body,
                "deepseek_v4_flash_output": "",
            })

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
