#!/usr/bin/env python3
"""Collect PRs from vllm-project/vllm-ascend and save to CSV.

Fetches both open and merged PRs, applies stratified sampling across time
ranges to produce a representative set for LLM evaluation.

Stratified sampling:
  - Fetch up to 400 PRs (pages 1-4, newest first)
  - Stratum 1 (indices 0-99):   select 40  (heavy weight on recent)
  - Stratum 2 (indices 100-199): select 10
  - Stratum 3 (indices 200-399): select 10
  - Total target: 60 PRs (capped at actual fetched count)

Usage:
    python collect_prs.py --output prs.csv
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
    (0,   100, 40),
    (100, 200, 10),
    (200, 400, 10),
]

KNOWN_PREFIXES = [
    "[Feat]", "[BugFix]", "[Bug]", "[Doc]", "[CI]", "[Refactor]",
    "[Perf]", "[Test]", "[Chore]", "[Platform]", "[Misc]",
    "[Installation]", "[Usage]", "[Feature]", "[RFC]", "[Community]",
]


def extract_prefix(title: str) -> str:
    title_lower = title.lower()
    for prefix in KNOWN_PREFIXES:
        if title_lower.startswith(prefix.lower()):
            return prefix
    m = re.match(r'^\[([^\]]+)\]', title)
    if m:
        # Try to match known prefix case-insensitively
        inner = m.group(1)
        inner_lower = inner.lower()
        for known in KNOWN_PREFIXES:
            if known.lower() == f"[{inner_lower}]":
                return known
        return f"[{inner}]"
    return "[Misc]"


def fetch_prs(page: int) -> list[dict]:
    url = f"https://api.github.com/repos/{REPO}/pulls"
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


def pr_state(pr: dict) -> str:
    if pr.get("state") == "open":
        return "open"
    if pr.get("merged_at") is not None:
        return "merged"
    return "closed"


def average_sample(items: list[dict], count: int) -> list[dict]:
    if count >= len(items):
        return items[:]
    step = (len(items) - 1) / (count - 1) if count > 1 else 0
    return [items[round(i * step)] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PRs to CSV")
    parser.add_argument("--output", default="prs.csv", help="Output CSV file")
    args = parser.parse_args()

    total_needed = max(end for _, end, _ in STRATA)
    all_prs: list[dict] = []
    page = 1
    while len(all_prs) < total_needed:
        print(f"Fetching page {page}...")
        raw = fetch_prs(page)
        if not raw:
            break
        all_prs.extend(raw)
        n_open = sum(1 for p in raw if p.get("state") == "open")
        n_merged = sum(1 for p in raw if p.get("merged_at") is not None)
        n_closed = sum(1 for p in raw if p.get("state") == "closed" and p.get("merged_at") is None)
        print(f"  page {page}: {len(raw)} total (open={n_open} merged={n_merged} closed={n_closed}),"
              f" accumulated {len(all_prs)}/{total_needed}")
        page += 1
        if len(raw) < PER_PAGE:
            break
        time.sleep(0.5)

    print(f"Fetched {len(all_prs)} PRs total")

    # Stratified average sampling
    selected: list[dict] = []
    for start, end, count in STRATA:
        pool = all_prs[start:min(end, len(all_prs))]
        picked = average_sample(pool, count)
        selected.extend(picked)
        print(f"  Stratum [{start}:{end}]: pool={len(pool)}, picked={len(picked)}")

    print(f"Total selected: {len(selected)}")

    # Distribution summary
    by_prefix: dict[str, int] = {}
    by_state: dict[str, int] = {}
    for pr in selected:
        prefix = extract_prefix(pr["title"])
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        by_state[pr_state(pr)] = by_state.get(pr_state(pr), 0) + 1
    print("Distribution by type:")
    for prefix, n in sorted(by_prefix.items()):
        print(f"  {prefix}: {n}")
    print("Distribution by state:")
    for state, n in sorted(by_state.items()):
        print(f"  {state}: {n}")

    # Write CSV
    fieldnames = ["pr_number", "title", "prefix", "state", "labels",
                  "created_at", "body", "deepseek_v4_flash_output",
                  "expected_ok_dsv4"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pr in selected:
            body = (pr.get("body") or "").replace("\r\n", "\n")
            labels = ",".join(label["name"] for label in pr.get("labels", []))
            writer.writerow({
                "pr_number": pr["number"],
                "title": pr["title"],
                "prefix": extract_prefix(pr["title"]),
                "state": pr_state(pr),
                "labels": labels,
                "created_at": pr["created_at"],
                "body": body,
                "deepseek_v4_flash_output": "",
                "expected_ok_dsv4": "",
            })

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
