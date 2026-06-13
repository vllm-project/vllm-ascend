#!/usr/bin/env python3
"""Test script: read issues CSV, call DeepSeek V4 Flash for each issue,
and write the LLM output back to the CSV's last column.

Usage:
    python .github/workflows/scripts/robot/tests/test_issues_csv.py --input issues.csv --output results.csv

Requires VLLM_BASE_URL and VLLM_API_KEY environment variables.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Add parent to path so we can import the step modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_system_prompt import load_system_prompt
from prepare_template import load_template


def call_vllm(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def build_user_prompt(row: dict) -> str:
    template_text = load_template(row["prefix"])
    return f"""## Issue Title
{row['title']}

## Issue Template (what the user was asked to fill in)
{template_text}

## Issue Content (what the user actually submitted)
{row['body']}

Please review this issue and provide your feedback in the specified format.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test issues CSV with DeepSeek V4 Flash")
    parser.add_argument("--input", default="issues.csv", help="Input CSV file")
    parser.add_argument("--output", default="results.csv", help="Output CSV file")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="Max issues to process (0 = all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # Read all rows
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} issues from {input_path}")

    system_prompt = load_system_prompt()
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"vLLM endpoint: {VLLM_BASE_URL}")
    print(f"Model: deepseek-v4-flash")

    # Determine range
    end = total if args.limit == 0 else min(args.start + args.limit, total)
    print(f"Processing issues {args.start} to {end - 1} ({end - args.start} total)")

    success = 0
    failed = 0

    for i in range(args.start, end):
        row = rows[i]
        issue_num = row["issue_number"]
        title = row["title"]
        print(f"\n[{i + 1}/{total}] #{issue_num}: {title[:80]}...")

        try:
            user_prompt = build_user_prompt(row)
            output = call_vllm(system_prompt, user_prompt)
            row["deepseek_v4_flash_output"] = output
            success += 1
            print(f"  OK ({len(output)} chars)")
        except Exception as e:
            row["deepseek_v4_flash_output"] = f"ERROR: {e}"
            failed += 1
            print(f"  FAILED: {e}")

        # Save progress after each issue
        with open(args.output, "w", newline="") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        time.sleep(0.1)  # small delay to avoid overwhelming the server

    print(f"\n{'=' * 60}")
    print(f"Done. Success: {success}, Failed: {failed}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
