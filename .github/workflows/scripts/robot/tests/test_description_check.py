#!/usr/bin/env python3
"""End-to-end test for the Issue Description Completeness Check bot.

Supports two modes:
  1. Default: load test cases from CSV, create issues, verify results
  2. --demo: run the 3 built-in demo tests (no CSV needed)

Usage:
    # Run with default CSV
    export GITHUB_TOKEN=$(gh auth token)
    python test_description_check.py --repo ai-infra-develop/vllm-ascend

    # Run with custom CSV
    python test_description_check.py --repo owner/repo --csv my_cases.csv

    # Run 3 built-in demo tests (no CSV required)
    python test_description_check.py --repo owner/repo --demo
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

LABEL_NAME = "need-detail-desc"
POLL_INTERVAL = 15
MAX_WAIT = 180
DEFAULT_CSV = Path(__file__).resolve().parent / "description_check_cases.csv"
MAX_RETRIES = 3
RETRY_DELAY = 5


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = ""
        self.details = ""

    def fail(self, msg: str):
        self.error = msg

    def ok(self, msg: str = ""):
        self.passed = True
        self.details = msg


def api_request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with retry on network errors."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=HEADERS, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.HTTPError:
            raise
    raise last_error


def api_get(url: str) -> dict:
    return api_request("GET", url).json()


def api_post(url: str, data: dict) -> dict:
    return api_request("POST", url, json=data).json()


def api_patch(url: str, data: dict) -> dict:
    return api_request("PATCH", url, json=data).json()


def api_delete(url: str) -> None:
    api_request("DELETE", url)


def api_close_issue(api_base: str, issue_number: int):
    api_patch(f"{api_base}/issues/{issue_number}", {"state": "closed"})


def create_issue(api_base: str, title: str, body: str) -> int:
    data = api_post(f"{api_base}/issues", {"title": title, "body": body})
    return data["number"]


def edit_issue(api_base: str, issue_number: int, title: str, body: str):
    api_patch(f"{api_base}/issues/{issue_number}", {"title": title, "body": body})


def get_labels(api_base: str, issue_number: int) -> list[str]:
    data = api_get(f"{api_base}/issues/{issue_number}")
    return [lb["name"] for lb in data.get("labels", [])]


def get_comments(api_base: str, issue_number: int) -> list[dict]:
    return api_get(f"{api_base}/issues/{issue_number}/comments")


def get_bot_comments(api_base: str, issue_number: int) -> list[dict]:
    comments = get_comments(api_base, issue_number)
    return [c for c in comments if c.get("user", {}).get("login", "").endswith("[bot]")]


def remove_label(api_base: str, issue_number: int, label: str):
    try:
        api_delete(f"{api_base}/issues/{issue_number}/labels/{label}")
    except requests.HTTPError:
        pass


def wait_for_bot_comment(api_base: str, issue_number: int, max_wait: int = MAX_WAIT) -> tuple[list[str], list[dict]]:
    """Poll until bot posts a comment. Returns (labels, bot_comments)."""
    start = time.time()
    while time.time() - start < max_wait:
        labels = get_labels(api_base, issue_number)
        bot_comments = get_bot_comments(api_base, issue_number)
        if bot_comments:
            return labels, bot_comments
        elapsed = int(time.time() - start)
        print(f"  Waiting for bot comment... ({elapsed}s elapsed)")
        time.sleep(POLL_INTERVAL)
    return get_labels(api_base, issue_number), get_bot_comments(api_base, issue_number)


def wait_then_check(api_base: str, issue_number: int, wait_sec: int = 90) -> tuple[list[str], list[dict]]:
    """Wait a fixed time, then check labels and comments."""
    for remaining in range(wait_sec, 0, -POLL_INTERVAL):
        print(f"  Waiting for bot to process... ({wait_sec - remaining}s elapsed)")
        time.sleep(min(POLL_INTERVAL, remaining))
    labels = get_labels(api_base, issue_number)
    bot_comments = get_bot_comments(api_base, issue_number)
    return labels, bot_comments


def load_csv_cases(csv_path: str) -> list[dict]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["expect_ok"] = row.get("expect_ok", "").strip().lower() == "true"
            row["expect_label"] = row.get("expect_label", "").strip().lower() == "true"
            row["expect_comment"] = row.get("expect_comment", "").strip().lower() == "true"
            row["body"] = (row.get("body") or "").replace("\\n", "\n")
            cases.append(row)
    return cases


def verify_case(api_base: str, case: dict) -> Result:
    title = case["title"]
    body = case.get("body", "")
    expect_ok = case.get("expect_ok", True)
    expect_label = case.get("expect_label", False)
    expect_comment = case.get("expect_comment", False)
    notes = case.get("notes", "")

    r = Result(f"{title} ({notes})" if notes else title)
    issue_number = None

    try:
        issue_number = create_issue(api_base, title, body)
        print(f"  Created issue #{issue_number}")

        if expect_comment:
            labels, bot_comments = wait_for_bot_comment(api_base, issue_number)
        else:
            labels, bot_comments = wait_then_check(api_base, issue_number)

        has_label = LABEL_NAME in labels
        has_comment = len(bot_comments) > 0

        checks = []
        if has_label != expect_label:
            checks.append(f"label: expected={expect_label} got={has_label} (labels={labels})")
        if has_comment != expect_comment:
            checks.append(f"comment: expected={expect_comment} got={has_comment}")

        if has_comment and not expect_comment:
            body_text = bot_comments[0]["body"][:200]
            checks.append(f"unexpected comment: {body_text}")

        if checks:
            r.fail("; ".join(checks))
        else:
            r.ok(f"#{issue_number} label={has_label} comment={has_comment}")

        return r
    except Exception as e:
        r.fail(str(e))
        return r
    finally:
        if issue_number:
            try:
                api_close_issue(api_base, issue_number)
            except Exception:
                pass


def run_csv_tests(api_base: str, csv_path: str):
    cases = load_csv_cases(csv_path)
    print(f"Loaded {len(cases)} test cases from {csv_path}")
    print()

    results = []
    for i, case in enumerate(cases):
        label = f"[{i + 1}/{len(cases)}]"
        print(f"{label} {case['title']}")
        if case.get("notes"):
            print(f"     Notes: {case['notes']}")
        result = verify_case(api_base, case)
        results.append(result)
        status = "PASS" if result.passed else f"FAIL: {result.error}"
        print(f"     {status}")
        print()

    return results


def run_demo_tests(api_base: str) -> list[Result]:
    """3 built-in demo tests."""

    def test_incomplete_bug() -> Result:
        r = Result("Demo: Incomplete bug → label + comment")
        issue_number = None
        try:
            issue_number = create_issue(api_base, "[Bug]: 调用接口报错", "有个错误，帮忙看看")
            print(f"  Created issue #{issue_number}")
            labels, comments = wait_for_bot_comment(api_base, issue_number)
            if LABEL_NAME not in labels:
                r.fail(f"Expected label '{LABEL_NAME}' not found. Labels: {labels}")
            elif not comments:
                r.fail("Expected bot comment but none found")
            else:
                r.ok(f"Issue #{issue_number}: label added, comment posted")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if issue_number:
                try:
                    api_close_issue(api_base, issue_number)
                except Exception:
                    pass

    def test_complete_bug() -> Result:
        r = Result("Demo: Complete bug → no label, no comment")
        issue_number = None
        try:
            issue_number = create_issue(
                api_base,
                "[Bug]: A5 CANN 8.2 conv2d 算子报错",
                "### 环境信息\n- Ubuntu 22.04\n- A5\n- CANN 8.2.0\n### 问题描述\nconv2d RuntimeError: conv2d forward failed\n复现: shape=(1,3,224,224)",
            )
            print(f"  Created issue #{issue_number}")
            labels, comments = wait_then_check(api_base, issue_number)
            if comments:
                r.fail(f"Unexpected bot comment: {comments[0]['body'][:200]}")
            else:
                r.ok(f"Issue #{issue_number}: clean pass")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if issue_number:
                try:
                    api_close_issue(api_base, issue_number)
                except Exception:
                    pass

    def test_edit() -> Result:
        r = Result("Demo: Edit vague→complete → label removed")
        issue_number = None
        try:
            issue_number = create_issue(api_base, "[Bug]: 安装失败", "装不上，报错了")
            print(f"  Created issue #{issue_number}")
            labels, _ = wait_for_bot_comment(api_base, issue_number)
            if LABEL_NAME not in labels:
                r.fail(f"Phase 1: Expected label not found. Labels: {labels}")
                return r
            print(f"  Phase 1: label added")
            time.sleep(5)
            edit_issue(
                api_base,
                issue_number,
                "[Bug]: Ubuntu 22.04 CANN 8.2 安装失败",
                "### 环境信息\n- Ubuntu 22.04\n- CANN 8.2.0\n### 问题描述\npip install ERROR: No matching distribution",
            )
            print(f"  Edited issue #{issue_number}")
            labels2, _ = wait_for_bot_comment(api_base, issue_number, max_wait=120)
            if LABEL_NAME in labels2:
                r.fail(f"Phase 2: Label should be removed. Labels: {labels2}")
            else:
                r.ok(f"Issue #{issue_number}: label added then removed")
            return r
        except Exception as e:
            r.fail(str(e))
            return r
        finally:
            if issue_number:
                try:
                    api_close_issue(api_base, issue_number)
                except Exception:
                    pass

    tests = [test_incomplete_bug, test_complete_bug, test_edit]
    results = []
    for i, fn in enumerate(tests):
        print(f"[{i + 1}/3] {fn.__doc__}")
        r = fn()
        results.append(r)
        print(f"     {'PASS' if r.passed else 'FAIL: ' + r.error}")
        print()
    return results


def print_summary(results: list[Result]):
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"Results: {passed}/{len(results)} passed")
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name}")
        if r.details:
            print(f"         {r.details}")
        if r.error:
            print(f"         Error: {r.error}")
    if passed != len(results):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Issue Description Completeness Check bot")
    parser.add_argument("--repo", default="ai-infra-develop/vllm-ascend", help="GitHub repo (owner/name)")
    parser.add_argument("--csv", default=None, help="Path to CSV test cases (default: description_check_cases.csv)")
    parser.add_argument("--demo", action="store_true", help="Run 3 built-in demo tests instead of CSV")
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    api_base = f"https://api.github.com/repos/{args.repo}"
    print(f"Repo: {args.repo}")
    print(f"Label: {LABEL_NAME}")
    print(f"Poll: {POLL_INTERVAL}s interval, {MAX_WAIT}s max wait")
    print()

    if args.demo:
        results = run_demo_tests(api_base)
    else:
        csv_path = args.csv or str(DEFAULT_CSV)
        if not Path(csv_path).exists():
            print(f"ERROR: CSV not found: {csv_path}")
            print("Use --csv to specify a path or --demo for built-in tests")
            sys.exit(1)
        results = run_csv_tests(api_base, csv_path)

    print_summary(results)


if __name__ == "__main__":
    main()
