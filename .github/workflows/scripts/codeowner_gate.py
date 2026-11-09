#!/usr/bin/env python3
#
# Copyright (c) 2026 vLLM Ascend contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CODEOWNERS merge gate.

Two subcommands used by the ``codeowner-approval`` job inside the ci-gate
workflow (``.github/workflows/pr_test.yaml``):

``check-approvals --pr N``
    Every file changed by the PR must have at least one CODEOWNERS owner whose
    latest review is APPROVED. Any GitHub user counts as a valid approver for
    an owner name (owners without write permission or without org membership
    are accepted on purpose; this is why the native
    ``require_code_owner_review`` is not used). The PR author never counts as
    an approver, even when listed as an owner.

``check-coverage``
    CODEOWNERS health check. Fails when:
    - a tracked file has no owner entry (active or TODO comment), unless it is
      explicitly marked as deliberately unowned with a ``SKIP(owner)`` comment;
    - a non-comment entry matches no tracked file (stale entry: deleted file,
      removed directory, glob with zero hits).

Comments carry two markers parsed from comment lines:
- ``TODO(owner): <pattern>`` followed by ``# <pattern> [@owner ...]``: pending
  entry, inactive but counts toward coverage.
- ``SKIP(owner): <pattern>``: deliberately unowned path, excluded from
  coverage requirements.
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

CODEOWNERS_PATH = ".github/CODEOWNERS"


@dataclass
class Rule:
    """One CODEOWNERS entry (active, TODO, or SKIP)."""

    pattern: str  # without leading '/', trailing '/' stripped
    owners: list[str] = field(default_factory=list)
    status: str = "active"  # active | todo | skip
    line: int = 0


def parse_codeowners(text: str) -> list[Rule]:
    """Parse CODEOWNERS content into rules, honoring TODO/SKIP comment markers."""
    rules: list[Rule] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Comment lines carry the TODO/SKIP markers: "# TODO(owner): ..." etc.
        comment_body = stripped[1:].strip() if stripped.startswith("#") else ""
        if comment_body.startswith("TODO(owner):"):
            todo_parts = comment_body[len("TODO(owner):") :].split()
            pattern = _normalize_pattern(todo_parts[0])
            owners: list[str] = []
            # Optional rule line follows: "# /path @owner1 @owner2"
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt.startswith("# /"):
                    parts = nxt[2:].split()
                    owners = [p for p in parts[1:] if p.startswith("@")]
            rules.append(Rule(pattern, owners, "todo", i + 1))
            i += 2
            continue
        if comment_body.startswith("SKIP(owner):"):
            pattern = _normalize_pattern(comment_body[len("SKIP(owner):") :].strip())
            rules.append(Rule(pattern, [], "skip", i + 1))
            i += 1
            continue
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        parts = stripped.split()
        owners = [p for p in parts[1:] if p.startswith("@")]
        rules.append(Rule(_normalize_pattern(parts[0]), owners, "active", i + 1))
        i += 1
    return rules


def _normalize_pattern(raw: str) -> str:
    pattern = raw.lstrip("/")
    return pattern.rstrip("/")


def pattern_matches(pattern: str, path: str) -> bool:
    """CODEOWNERS matching: gitignore-style prefix for dirs, fnmatch for globs.

    A ``*`` never crosses a path separator (gitignore semantics), so
    ``ops/*.py`` matches ``ops/foo.py`` but not ``ops/sub/foo.py``.
    """
    if "*" in pattern:
        pat_parts = pattern.split("/")
        path_parts = path.split("/")
        if len(pat_parts) != len(path_parts):
            return False
        return all(fnmatch.fnmatch(p, g) for p, g in zip(path_parts, pat_parts))
    return path == pattern or path.startswith(pattern + "/")


def match_rule(rules: list[Rule], path: str) -> Rule | None:
    """Last matching rule wins (CODEOWNERS semantics)."""
    result = None
    for rule in rules:
        if pattern_matches(rule.pattern, path):
            result = rule
    return result


def active_owners_for(rules: list[Rule], path: str) -> list[str]:
    """Owners required for a path, resolving TODO fallbacks to an active rule."""
    owners: list[str] = []
    for rule in rules:
        if not pattern_matches(rule.pattern, path):
            continue
        has_owners = bool(rule.owners)
        if rule.status == "active" and has_owners or rule.status == "todo" and not owners:
            owners = rule.owners
    return owners


def gh_api(endpoint: str, **params: object) -> object:
    repo = os.environ["GITHUB_REPOSITORY"]
    endpoint = endpoint.format(owner=repo.split("/")[0], repo=repo.split("/")[1], **params)
    out = subprocess.run(
        # --paginate + --slurp: gh concatenates raw pages for array endpoints,
        # which is not valid JSON on its own; --slurp wraps all pages in one
        # array that we flatten below.
        ["gh", "api", endpoint, "--paginate", "--slurp"],
        capture_output=True,
        text=True,
        check=True,
    )
    pages = json.loads(out.stdout)
    if pages and isinstance(pages[0], list):
        return [item for page in pages for item in page]
    return pages[-1]


def get_pr_files(pr_number: int) -> list[str]:
    data = gh_api("repos/{owner}/{repo}/pulls/{pr}/files", pr=pr_number)
    return sorted({item["filename"] for item in data})


def get_pr_reviews(pr_number: int) -> dict[str, str]:
    """Latest review state per user."""
    data = gh_api("repos/{owner}/{repo}/pulls/{pr}/reviews", pr=pr_number)
    latest: dict[str, str] = {}
    for review in data:
        user = review["user"]["login"]
        if review["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest[user] = review["state"]
    return latest


def check_approvals(
    pr_number: int, pr_author: str, rules: list[Rule], files: list[str], reviews: dict[str, str]
) -> list[str]:
    """Return list of failures (empty means pass)."""
    failures = []
    for path in files:
        owners = active_owners_for(rules, path)
        approved = [
            o.lstrip("@") for o in owners if reviews.get(o.lstrip("@")) == "APPROVED" and o.lstrip("@") != pr_author
        ]
        if not owners:
            failures.append(f"{path}: no CODEOWNERS owner matches this file")
            continue
        eligible = [o for o in owners if o.lstrip("@") != pr_author]
        if not eligible:
            failures.append(
                f"{path}: only owner is the PR author ({pr_author}); another owner must be added to CODEOWNERS"
            )
        elif not approved:
            failures.append(f"{path}: waiting for approval from one of: {', '.join(eligible)}")
    return failures


def list_tracked_files(repo_root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True, cwd=repo_root)
    return out.stdout.splitlines()


def check_coverage(rules: list[Rule], tracked: list[str]) -> list[str]:
    """CODEOWNERS health: coverage of tracked files + stale entry detection."""
    failures = []

    def has_rule(path: str, statuses: set[str]) -> bool:
        # An active rule without owners is ignored by GitHub, so it provides
        # no real coverage.
        return any(
            r.status in statuses and pattern_matches(r.pattern, path) and (r.owners or r.status != "active")
            for r in rules
        )

    covered = {p for p in tracked if has_rule(p, {"active", "todo"})}
    deliberately_unowned = {p for p in tracked if has_rule(p, {"skip"})}
    for path in tracked:
        if path in deliberately_unowned:
            continue
        if path == CODEOWNERS_PATH:
            continue
        if path not in covered:
            failures.append(f"file has no CODEOWNERS entry: {path}")

    tracked_set = set(tracked)
    for rule in rules:
        if rule.status != "active":
            continue
        if not rule.owners:
            failures.append(
                f"CODEOWNERS entry has no owner and is ignored by GitHub (line {rule.line}): /{rule.pattern}"
            )
            continue
        hit = any(pattern_matches(rule.pattern, path) for path in tracked_set)
        if not hit:
            failures.append(f"stale CODEOWNERS entry (line {rule.line}): /{rule.pattern} matches no tracked file")
    return failures


def load_reviews_from_file(path: str) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def load_files_from_file(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_approvals = sub.add_parser("check-approvals", help="gate on owner approvals")
    p_approvals.add_argument("--pr", type=int, help="PR number (uses gh api)")
    p_approvals.add_argument("--author", default="", help="PR author login")
    p_approvals.add_argument("--files-json", help="local fixture: changed files JSON")
    p_approvals.add_argument("--reviews-json", help="local fixture: reviews JSON")

    p_cov = sub.add_parser("check-coverage", help="CODEOWNERS health check")
    p_cov.add_argument("--files-json", help="local fixture: tracked files JSON")
    p_cov.add_argument("--repo-root", default=".")

    args = parser.parse_args()
    with open(CODEOWNERS_PATH) as f:
        rules = parse_codeowners(f.read())

    if args.command == "check-approvals":
        if args.files_json and args.reviews_json:
            files = load_files_from_file(args.files_json)
            reviews = load_reviews_from_file(args.reviews_json)
        else:
            if not args.pr:
                parser.error("check-approvals requires --pr or fixtures")
            files = get_pr_files(args.pr)
            reviews = get_pr_reviews(args.pr)
            if not args.author:
                pr_data = gh_api("repos/{owner}/{repo}/pulls/{pr}", pr=args.pr)
                args.author = pr_data["user"]["login"]
        failures = check_approvals(args.pr or 0, args.author, rules, files, reviews)
        if failures:
            print("codeowner approval check FAILED:")
            for f in failures:
                print(f"  ::error::{f}")
            return 1
        print("codeowner approval check PASSED")
        return 0

    if args.command == "check-coverage":
        if args.files_json:
            tracked = load_files_from_file(args.files_json)
        else:
            tracked = list_tracked_files(args.repo_root)
        failures = check_coverage(rules, tracked)
        if failures:
            print("codeowner coverage check FAILED:")
            for f in failures:
                print(f"  ::error::{f}")
            return 1
        print(f"codeowner coverage check PASSED ({len(tracked)} files)")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
