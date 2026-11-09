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
    CODEOWNERS health check (strict). Fails when:
    - a tracked file resolves to no owner at all - no matching entry, or only
      ownerless TODO entries - unless it is explicitly marked as deliberately
      unowned with a ``SKIP(owner)`` comment;
    - a non-comment entry matches no tracked file (stale entry: deleted file,
      removed directory, glob with zero hits);
    - an active entry has no owners (GitHub ignores it silently).

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
import urllib.request
from dataclasses import dataclass, field
from typing import Any

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


def resolve_owners(rules: list[Rule], path: str) -> tuple[list[str], bool]:
    """Owners responsible for ``path``.

    Returns ``(owners, skipped)``. Owners of EVERY matching active rule are
    unioned: a parent module owner also covers its submodules, so a more
    specific rule adds owners rather than replacing them.

    - one approval from any owner satisfies the approval gate;
    - a pending "changes requested" from any owner blocks the file;
    - a matching ``SKIP(owner)`` rule is the last word on its subtree -
      deliberately unowned, neither approval nor veto applies.
      ``TODO(owner)`` fills owners only when no active rule with owners
      matched and no SKIP overrode the subtree.
    """
    owners: list[str] = []
    skipped = False
    for rule in rules:
        if not pattern_matches(rule.pattern, path):
            continue
        if rule.status == "skip":
            owners = []
            skipped = True
        elif rule.status == "active" and rule.owners:
            owners.extend(o for o in rule.owners if o not in owners)
            skipped = False
        elif rule.status == "todo" and not owners and not skipped:
            owners = rule.owners
    return owners, skipped


def _token() -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit("GH_TOKEN or GITHUB_TOKEN must be set for GitHub API access")


def gh_api(endpoint: str, **params: object) -> Any:
    """Paginated GitHub REST call using urllib (no gh CLI dependency).

    Self-hosted runners do not ship the gh CLI, so requests go through the
    stdlib instead. All pages are fetched and concatenated: for array
    endpoints items are flattened into one list, for object endpoints the
    last page wins.
    """
    repo = os.environ["GITHUB_REPOSITORY"]
    path = endpoint.format(owner=repo.split("/")[0], repo=repo.split("/")[1], **params)
    items: list[object] = []
    is_array = False
    url: str | None = f"https://api.github.com/{path}?per_page=100"
    while url:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {_token()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request) as response:
            body = json.load(response)
        is_array = isinstance(body, list)
        items.extend(body if is_array else [body])
        url = _next_link(response.headers.get("Link", ""))
    return items if is_array else (items[-1] if items else {})


def _next_link(link_header: str) -> str | None:
    """Extract the rel="next" URL from a GitHub Link header."""
    for part in link_header.split(","):
        segment = part.strip()
        if segment.endswith('rel="next"'):
            return segment[segment.index("<") + 1 : segment.index(">")]
    return None


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
        owners, skipped = resolve_owners(rules, path)
        if skipped:
            # Deliberately unowned path (SKIP(owner)): no approval required.
            continue
        # A pending "changes requested" from any owner of the file (excluding
        # the PR author) blocks it even when another owner approved; the
        # author must address the feedback and get the owner to re-review.
        blocking = [
            o.lstrip("@")
            for o in owners
            if o.lstrip("@") != pr_author and reviews.get(o.lstrip("@")) == "CHANGES_REQUESTED"
        ]
        if blocking:
            failures.append(
                f"{path}: changes requested by {', '.join('@' + b for b in blocking)}; "
                "address the feedback, then ask them to re-review (approve or dismiss)"
            )
            continue
        eligible = [o for o in owners if o.lstrip("@") != pr_author]
        approved = [o.lstrip("@") for o in eligible if reviews.get(o.lstrip("@")) == "APPROVED"]
        if not owners:
            failures.append(f"{path}: no CODEOWNERS owner matches this file")
            continue
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
    """CODEOWNERS health: real ownership of tracked files + stale entries.

    Strict mode: a file whose resolved owner set is empty fails the check,
    even when an ownerless TODO entry nominally covers it - an entry without
    owners means nobody is accountable. Only ``SKIP(owner)`` paths are
    exempt.
    """
    failures = []

    for path in tracked:
        if path == CODEOWNERS_PATH:
            continue
        owners, skipped = resolve_owners(rules, path)
        if skipped or owners:
            continue
        failures.append(f"file has no CODEOWNERS owner: {path}")

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
            for failure in failures:
                print(f"  ::error::{failure}")
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
            for failure in failures:
                print(f"  ::error::{failure}")
            return 1
        print(f"codeowner coverage check PASSED ({len(tracked)} files)")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
