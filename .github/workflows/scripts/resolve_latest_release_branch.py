#!/usr/bin/env python3
"""Pick the newest ``releases/v*`` branch by creation time.

Git does not record when a branch ref was created. On GitHub the durable
proxy is the ``created_at`` of the earliest pull request that targets the
branch: the branch existed by then, and release branches receive a PR as
soon as they are cut. The branch whose earliest PR is the most recent is
the newest release branch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import regex as re

RELEASE_BRANCH_RE = re.compile(r"^releases/v[0-9][0-9A-Za-z._-]*$")
COVERAGE_SUFFIX_RE = re.compile(r"^v[0-9][0-9A-Za-z._-]*$")
SEARCH_URL = "https://api.github.com/search/issues"
RETRY_STATUS = {403, 429}
RETRY_DELAY_SECONDS = 2


def filter_release_branches(names: list[str]) -> list[str]:
    """Keep versioned release branches. Drop revert branches and other refs."""
    return [name for name in names if RELEASE_BRANCH_RE.fullmatch(name)]


def select_latest(created_at_by_branch: dict[str, str]) -> str:
    """Return the branch with the latest creation timestamp.

    ISO-8601 timestamps from GitHub sort lexicographically. Equal timestamps
    break toward the greater branch name so the result is stable.
    """
    if not created_at_by_branch:
        raise ValueError("no release branch has a creation timestamp")
    return max(created_at_by_branch.items(), key=lambda item: (item[1], item[0]))[0]


def coverage_file_suffix(branch: str) -> str:
    """Filename suffix for a release branch, such as ``v0.30.0rc2``.

    ``main`` has no suffix, so its object names stay ``coverage.tar``.
    """
    if branch == "main":
        return ""
    if RELEASE_BRANCH_RE.fullmatch(branch) is None:
        raise ValueError(f"unsupported coverage branch: {branch}")
    return branch.split("/", 1)[1]


def coverage_object_name(filename: str, suffix: str) -> str:
    """Insert ``suffix`` before the extension: ``coverage.tar`` -> ``coverage-v0.30.0rc2.tar``."""
    if suffix and COVERAGE_SUFFIX_RE.fullmatch(suffix) is None:
        raise ValueError(f"unsupported coverage file suffix: {suffix}")
    if not suffix:
        return filename
    stem, extension = filename.rsplit(".", 1)
    return f"{stem}-{suffix}.{extension}"


def created_at_from_search(payload: dict) -> str | None:
    items = payload.get("items") or []
    if not items:
        return None
    created_at = items[0].get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("search item is missing created_at")
    return created_at


def earliest_pr_created_at(repo: str, branch: str, token: str) -> str | None:
    """Return the creation time of the earliest PR targeting ``branch``."""
    if RELEASE_BRANCH_RE.fullmatch(branch) is None:
        raise ValueError(f"refusing to query unexpected branch name: {branch}")
    query = f'repo:{repo} is:pr base:"{branch}"'
    query_params = {"q": query, "sort": "created", "order": "asc", "per_page": "1"}
    url = SEARCH_URL + "?" + urllib.parse.urlencode(query_params)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "vllm-ascend-coverage",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    payload = _read_json(request)
    return created_at_from_search(payload)


def _read_json(request: urllib.request.Request, allow_retry: bool = True) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if allow_retry and exc.code in RETRY_STATUS:
            time.sleep(RETRY_DELAY_SECONDS)
            return _read_json(request, allow_retry=False)
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub search failed ({exc.code}): {detail}") from exc


def resolve_latest_release_branch(repo: str, branch_names: list[str], token: str) -> str:
    candidates = filter_release_branches(branch_names)
    if not candidates:
        raise ValueError("no releases/v* branches")
    created_at_by_branch: dict[str, str] = {}
    for branch in candidates:
        created_at = earliest_pr_created_at(repo, branch, token)
        if created_at is None:
            print(f"skip {branch}: no pull request targets it", file=sys.stderr)
            continue
        print(f"{created_at} {branch}", file=sys.stderr)
        created_at_by_branch[branch] = created_at
    return select_latest(created_at_by_branch)


def _read_branch_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--branches-file", default="")
    parser.add_argument(
        "--suffix-for",
        default="",
        help="Print only the coverage filename suffix for this branch and exit.",
    )
    args = parser.parse_args()

    if args.suffix_for:
        try:
            value = coverage_file_suffix(args.suffix_for)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(2) from exc
        print(value)
        return
    if not args.branches_file:
        parser.error("--branches-file is required")

    token = os.environ.get("GITHUB_TOKEN", "")
    if not args.repo:
        raise SystemExit("missing --repo or GITHUB_REPOSITORY")
    if not token:
        raise SystemExit("missing GITHUB_TOKEN")

    branch = resolve_latest_release_branch(args.repo, _read_branch_names(args.branches_file), token)
    suffix = coverage_file_suffix(branch)
    # Shell-safe: both values are constrained by RELEASE_BRANCH_RE.
    print(f"TARGET_BRANCH={branch}")
    print(f"COVERAGE_FILE_SUFFIX={suffix}")


if __name__ == "__main__":
    main()
