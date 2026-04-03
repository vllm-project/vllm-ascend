#!/usr/bin/env python3
"""
Fetch all commits between two tags from a GitHub repository.
Usage: python fetch_commits.py [--token YOUR_GITHUB_TOKEN]
"""

import argparse
import os
import re

import dotenv
import requests

# Load .env.local first (higher priority), then .env as fallback
dotenv.load_dotenv(".env.local")
dotenv.load_dotenv()  # .env as fallback


def get_github_token():
    """Get GitHub token from environment or argument."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def resolve_tag_to_sha(base_url: str, tag: str, headers: dict) -> str:
    """Resolve a tag name to its commit SHA."""
    print(f"Resolving tag {tag}...")

    tag_resp = requests.get(f"{base_url}/git/refs/tags/{tag}", headers=headers)
    if tag_resp.status_code != 200:
        raise Exception(f"Failed to get tag {tag}: {tag_resp.text}")

    tag_data = tag_resp.json()
    sha = tag_data["object"]["sha"]

    # If it's an annotated tag, we need to get the commit it points to
    if tag_data["object"]["type"] == "tag":
        tag_obj_resp = requests.get(f"{base_url}/git/tags/{sha}", headers=headers)
        if tag_obj_resp.status_code == 200:
            sha = tag_obj_resp.json()["object"]["sha"]

    return sha


def resolve_commit_sha(base_url: str, commit_ref: str, headers: dict) -> str:
    """Resolve a commit reference (SHA or short SHA) to full SHA."""
    print(f"Resolving commit {commit_ref}...")

    commit_resp = requests.get(f"{base_url}/commits/{commit_ref}", headers=headers)
    if commit_resp.status_code != 200:
        raise Exception(f"Failed to get commit {commit_ref}: {commit_resp.text}")

    return commit_resp.json()["sha"]


def get_default_branch_head(base_url: str, headers: dict) -> tuple[str, str]:
    """
    Get the HEAD commit of the default branch.

    Returns:
        Tuple of (branch_name, head_sha)
    """
    print("Getting default branch HEAD...")

    # Get repository info to find default branch
    repo_resp = requests.get(base_url, headers=headers)
    if repo_resp.status_code != 200:
        raise Exception(f"Failed to get repository info: {repo_resp.text}")

    default_branch = repo_resp.json()["default_branch"]
    print(f"  Default branch: {default_branch}")

    # Get the HEAD commit of the default branch
    branch_resp = requests.get(f"{base_url}/branches/{default_branch}", headers=headers)
    if branch_resp.status_code != 200:
        raise Exception(f"Failed to get branch {default_branch}: {branch_resp.text}")

    head_sha = branch_resp.json()["commit"]["sha"]
    print(f"  HEAD: {head_sha[:8]}")

    return (default_branch, head_sha)


def get_all_tags(base_url: str, headers: dict) -> list[dict]:
    """Get all tags from the repository with their commit SHAs and dates."""
    print("Fetching all tags...")

    all_tags = []
    page = 1
    per_page = 100

    while True:
        resp = requests.get(
            f"{base_url}/tags",
            headers=headers,
            params={"per_page": per_page, "page": page},
        )

        if resp.status_code != 200:
            raise Exception(f"Failed to get tags: {resp.text}")

        tags = resp.json()
        if not tags:
            break

        all_tags.extend(tags)
        page += 1

        if len(tags) < per_page:
            break

    print(f"  Found {len(all_tags)} tags")
    return all_tags


def get_commit_date(base_url: str, sha: str, headers: dict) -> str:
    """Get the commit date for a given SHA."""
    commit_resp = requests.get(f"{base_url}/commits/{sha}", headers=headers)
    if commit_resp.status_code != 200:
        return None
    return commit_resp.json()["commit"]["committer"]["date"]


def find_previous_tag(
    base_url: str, head_sha: str, headers: dict, tag_pattern: str | None = None
) -> tuple[str, str] | None:
    """
    Find the most recent tag that is an ancestor of the given commit.

    Uses git history to find tags that are reachable from the commit.

    Args:
        base_url: GitHub API base URL
        head_sha: The commit SHA to search from
        headers: Request headers
        tag_pattern: Optional regex pattern to filter tags (e.g., r'^v\\d+\\.\\d+\\.\\d+$')

    Returns:
        Tuple of (tag_name, tag_sha) or None if no tag found
    """
    print(f"Finding previous tag before commit {head_sha[:8]}...")

    # Get the date of the head commit
    head_date = get_commit_date(base_url, head_sha, headers)
    if not head_date:
        print("  Warning: Could not get head commit date")
        return None

    print(f"  Head commit date: {head_date}")

    # Get all tags
    all_tags = get_all_tags(base_url, headers)

    # Filter tags by pattern if provided
    if tag_pattern:
        import re

        pattern = re.compile(tag_pattern)
        all_tags = [t for t in all_tags if pattern.match(t["name"])]
        print(f"  After pattern filter: {len(all_tags)} tags")

    # For each tag, check if it's an ancestor of head_sha and get its date
    tag_candidates = []

    for tag in all_tags:
        tag_name = tag["name"]
        tag_commit_sha = tag["commit"]["sha"]

        # Skip if this is the same commit as head
        if tag_commit_sha == head_sha:
            continue

        # Check if this tag's commit is an ancestor of head
        compare_resp = requests.get(f"{base_url}/compare/{tag_commit_sha}...{head_sha}", headers=headers)

        if compare_resp.status_code != 200:
            continue

        compare_data = compare_resp.json()

        # If tag is behind head (status = "behind" or "ahead"), it's an ancestor
        # We want tags where the comparison shows head is ahead
        if compare_data.get("status") in ["ahead", "diverged"]:
            # Get the tag's commit date
            tag_date = get_commit_date(base_url, tag_commit_sha, headers)
            if tag_date and tag_date < head_date:
                tag_candidates.append(
                    {
                        "name": tag_name,
                        "sha": tag_commit_sha,
                        "date": tag_date,
                        "ahead_by": compare_data.get("ahead_by", 0),
                    }
                )
                print(f"  Found candidate: {tag_name} ({compare_data.get('ahead_by', 0)} commits behind)")

    if not tag_candidates:
        print("  No previous tag found")
        return None

    # Sort by date (most recent first) or by ahead_by (smallest first)
    # Using ahead_by gives us the closest tag
    tag_candidates.sort(key=lambda x: x["ahead_by"])

    best_tag = tag_candidates[0]
    print(f"  Selected: {best_tag['name']} ({best_tag['ahead_by']} commits behind)")

    return (best_tag["name"], best_tag["sha"])


def fetch_commits_between_tags(
    owner: str, repo: str, base_tag: str, head_tag: str, token: str | None = None
) -> list[dict]:
    """
    Fetch all commits between two tags by walking the commit graph.

    This method traverses from head_tag back to base_tag, collecting all commits.
    It properly handles the commit history and doesn't rely on date filtering.

    Args:
        owner: Repository owner (e.g., 'vllm-project')
        repo: Repository name (e.g., 'vllm')
        base_tag: Base tag (older, e.g., 'v0.11.2')
        head_tag: Head tag (newer, e.g., 'v0.12.0')
        token: Optional GitHub token for higher rate limits

    Returns:
        List of commit dictionaries
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Resolve tags to commit SHAs
    base_sha = resolve_tag_to_sha(base_url, base_tag, headers)
    head_sha = resolve_tag_to_sha(base_url, head_tag, headers)

    print(f"\nBase SHA ({base_tag}): {base_sha}")
    print(f"Head SHA ({head_tag}): {head_sha}")

    # First, use Compare API to get total commit count (for progress info)
    print(f"\nComparing {base_tag}...{head_tag}...")
    compare_resp = requests.get(f"{base_url}/compare/{base_sha}...{head_sha}", headers=headers)
    if compare_resp.status_code == 200:
        compare_data = compare_resp.json()
        total_commits = compare_data.get("total_commits", "unknown")
        print(f"Total commits to fetch: {total_commits}")

    # Walk the commit history from head to base
    # We use the commits API starting from head_sha and stop when we reach base_sha
    all_commits = []
    seen_shas = set()
    seen_shas.add(base_sha)  # Don't include the base commit itself

    # BFS traversal of commit graph
    to_visit = [head_sha]
    page_count = 0

    print(f"\nFetching commits from {head_tag} back to {base_tag}...")

    while to_visit:
        current_sha = to_visit.pop(0)

        if current_sha in seen_shas:
            continue

        seen_shas.add(current_sha)

        # Fetch commit details
        commit_resp = requests.get(f"{base_url}/commits/{current_sha}", headers=headers)

        if commit_resp.status_code != 200:
            print(f"  Warning: Failed to fetch commit {current_sha[:8]}")
            continue

        commit = commit_resp.json()
        all_commits.append(commit)

        # Add parent commits to visit queue
        for parent in commit.get("parents", []):
            parent_sha = parent["sha"]
            if parent_sha not in seen_shas:
                to_visit.append(parent_sha)

        # Progress logging
        if len(all_commits) % 50 == 0:
            page_count += 1
            print(f"  Fetched {len(all_commits)} commits...")

    print(f"  Completed: {len(all_commits)} commits fetched")

    return all_commits


def fetch_commits_by_date_range(
    owner: str,
    repo: str,
    since: str,
    until: str,
    token: str | None = None,
    branch: str | None = None,
) -> list[dict]:
    """
    Fetch all commits within a date range.

    Args:
        owner: Repository owner (e.g., 'vllm-project')
        repo: Repository name (e.g., 'vllm')
        since: Start date (ISO 8601 format, e.g., '2025-01-01' or '2025-01-01T00:00:00Z')
        until: End date (ISO 8601 format, e.g., '2025-01-31' or '2025-01-31T23:59:59Z')
        token: Optional GitHub token for higher rate limits
        branch: Optional branch name (defaults to repository's default branch)

    Returns:
        List of commit dictionaries
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    per_page = 100

    # Normalize date format - add time if not present
    if len(since) == 10:  # YYYY-MM-DD format
        since = f"{since}T00:00:00Z"
    if len(until) == 10:  # YYYY-MM-DD format
        until = f"{until}T23:59:59Z"

    print(f"\nFetching commits from {since} to {until}...")
    if branch:
        print(f"  Branch: {branch}")

    all_commits = []
    page = 1

    while True:
        params = {"since": since, "until": until, "per_page": per_page, "page": page}
        if branch:
            params["sha"] = branch

        response = requests.get(f"{base_url}/commits", headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch commits: {response.text}")

        commits = response.json()
        if not commits:
            break

        all_commits.extend(commits)
        print(f"  Page {page}: fetched {len(commits)} commits (total: {len(all_commits)})")

        if len(commits) < per_page:
            break

        page += 1

    print(f"  Completed: {len(all_commits)} commits fetched")
    return all_commits


def get_merge_base(base_url: str, base_sha: str, head_sha: str, headers: dict) -> str | None:
    """
    Get the merge base (common ancestor) of two commits.

    Args:
        base_url: GitHub API base URL for the repo
        base_sha: First commit SHA
        head_sha: Second commit SHA
        headers: Request headers

    Returns:
        Merge base commit SHA, or None if not found
    """
    # GitHub Compare API returns merge_base_commit
    compare_resp = requests.get(
        f"{base_url}/compare/{base_sha}...{head_sha}",
        headers=headers,
    )

    if compare_resp.status_code != 200:
        return None

    compare_data = compare_resp.json()
    merge_base = compare_data.get("merge_base_commit", {}).get("sha")
    return merge_base


def fetch_commits_by_walking_history(
    base_url: str,
    base_sha: str,
    head_sha: str,
    base_tag: str,
    head_tag: str,
    headers: dict,
    stop_sha: str | None = None,
) -> list[dict]:
    """
    Fetch commits by walking the commit history from head to a stop point.

    This method correctly handles release branches with cherry-picks.
    It walks the head's commit history until it reaches the stop commit.

    Args:
        base_url: GitHub API base URL for the repo
        base_sha: Base commit SHA (for display purposes)
        head_sha: Head commit SHA (newer)
        base_tag: Display name for base reference
        head_tag: Display name for head reference
        headers: Request headers
        stop_sha: SHA to stop at (if None, uses base_sha)

    Returns:
        List of commit dictionaries (excluding stop commit)
    """
    per_page = 100
    all_commits = []
    page = 1
    target_sha = stop_sha or base_sha

    print(f"\nWalking commit history from {head_tag} back to {base_tag}...")
    print(f"  Stop SHA: {target_sha[:8]}")

    while True:
        response = requests.get(
            f"{base_url}/commits",
            headers=headers,
            params={"sha": head_sha, "per_page": per_page, "page": page},
        )

        if response.status_code != 200:
            print(f"  Warning: API error on page {page}, stopping")
            break

        commits = response.json()
        if not commits:
            print(f"  No more commits found on page {page}")
            break

        found_stop = False
        for commit in commits:
            if commit["sha"] == target_sha:
                # Reached stop commit, stop (don't include it)
                found_stop = True
                break
            all_commits.append(commit)

        print(f"  Page {page}: fetched {len(commits)} commits (total: {len(all_commits)})")

        if found_stop:
            print(f"  Reached stop commit ({target_sha[:8]})")
            break

        if len(commits) < per_page:
            print("  Warning: Reached end of history without finding stop commit")
            break

        page += 1

    return all_commits


def fetch_commits_between_tags_fast(
    owner: str,
    repo: str,
    base_tag: str,
    head_tag: str,
    token: str | None = None,
    head_is_commit: bool = False,
    base_is_commit: bool = False,
) -> list[dict]:
    """
    Fetch all commits between two tags using GitHub Compare API with pagination.

    This properly fetches only the commits between the two tags.
    Automatically handles diverged branches (e.g., release branches with cherry-picks)
    by falling back to walking the commit history.

    Args:
        owner: Repository owner (e.g., 'vllm-project')
        repo: Repository name (e.g., 'vllm')
        base_tag: Base tag (older, e.g., 'v0.11.2') or commit SHA if base_is_commit=True
        head_tag: Head tag (newer, e.g., 'v0.12.0') or commit SHA if head_is_commit=True
        token: Optional GitHub token for higher rate limits
        head_is_commit: If True, treat head_tag as a commit SHA instead of a tag
        base_is_commit: If True, treat base_tag as a commit SHA instead of a tag

    Returns:
        List of commit dictionaries
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    per_page = 100

    # Resolve to commit SHAs
    if base_is_commit:
        base_sha = resolve_commit_sha(base_url, base_tag, headers)
    else:
        base_sha = resolve_tag_to_sha(base_url, base_tag, headers)

    if head_is_commit:
        head_sha = resolve_commit_sha(base_url, head_tag, headers)
    else:
        head_sha = resolve_tag_to_sha(base_url, head_tag, headers)

    print(f"\nBase SHA ({base_tag}): {base_sha}")
    print(f"Head SHA ({head_tag}): {head_sha}")

    # Use Compare API to check relationship and get commits
    print(f"\nComparing {base_tag}...{head_tag}...")
    compare_resp = requests.get(
        f"{base_url}/compare/{base_sha}...{head_sha}",
        headers=headers,
        params={"per_page": per_page},
    )

    if compare_resp.status_code != 200:
        raise Exception(f"Failed to compare: {compare_resp.text}")

    compare_data = compare_resp.json()
    status = compare_data.get("status", "unknown")
    total_commits = compare_data.get("total_commits", 0)

    print(f"  Comparison status: {status}")
    print(f"  Total commits: {total_commits}")

    # Get merge_base for potential fallback
    merge_base = compare_data.get("merge_base_commit", {}).get("sha")

    # If branches have diverged (e.g., release branch with cherry-picks),
    # we need to filter by PR numbers to avoid duplicates
    is_diverged = status == "diverged"

    if is_diverged:
        print("\n  Branches have diverged (likely a release branch scenario)")
        print("  Will filter by PR numbers to handle cherry-picks...")
        if merge_base:
            print(f"  Merge base: {merge_base[:8]}")

    # Use Compare API results
    all_commits = compare_data.get("commits", [])
    print(f"  Initial fetch: {len(all_commits)} commits")

    if len(all_commits) >= total_commits:
        print("  All commits fetched in initial response")
        return all_commits

    # Need to paginate - try Compare API pagination first
    page = 1
    while len(all_commits) < total_commits:
        page += 1
        print(f"  Fetching page {page}...")

        compare_resp = requests.get(
            f"{base_url}/compare/{base_sha}...{head_sha}",
            headers=headers,
            params={"per_page": per_page, "page": page},
        )

        if compare_resp.status_code != 200:
            # Compare API doesn't support pagination well for large diffs
            print("  Compare API pagination not supported, using commit walk...")
            break

        page_data = compare_resp.json()
        page_commits = page_data.get("commits", [])

        if not page_commits:
            break

        all_commits.extend(page_commits)
        print(f"  Page {page}: got {len(page_commits)} commits (total: {len(all_commits)})")

    # If we still don't have all commits, walk the history
    if len(all_commits) < total_commits:
        print(f"\n  Need to fetch remaining {total_commits - len(all_commits)} commits via history walk...")

        # For diverged branches, use merge_base as stop point
        # For non-diverged, use base_sha
        stop_sha = merge_base if (status == "diverged" and merge_base) else base_sha

        # Get commits we already have
        seen_shas = {c["sha"] for c in all_commits}

        # Walk from head, collecting commits not already seen, until we reach stop point
        walk_commits = []
        walk_page = 1
        found_stop = False

        while len(all_commits) + len(walk_commits) < total_commits and not found_stop:
            response = requests.get(
                f"{base_url}/commits",
                headers=headers,
                params={"sha": head_sha, "per_page": per_page, "page": walk_page},
            )

            if response.status_code != 200:
                print(f"  Warning: API error on page {walk_page}")
                break

            commits = response.json()
            if not commits:
                break

            for commit in commits:
                sha = commit["sha"]
                if sha == stop_sha:
                    found_stop = True
                    break
                if sha not in seen_shas:
                    seen_shas.add(sha)
                    walk_commits.append(commit)

            print(f"  Walk page {walk_page}: found {len(walk_commits)} additional commits")
            walk_page += 1

        # Combine: Compare API commits first (they're in order), then walk commits
        # Actually, we should return all unique commits
        all_commits.extend(walk_commits)
        print(f"  Total after walk: {len(all_commits)} commits")

    # For diverged branches, filter out commits whose PRs are already in base release
    # This handles cherry-picks that exist in both releases
    if is_diverged:
        print(f"\n  Filtering out PRs already in {base_tag}...")

        # Get base release commits to extract PR numbers
        print(f"  Fetching {base_tag} commits...")
        base_commits = []
        base_page = 1
        while True:
            response = requests.get(
                f"{base_url}/commits",
                headers=headers,
                params={"sha": base_sha, "per_page": per_page, "page": base_page},
            )
            if response.status_code != 200:
                break
            commits = response.json()
            if not commits:
                break

            for commit in commits:
                if merge_base and commit["sha"] == merge_base:
                    break
                base_commits.append(commit)
            else:
                base_page += 1
                continue
            break

        print(f"  Found {len(base_commits)} commits in {base_tag}")

        # Extract PR numbers from base commits
        base_pr_numbers = set()
        for commit in base_commits:
            message = commit.get("commit", {}).get("message", "")
            pr_num = extract_pr_number(message)
            if pr_num:
                base_pr_numbers.add(pr_num)
        print(f"  Found {len(base_pr_numbers)} unique PRs in {base_tag}")

        # Filter out commits whose PR is already in base
        filtered_commits = []
        skipped_count = 0
        for commit in all_commits:
            message = commit.get("commit", {}).get("message", "")
            pr_num = extract_pr_number(message)
            if pr_num and pr_num in base_pr_numbers:
                skipped_count += 1
                continue
            filtered_commits.append(commit)

        print(f"  Skipped {skipped_count} commits (PRs already in {base_tag})")
        print(f"  Final count: {len(filtered_commits)} new commits in {head_tag}")
        return filtered_commits

    return all_commits


def extract_contributors(commits: list[dict]) -> dict:
    """
    Extract unique contributors from commits.

    Returns a dict with:
        - contributors: set of (login, name) tuples
        - by_login: dict mapping login -> contributor info
        - by_email: dict mapping email -> contributor info (for commits without GitHub user)
    """
    contributors_by_login = {}
    contributors_by_email = {}

    for commit in commits:
        # Try to get GitHub user info first (author field)
        author = commit.get("author")
        if author and author.get("login"):
            login = author["login"]
            if login not in contributors_by_login:
                contributors_by_login[login] = {
                    "login": login,
                    "name": commit.get("commit", {}).get("author", {}).get("name", ""),
                    "email": commit.get("commit", {}).get("author", {}).get("email", ""),
                    "avatar_url": author.get("avatar_url", ""),
                    "html_url": author.get("html_url", ""),
                    "commits": 0,
                }
            contributors_by_login[login]["commits"] += 1
        else:
            # Fallback to git author info
            git_author = commit.get("commit", {}).get("author", {})
            email = git_author.get("email", "")
            name = git_author.get("name", "")

            if email and email not in contributors_by_email:
                contributors_by_email[email] = {
                    "login": None,
                    "name": name,
                    "email": email,
                    "avatar_url": "",
                    "html_url": "",
                    "commits": 0,
                }
            if email:
                contributors_by_email[email]["commits"] += 1

    return {
        "by_login": contributors_by_login,
        "by_email": contributors_by_email,
        "total": len(contributors_by_login) + len(contributors_by_email),
    }


def get_tag_date(base_url: str, tag: str, headers: dict) -> str:
    """Get the date of a tag's commit."""
    # First resolve the tag to a commit SHA
    tag_resp = requests.get(f"{base_url}/git/refs/tags/{tag}", headers=headers)
    if tag_resp.status_code != 200:
        return None

    tag_data = tag_resp.json()
    sha = tag_data["object"]["sha"]

    # If it's an annotated tag, get the underlying commit
    if tag_data["object"]["type"] == "tag":
        tag_obj_resp = requests.get(f"{base_url}/git/tags/{sha}", headers=headers)
        if tag_obj_resp.status_code == 200:
            sha = tag_obj_resp.json()["object"]["sha"]

    # Get the commit date
    commit_resp = requests.get(f"{base_url}/commits/{sha}", headers=headers)
    if commit_resp.status_code == 200:
        return commit_resp.json()["commit"]["committer"]["date"]

    return None


def check_contributor_is_new(owner: str, repo: str, login: str, before_date: str, headers: dict) -> bool:
    """
    Check if a contributor has any commits before a given date.

    Returns True if this is their first contribution (no commits before the date).
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Search for commits by this author before the base tag date
    response = requests.get(
        f"{base_url}/commits",
        headers=headers,
        params={"author": login, "until": before_date, "per_page": 1},
    )

    if response.status_code == 200:
        commits = response.json()
        # If no commits found before the date, they're a new contributor
        return len(commits) == 0

    return False


def find_first_contribution(commits: list[dict], login: str) -> dict | None:
    """
    Find the first (earliest) contribution by a user in the commit list.

    Returns the commit dict or None.
    """
    user_commits = []
    for commit in commits:
        author = commit.get("author")
        if author and author.get("login") == login:
            user_commits.append(commit)

    # Commits are usually newest first, so reverse to get oldest first
    if user_commits:
        return user_commits[-1]  # Last one is the oldest/first contribution
    return None


def calculate_new_contributors_via_generate_notes(
    owner: str,
    repo: str,
    base_tag: str,
    head_tag: str,
    token: str | None = None,
) -> list[dict]:
    """
    Calculate new contributors using GitHub's generate-notes API.

    This is more accurate than checking commit history because GitHub
    tracks contributor status internally.

    Args:
        owner: Repository owner
        repo: Repository name
        base_tag: The base tag (older version)
        head_tag: The head tag (newer version)
        token: GitHub token

    Returns:
        List of new contributor info dicts with login and first_pr fields
    """
    import re
    import subprocess

    print("\nGetting new contributors via GitHub generate-notes API...")

    # Use gh CLI to call the generate-notes API
    cmd = [
        "gh",
        "api",
        f"repos/{owner}/{repo}/releases/generate-notes",
        "-f",
        f"tag_name={head_tag}",
        "-f",
        f"target_commitish={head_tag}",
        "-f",
        f"previous_tag_name={base_tag}",
        "--jq",
        ".body",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  Warning: gh CLI failed: {result.stderr}")
            return []

        body = result.stdout

        # Parse new contributors from the generated notes
        # Format: "* @username made their first contribution in https://github.com/owner/repo/pull/12345"
        pattern = r"\* @(\S+) made their first contribution in https://github\.com/[^/]+/[^/]+/pull/(\d+)"
        matches = re.findall(pattern, body)

        new_contributors = []
        for login, pr_number in matches:
            new_contributors.append(
                {
                    "login": login,
                    "first_pr": pr_number,
                }
            )

        print(f"  Found {len(new_contributors)} new contributors")
        return new_contributors

    except subprocess.TimeoutExpired:
        print("  Warning: gh CLI timed out")
        return []
    except FileNotFoundError:
        print("  Warning: gh CLI not found, falling back to legacy method")
        return []


def calculate_new_contributors(
    commits: list[dict],
    current_contributors: dict,
    owner: str,
    repo: str,
    base_tag: str,
    head_tag: str = "",
    token: str | None = None,
) -> list[dict]:
    """
    Calculate which contributors are new (first-time) in this release.

    First tries GitHub's generate-notes API (more accurate), then falls back
    to checking commit history if that fails.

    Args:
        commits: List of commits in the current release
        current_contributors: Output from extract_contributors()
        owner: Repository owner
        repo: Repository name
        base_tag: The base tag (older version)
        head_tag: The head tag (newer version)
        token: GitHub token

    Returns:
        List of new contributor info dicts with first_pr field
    """
    # Try the accurate method first (via generate-notes API)
    if head_tag:
        new_contributors = calculate_new_contributors_via_generate_notes(
            owner=owner,
            repo=repo,
            base_tag=base_tag,
            head_tag=head_tag,
            token=token,
        )
        if new_contributors:
            return new_contributors

    # Fall back to legacy method (checking commit history)
    print("\nFalling back to legacy new contributor detection...")

    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Get the date of the base tag
    print("Getting base tag date...")
    base_date = get_tag_date(base_url, base_tag, headers)
    if not base_date:
        print(f"  Warning: Could not get date for tag {base_tag}")
        return []

    print(f"  Base tag date: {base_date}")

    new_contributors = []
    logins = list(current_contributors["by_login"].keys())
    total = len(logins)

    print(f"\nChecking {total} contributors for first-time status...")

    for i, login in enumerate(logins):
        if (i + 1) % 20 == 0:
            print(f"  Checked {i + 1}/{total} contributors...")

        is_new = check_contributor_is_new(owner, repo, login, base_date, headers)

        if is_new:
            info = current_contributors["by_login"][login].copy()

            # Find their first PR in this release
            first_commit = find_first_contribution(commits, login)
            if first_commit:
                message = first_commit.get("commit", {}).get("message", "")
                pr_number = extract_pr_number(message)
                info["first_pr"] = pr_number
                info["first_commit_sha"] = first_commit.get("sha", "")[:8]

            new_contributors.append(info)

    print(f"  Found {len(new_contributors)} new contributors (legacy method)")

    return new_contributors


def generate_contributor_stats(
    commits: list[dict],
    owner: str,
    repo: str,
    base_tag: str,
    head_tag: str,
    token: str | None = None,
    check_new: bool = True,
) -> dict:
    """
    Generate contributor statistics for the release.

    Returns a dict with all statistics data.
    """
    print("\n" + "=" * 60)
    print("CONTRIBUTOR STATISTICS")
    print("=" * 60)

    # Extract contributors from current commits
    contributors = extract_contributors(commits)

    print(f"\nTotal commits: {len(commits)}")
    print(f"Total contributors: {contributors['total']}")
    print(f"  - With GitHub account: {len(contributors['by_login'])}")
    print(f"  - Without GitHub account (by email): {len(contributors['by_email'])}")

    new_count = 0
    new_contributors_list = []

    if check_new:
        # Calculate new contributors (tries GitHub generate-notes API first, then falls back to commit history)
        new_contributors_list = calculate_new_contributors(
            commits=commits,
            current_contributors=contributors,
            owner=owner,
            repo=repo,
            base_tag=base_tag,
            head_tag=head_tag,
            token=token,
        )
        new_count = len(new_contributors_list)

        print(f"\nNew contributors (first-time): {new_count}")

        if new_contributors_list:
            print("\nNew contributors list:")
            for c in sorted(new_contributors_list, key=lambda x: x["login"].lower()):
                pr_info = f" in #{c['first_pr']}" if c.get("first_pr") else ""
                print(f"  - @{c['login']} made their first contribution{pr_info}")

    # Print summary line for release notes
    print("\n" + "-" * 60)
    print("RELEASE NOTES SUMMARY LINE:")
    print("-" * 60)
    if check_new:
        summary_line = (
            f"This release features {len(commits)} commits from {contributors['total']} contributors ({new_count} new)!"
        )
    else:
        summary_line = f"This release features {len(commits)} commits from {contributors['total']} contributors!"
    print(summary_line)
    print("-" * 60)

    # Get all contributors sorted by commit count
    all_contributors_list = list(contributors["by_login"].values()) + list(contributors["by_email"].values())
    sorted_contributors = sorted(all_contributors_list, key=lambda x: x["commits"], reverse=True)

    # Print top contributors
    print("\nTop contributors by commit count:")
    for i, c in enumerate(sorted_contributors[:20], 1):
        if c.get("login"):
            print(f"  {i:2}. @{c['login']:20} - {c['commits']:3} commits")
        else:
            print(f"  {i:2}. {c['name']:20} - {c['commits']:3} commits (no GitHub account)")

    return {
        "total_commits": len(commits),
        "total_contributors": contributors["total"],
        "new_contributors": new_count if check_new else None,
        "new_contributors_list": new_contributors_list,
        "contributors": contributors,
        "sorted_contributors": sorted_contributors,
        "summary_line": summary_line,
        "base_tag": base_tag,
        "head_tag": head_tag,
        "owner": owner,
        "repo": repo,
    }


def save_contributor_stats(stats: dict, output_file: str, owner: str, repo: str):
    """
    Save contributor statistics to a markdown file.

    Args:
        stats: Statistics dict from generate_contributor_stats()
        output_file: Output file path
        owner: Repository owner
        repo: Repository name
    """
    lines = []

    # Header
    lines.append(f"# Contributor Statistics: {stats['base_tag']} → {stats['head_tag']}")
    lines.append("")

    # Summary for release notes
    lines.append("## Release Notes Summary")
    lines.append("")
    lines.append(f"> {stats['summary_line']}")
    lines.append("")

    # Overview stats
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total Commits**: {stats['total_commits']}")
    lines.append(f"- **Total Contributors**: {stats['total_contributors']}")
    if stats["new_contributors"] is not None:
        lines.append(f"- **New Contributors**: {stats['new_contributors']}")
    lines.append("")

    # Top contributors table
    lines.append("## Top Contributors")
    lines.append("")
    lines.append("| Rank | Contributor | Commits |")
    lines.append("|------|-------------|---------|")

    for i, c in enumerate(stats["sorted_contributors"][:30], 1):
        if c.get("login"):
            contributor_link = f"[@{c['login']}](https://github.com/{c['login']})"
        else:
            contributor_link = c["name"]
        lines.append(f"| {i} | {contributor_link} | {c['commits']} |")

    lines.append("")

    # New contributors section
    if stats["new_contributors_list"]:
        lines.append("## New Contributors 🎉")
        lines.append("")

        sorted_new = sorted(stats["new_contributors_list"], key=lambda x: x["login"].lower())
        for c in sorted_new:
            pr_num = c.get("first_pr")
            if pr_num:
                pr_link = f"https://github.com/{owner}/{repo}/pull/{pr_num}"
                lines.append(f"* @{c['login']} made their first contribution in {pr_link}")
            else:
                lines.append(f"* @{c['login']} made their first contribution")
        lines.append("")

    # All contributors section (collapsed)
    lines.append("## All Contributors")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand full list</summary>")
    lines.append("")
    lines.append("| Contributor | Commits |")
    lines.append("|-------------|---------|")

    for c in stats["sorted_contributors"]:
        if c.get("login"):
            contributor_link = f"[@{c['login']}](https://github.com/{c['login']})"
        else:
            contributor_link = c["name"]
        lines.append(f"| {contributor_link} | {c['commits']} |")

    lines.append("")
    lines.append("</details>")
    lines.append("")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved contributor statistics to {output_file}")


def extract_pr_number(message: str) -> str | None:
    """Extract PR number from commit message."""
    # Common patterns: (#12345), (https://github.com/.../pull/12345)
    patterns = [
        r"\(#(\d+)\)",  # (#12345)
        r"pull/(\d+)",  # https://github.com/.../pull/12345
        r"#(\d+)$",  # #12345 at end
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return None


def format_commit_message(
    commit: dict,
    owner: str,
    repo: str,
    include_sha: bool = False,
    include_date: bool = False,
) -> str:
    """
    Format a commit message for the output file.

    Format: [Category] Description in https://github.com/owner/repo/pull/XXXX
    or: [Category] Description (#XXXX)

    If include_sha is True, prepends the full SHA: `sha` Message (#XXXX)
    If include_date is True, prepends the date: [YYYY-MM-DD] Message (#XXXX)
    """
    message = commit["commit"]["message"]
    sha = commit.get("sha", "")

    # Get commit date (use committer date for when it was merged)
    commit_date = ""
    if include_date:
        date_str = commit.get("commit", {}).get("committer", {}).get("date", "")
        if date_str:
            # Parse ISO format and extract date part (YYYY-MM-DD)
            commit_date = date_str[:10]

    # Get the first line of the commit message
    first_line = message.split("\n")[0].strip()

    # Extract PR number if present
    pr_number = extract_pr_number(first_line)

    # Clean up the message - remove existing PR references for reformatting
    clean_message = first_line
    clean_message = re.sub(r"\s*\(#\d+\)\s*$", "", clean_message)
    clean_message = re.sub(r"\s*https://github\.com/[^/]+/[^/]+/pull/\d+\s*", "", clean_message)
    clean_message = re.sub(r"\s+in\s*$", "", clean_message)
    clean_message = clean_message.strip()

    # Format output
    if pr_number:
        # Check if message already contains the full URL pattern
        if f"https://github.com/{owner}/{repo}/pull/" in first_line:
            formatted = first_line
        else:
            formatted = f"{clean_message} (#{pr_number})"
    else:
        formatted = clean_message

    # Prepend metadata if requested
    prefix_parts = []
    if include_date and commit_date:
        prefix_parts.append(f"[{commit_date}]")
    if include_sha and sha:
        prefix_parts.append(f"`{sha}`")

    if prefix_parts:
        formatted = f"{' '.join(prefix_parts)} {formatted}"

    return formatted


def save_commits_to_file(
    commits: list[dict],
    output_file: str,
    owner: str,
    repo: str,
    sort_mode: str = "chronological",
    include_sha: bool = False,
    include_date: bool = False,
):
    """
    Save formatted commits to a markdown file.

    Args:
        commits: List of commit dictionaries
        output_file: Output file path
        owner: Repository owner
        repo: Repository name
        sort_mode: "chronological" (newest first, like GitHub),
                   "alphabetical" (by commit message),
                   "reverse" (oldest first)
        include_sha: If True, include full commit SHA in output
        include_date: If True, include commit date in output
    """
    print(f"\nFormatting and saving {len(commits)} commits to {output_file}...")

    formatted_lines = []
    for commit in commits:
        formatted = format_commit_message(commit, owner, repo, include_sha=include_sha, include_date=include_date)
        formatted_lines.append(formatted)

    # Sort based on mode
    if sort_mode == "alphabetical":
        formatted_lines.sort(key=lambda x: x.lower())
        print("  Sorted alphabetically by commit message")
    elif sort_mode == "reverse":
        formatted_lines.reverse()
        print("  Sorted chronologically (oldest first)")
    else:
        # chronological - keep original order (newest first, as returned by API)
        print("  Keeping chronological order (newest first)")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    print(f"Saved {len(formatted_lines)} commits to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch commits between two GitHub tags or between a tag and a commit")
    parser.add_argument(
        "--owner",
        default="vllm-project",
        help="Repository owner (default: vllm-project)",
    )
    parser.add_argument("--repo", default="vllm", help="Repository name (default: vllm)")
    parser.add_argument(
        "--base-tag",
        help="Base tag (older, e.g., v0.11.2). If not provided with --head-commit, will auto-detect previous tag.",
    )
    parser.add_argument(
        "--head-tag",
        help="Head tag (newer, e.g., v0.12.0). Use this OR --head-commit. If neither specified, uses "
        "HEAD of default branch.",
    )
    parser.add_argument(
        "--head-commit",
        help="Head commit SHA (can be short or full). If not specified and no --head-tag, uses HEAD of default branch.",
    )
    parser.add_argument(
        "--tag-pattern",
        default=r"^v\d+\.\d+\.\d+$",
        help="Regex pattern to filter tags when auto-detecting previous tag (default: ^v\\d+\\.\\d+\\.\\d+$)",
    )
    parser.add_argument(
        "--output",
        default="0-current-raw-commits.md",
        help="Output file (default: 0-current-raw-commits.md)",
    )
    parser.add_argument("--token", help="GitHub token (or set GITHUB_TOKEN env var)")
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slower but more thorough commit-by-commit fetching",
    )
    parser.add_argument(
        "--sort",
        choices=["chronological", "alphabetical", "reverse"],
        default="chronological",
        help="Sort mode: chronological (newest first, like GitHub), alphabetical (by message), reverse (oldest first)",
    )
    parser.add_argument("--stats", action="store_true", help="Generate and save contributor statistics")
    parser.add_argument(
        "--stats-output",
        default="0-contributor-stats.md",
        help="Output file for contributor statistics (default: 0-contributor-stats.md)",
    )
    parser.add_argument(
        "--no-new-check",
        action="store_true",
        help="Skip checking for new contributors (faster, avoids extra API calls)",
    )
    parser.add_argument(
        "--include-sha",
        action="store_true",
        help="Include full commit SHA in output (format: `sha` message)",
    )
    parser.add_argument(
        "--include-date",
        action="store_true",
        help="Include commit date in output (format: [YYYY-MM-DD] message)",
    )
    parser.add_argument(
        "--since",
        help="Fetch commits since this date (ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ). "
        "Use with --until for date range mode.",
    )
    parser.add_argument(
        "--until",
        help="Fetch commits until this date (ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ). "
        "Use with --since for date range mode.",
    )
    parser.add_argument(
        "--branch",
        help="Branch to fetch commits from (only used with --since/--until date range mode)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.head_tag and args.head_commit:
        parser.error("Cannot specify both --head-tag and --head-commit")

    # Check for date range mode
    date_range_mode = args.since is not None or args.until is not None
    if date_range_mode:
        if not args.since or not args.until:
            parser.error("Both --since and --until must be specified for date range mode")
        if args.head_tag or args.head_commit or args.base_tag:
            parser.error("Cannot use --since/--until with --head-tag, --head-commit, or --base-tag")

    token = args.token or get_github_token()

    if not token:
        print("Warning: No GitHub token provided. Rate limits will be stricter.")
        print("Set GITHUB_TOKEN environment variable or use --token argument.")
        print()

    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{args.owner}/{args.repo}"

    try:
        # Date range mode
        if date_range_mode:
            print(f"\n{'=' * 60}")
            print(f"Fetching commits by date range: {args.since} → {args.until}")
            if args.branch:
                print(f"Branch: {args.branch}")
            print(f"{'=' * 60}")

            commits = fetch_commits_by_date_range(
                owner=args.owner,
                repo=args.repo,
                since=args.since,
                until=args.until,
                token=token,
                branch=args.branch,
            )

            print(f"\nTotal commits found: {len(commits)}")

            save_commits_to_file(
                commits=commits,
                output_file=args.output,
                owner=args.owner,
                repo=args.repo,
                sort_mode=args.sort,
                include_sha=args.include_sha,
                include_date=args.include_date,
            )

            # Stats not fully supported in date range mode (no base_tag for new contributor check)
            if args.stats:
                print("\nNote: Contributor statistics in date range mode won't check for new contributors.")
                stats = generate_contributor_stats(
                    commits=commits,
                    owner=args.owner,
                    repo=args.repo,
                    base_tag=args.since,
                    head_tag=args.until,
                    token=token,
                    check_new=False,  # Can't check new contributors without a base tag
                )
                save_contributor_stats(
                    stats=stats,
                    output_file=args.stats_output,
                    owner=args.owner,
                    repo=args.repo,
                )

            return

        # Tag/commit mode (existing logic)
        # Determine head reference
        head_is_commit = False
        head_ref = None
        head_display_name = None

        if args.head_tag:
            head_ref = args.head_tag
            head_is_commit = False
            head_display_name = args.head_tag
        elif args.head_commit:
            head_ref = args.head_commit
            head_is_commit = True
            head_display_name = args.head_commit[:8] if len(args.head_commit) > 8 else args.head_commit
        else:
            # Auto-detect HEAD of default branch
            branch_name, head_sha = get_default_branch_head(base_url, headers)
            head_ref = head_sha
            head_is_commit = True
            head_display_name = f"{branch_name} ({head_sha[:8]})"

        base_tag = args.base_tag
        base_is_commit = False

        # Auto-detect previous tag if needed
        if not base_tag and head_is_commit:
            print("Auto-detecting previous tag...")
            head_sha = resolve_commit_sha(base_url, head_ref, headers)

            result = find_previous_tag(
                base_url=base_url,
                head_sha=head_sha,
                headers=headers,
                tag_pattern=args.tag_pattern,
            )

            if result is None:
                raise Exception("Could not find a previous tag. Please specify --base-tag manually.")

            base_tag, _ = result
            print(f"\nUsing auto-detected base tag: {base_tag}")
        elif not base_tag:
            parser.error("Must specify --base-tag when using --head-tag")

        print(f"\n{'=' * 60}")
        print(f"Fetching commits: {base_tag} → {head_display_name}")
        print(f"{'=' * 60}")

        if args.slow:
            # Note: slow mode doesn't support commit SHA yet, only tags
            if head_is_commit:
                print("Warning: --slow mode with --head-commit not fully supported, using fast mode")
            commits = fetch_commits_between_tags_fast(
                owner=args.owner,
                repo=args.repo,
                base_tag=base_tag,
                head_tag=head_ref,
                token=token,
                head_is_commit=head_is_commit,
                base_is_commit=base_is_commit,
            )
        else:
            commits = fetch_commits_between_tags_fast(
                owner=args.owner,
                repo=args.repo,
                base_tag=base_tag,
                head_tag=head_ref,
                token=token,
                head_is_commit=head_is_commit,
                base_is_commit=base_is_commit,
            )

        print(f"\nTotal commits found: {len(commits)}")

        save_commits_to_file(
            commits=commits,
            output_file=args.output,
            owner=args.owner,
            repo=args.repo,
            sort_mode=args.sort,
            include_sha=args.include_sha,
            include_date=args.include_date,
        )

        # Generate and save contributor statistics if requested
        if args.stats:
            stats = generate_contributor_stats(
                commits=commits,
                owner=args.owner,
                repo=args.repo,
                base_tag=base_tag,
                head_tag=head_display_name,
                token=token,
                check_new=not args.no_new_check,
            )
            save_contributor_stats(
                stats=stats,
                output_file=args.stats_output,
                owner=args.owner,
                repo=args.repo,
            )

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
