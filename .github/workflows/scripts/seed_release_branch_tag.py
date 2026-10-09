# SPDX-License-Identifier: Apache-2.0
"""Give a new release branch a native setuptools-scm version baseline."""

import argparse
import subprocess
from pathlib import Path

import regex as re


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()


def release_version(branch: str) -> str:
    component = r"(?:0|[1-9][0-9]*)"
    match = re.fullmatch(rf"releases/v({component}\.{component}\.{component})(?:rc)?", branch)
    if match is None:
        raise ValueError("Expected a release branch named releases/vX.Y.Z or releases/vX.Y.Zrc")
    return match.group(1)


def existing_version_tag(repo: Path, version: str, branch_ref: str) -> str | None:
    # Read remote refs so an unpublished local tag cannot hide a missing baseline.
    pattern = re.compile(rf"v{re.escape(version)}(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?")
    remote_tags = git(repo, "ls-remote", "--tags", "--refs", "origin").splitlines()
    conflicting = []
    for entry in remote_tags:
        _, ref = entry.split()
        tag = ref.removeprefix("refs/tags/")
        if not pattern.fullmatch(tag):
            continue
        git(repo, "fetch", "--no-tags", "origin", ref)
        tagged_commit = git(repo, "rev-parse", "FETCH_HEAD^{commit}")
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", tagged_commit, branch_ref], cwd=repo, capture_output=True
        )
        if result.returncode == 0:
            return tag
        if result.returncode != 1:
            raise RuntimeError("Could not validate the existing version tag's ancestry")
        conflicting.append(tag)
    if conflicting:
        raise ValueError(f"Version tags exist outside the release branch history: {', '.join(conflicting)}")
    return None


def seed_release_tag(repo: Path, branch: str, expected_sha: str | None = None) -> str:
    version = release_version(branch)
    if expected_sha is not None and re.fullmatch(r"[0-9a-fA-F]{40}", expected_sha) is None:
        raise ValueError("The creation commit must be a full SHA-1 commit ID")
    branch_ref = f"refs/remotes/origin/{branch}"
    git(repo, "fetch", "--no-tags", "origin", f"+refs/heads/{branch}:{branch_ref}")
    tip = git(repo, "rev-parse", f"{branch_ref}^{{commit}}")
    target = expected_sha.lower() if expected_sha is not None else tip
    result = subprocess.run(["git", "merge-base", "--is-ancestor", target, tip], cwd=repo, capture_output=True)
    if result.returncode != 0:
        raise ValueError("The creation commit is not in the current release branch history")

    existing = existing_version_tag(repo, version, branch_ref)
    if existing is not None:
        return f"Already versioned by {existing}"
    tag = f"v{version}rc0"
    try:
        # Push the commit directly: a failed push leaves no misleading local tag.
        git(repo, "push", "origin", f"{target}:refs/tags/{tag}")
    except subprocess.CalledProcessError:
        # Another branch-creation/dispatch job may have seeded the same family.
        existing = existing_version_tag(repo, version, branch_ref)
        if existing is None:
            raise
        return f"Already versioned by {existing}"
    return f"Created {tag} at {target}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--creation-sha")
    args = parser.parse_args()
    try:
        print(seed_release_tag(Path.cwd(), args.branch, args.creation_sha))
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            parser.exit(1, exc.stderr)
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
