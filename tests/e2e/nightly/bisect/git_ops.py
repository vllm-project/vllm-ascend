# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
"""Git helpers: resolve commits, build the candidate list, checkout, diff."""

import fnmatch
import logging
import re
import subprocess
from pathlib import Path

from tests.e2e.nightly.bisect.config import Candidate

logger = logging.getLogger(__name__)

# Matches the "(#12345)" trailer that squash-merged PRs leave in the subject.
_PR_RE = re.compile(r"\(#(\d+)\)\s*$")


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in ``repo`` and return stripped stdout."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def resolve_commit(repo: Path, ref: str) -> str:
    """Resolve any ref/short-sha/``pull/NNN/head`` style ref to a full sha.

    Falls back to fetching the ref (PR head or branch) when it is not present
    locally, mirroring ``run.sh::checkout_src``.
    """
    try:
        return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except GitError:
        pass

    # Maybe it's a bare PR number -> fetch refs/pull/<n>/head.
    pr = ref.lstrip("#")
    fetch_ref = ref
    if pr.isdigit():
        fetch_ref = f"pull/{pr}/head"
    logger.info("Ref %s not local; fetching %s", ref, fetch_ref)
    _git(repo, "fetch", "--quiet", "origin", f"refs/{fetch_ref}:refs/bisect_tmp/{pr}", check=False)
    try:
        return _git(repo, "rev-parse", "--verify", f"refs/bisect_tmp/{pr}^{{commit}}")
    except GitError as exc:
        raise GitError(f"Could not resolve ref {ref!r}") from exc


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """True if ``ancestor`` is reachable from ``descendant`` (good before bad)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _parse_pr(subject: str) -> str | None:
    m = _PR_RE.search(subject)
    return m.group(1) if m else None


def candidate_list(repo: Path, good: str, bad: str) -> list[Candidate]:
    """Commits in ``(good, bad]`` along the first-parent mainline, oldest first.

    ``good`` itself is excluded (it is the known-good baseline). ``bad`` is the
    last element. The returned list is the bisect search space, one commit per
    element (commit-atomic bisect).
    """
    if not is_ancestor(repo, good, bad):
        raise GitError(
            f"good ({good[:12]}) is not an ancestor of bad ({bad[:12]}); "
            "the bisect range is invalid"
        )
    # --first-parent keeps us on the mainline so a single PR == a single commit,
    # avoiding expansion of intra-PR commits from merge-style histories.
    # \x1f (unit separator) safely splits sha from subject.
    raw = _git(
        repo,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x1f%s",
        f"{good}..{bad}",
    )
    candidates: list[Candidate] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "\x1f" not in line:
            continue
        sha, subject = line.split("\x1f", 1)
        candidates.append(
            Candidate(commit=sha, pr_number=_parse_pr(subject), subject=subject)
        )
    if not candidates:
        raise GitError("Empty candidate list; good and bad may be identical")
    logger.info("Built %d candidate commits between good and bad", len(candidates))
    return candidates


def describe(repo: Path, commit: str) -> Candidate:
    """Build a Candidate (sha + PR + subject) for a single commit."""
    sha = resolve_commit(repo, commit)
    subject = _git(repo, "log", "-1", "--format=%s", sha)
    return Candidate(commit=sha, pr_number=_parse_pr(subject), subject=subject)


def checkout(repo: Path, commit: str) -> None:
    """Detached checkout of ``commit`` (discarding tracked-file changes)."""
    _git(repo, "checkout", "--force", "--detach", commit)
    logger.info("Checked out %s", commit[:12])


def current_commit(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def changed_files(repo: Path, base: str, target: str) -> list[str]:
    """Files changed between ``base`` and ``target`` (both inclusive of range)."""
    out = _git(repo, "diff", "--name-only", base, target)
    return [line.strip() for line in out.splitlines() if line.strip()]


def matches_any(files: list[str], globs: tuple[str, ...]) -> list[str]:
    """Return the subset of ``files`` matching any glob in ``globs``."""
    hits: list[str] = []
    for f in files:
        if any(fnmatch.fnmatch(f, g) for g in globs):
            hits.append(f)
    return hits
