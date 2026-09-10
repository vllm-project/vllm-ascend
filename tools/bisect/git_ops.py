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
import subprocess
from pathlib import Path

import regex as re

from tools.bisect.config import Candidate

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


def _try_rev_parse(repo: Path, ref: str) -> str | None:
    """Return the full sha for ``ref`` if it resolves locally, else None."""
    try:
        return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except GitError:
        return None


def _is_shallow(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--is-shallow-repository", check=False).strip() == "true"


def _looks_like_sha(ref: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", ref))


def resolve_commit(repo: Path, ref: str) -> str:
    """Resolve a ref / short-sha / full-sha / PR number to a full commit sha.

    Robust against the nightly container's ``git clone --depth 1`` shallow clone:
    a good commit read from the status table is a full sha that is usually *not*
    in the shallow local history, and ``git fetch origin <sha>`` is refused by
    most servers (arbitrary-sha fetch is disabled). So we progressively recover
    history: PR-head fetch -> direct ref fetch -> ``--unshallow`` -> full
    all-branch fetch, re-checking after each step. Every step is logged so a
    failure shows exactly what was attempted.
    """
    # 0) Already present locally?
    sha = _try_rev_parse(repo, ref)
    if sha:
        return sha

    is_sha = _looks_like_sha(ref)
    pr = ref.lstrip("#")
    attempts: list[str] = []

    # 1) Bare PR number -> fetch its head ref.
    if pr.isdigit():
        logger.info("Ref %s not local; fetching PR head refs/pull/%s/head", ref, pr)
        _git(repo, "fetch", "--quiet", "origin", f"refs/pull/{pr}/head:refs/bisect_tmp/pr_{pr}", check=False)
        sha = _try_rev_parse(repo, f"refs/bisect_tmp/pr_{pr}")
        if sha:
            return sha
        attempts.append(f"fetch refs/pull/{pr}/head")

    # 2) Shallow clone -> deepen to FULL history FIRST. This is both the reliable
    #    way to get a good-table sha (it lives on origin's mainline history) and
    #    -- crucially -- it keeps the history between good..bad complete. We must
    #    NOT let a by-sha fetch (step 3) succeed first on a shallow repo: that
    #    would add only the single commit behind a new shallow boundary, leaving
    #    intermediate commits missing so `git log good..bad` would silently
    #    return a truncated candidate list and break the bisect.
    if _is_shallow(repo):
        logger.info(
            "Repo is a shallow clone; running 'git fetch --unshallow' to recover full history (this can take a while)"
        )
        _git(repo, "fetch", "--unshallow", "--quiet", "origin", check=False)
        sha = _try_rev_parse(repo, ref)
        if sha:
            return sha
        attempts.append("fetch --unshallow origin")

    # 3) Direct fetch by ref name (a branch/tag not yet tracked locally; safe now
    #    that any shallow repo has been fully unshallowed above).
    logger.info("Trying direct fetch of %r from origin", ref)
    _git(repo, "fetch", "--quiet", "origin", ref, check=False)
    sha = _try_rev_parse(repo, ref) or (_try_rev_parse(repo, "FETCH_HEAD") if not is_sha else None)
    if sha:
        return sha
    attempts.append(f"fetch origin {ref}")

    # 4) Last resort: fetch full history of all branches and retry.
    logger.info("Fetching all branches with full history as a last resort")
    _git(repo, "fetch", "--quiet", "--tags", "origin", "+refs/heads/*:refs/remotes/origin/*", check=False)
    sha = _try_rev_parse(repo, ref)
    if sha:
        return sha
    attempts.append("fetch all branches")

    raise GitError(
        f"Could not resolve ref {ref!r} after: {', '.join(attempts)}. "
        f"(shallow={_is_shallow(repo)}, looks_like_sha={is_sha}). "
        "If this is a full sha from the good table, ensure 'origin' points at a "
        "remote that contains it and that the network allows fetching."
    )


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
    # Guard: a shallow repo can make `git log good..bad` silently truncate the
    # candidate list (history cut at the shallow boundary), which would corrupt
    # the search. resolve_commit() unshallows when needed, but verify here so we
    # fail loudly rather than bisect a partial range.
    if _is_shallow(repo):
        raise GitError(
            "Repository is still a shallow clone; the good..bad history may be "
            "incomplete and the candidate list could be truncated. Run "
            "'git fetch --unshallow' (or re-run; resolve_commit normally does "
            "this) before bisecting."
        )

    if not is_ancestor(repo, good, bad):
        raise GitError(f"good ({good[:12]}) is not an ancestor of bad ({bad[:12]}); the bisect range is invalid")
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
        candidates.append(Candidate(commit=sha, pr_number=_parse_pr(subject), subject=subject))
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


def file_at_commit(repo: Path, commit: str, rel_path: str) -> str | None:
    """Return the contents of ``rel_path`` at ``commit`` without checking out.

    Uses ``git show <commit>:<path>``; returns None if the file is absent there.
    """
    out = _git(repo, "show", f"{commit}:{rel_path}", check=False)
    return out if out else None


def commit_changed_files(repo: Path, commit: str) -> list[str]:
    """Files changed by a single ``commit`` relative to its first parent.

    This is the "what did this PR touch" view (``git show --name-only``). For a
    root commit with no parent it lists the full tree.
    """
    out = _git(repo, "show", "--first-parent", "--name-only", "--format=", commit)
    return [line.strip() for line in out.splitlines() if line.strip()]


def matches_any(files: list[str], globs: tuple[str, ...]) -> list[str]:
    """Return the subset of ``files`` matching any glob in ``globs``."""
    hits: list[str] = []
    for f in files:
        if any(fnmatch.fnmatch(f, g) for g in globs):
            hits.append(f)
    return hits
