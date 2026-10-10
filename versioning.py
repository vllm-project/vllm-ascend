# SPDX-License-Identifier: Apache-2.0
"""Build-time version selection for release branches."""

import subprocess
from dataclasses import replace
from functools import partial
from inspect import signature
from typing import Any

from packaging.version import InvalidVersion, Version
from setuptools_scm import get_version
from setuptools_scm.git import DEFAULT_DESCRIBE
from setuptools_scm.version import ScmVersion, guess_next_dev_version


def _git(root: str, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


def _release_version(branch: str | None) -> str | None:
    if branch is None or not branch.startswith("releases/v"):
        return None
    version = branch.removeprefix("releases/v").removesuffix("rc")
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isascii() or not part.isdecimal() or str(int(part)) != part for part in parts):
        return None
    return version


def _version_tags(root: str) -> dict[str, Version]:
    tags = {}
    # Tags on unrelated histories cannot supply this checkout's version.
    for tag in _git(root, "for-each-ref", "--merged=HEAD", "--format=%(refname:strip=2)", "refs/tags").splitlines():
        if not tag.startswith("v"):
            continue
        try:
            version = Version(tag[1:])
        except InvalidVersion:
            continue
        if version.local is None:
            tags[tag] = version
    return tags


def _closest_tag(root: str, tags: dict[str, Version]) -> str:
    matches = [argument for tag in tags for argument in ("--match", tag)]
    closest = _git(root, "describe", "--tags", "--abbrev=0", f"--candidates={len(tags)}", *matches)
    commit = _git(root, "rev-parse", f"refs/tags/{closest}^{{commit}}")
    # Peel recursively: --points-at alone omits annotated tags targeting
    # another annotated tag instead of directly targeting the commit.
    tag_commits = _git(root, "rev-parse", *(f"refs/tags/{tag}^{{commit}}" for tag in tags)).splitlines()
    # At the same commit, rc1/final/post-release tags must supersede rc0 even
    # when Git would prefer an older annotated tag over a lightweight tag.
    return max((tag for tag, target in zip(tags, tag_commits, strict=True) if target == commit), key=tags.__getitem__)


def _untagged_release_version(version: ScmVersion, release: str) -> str:
    # Retain distance/hash/dirty fields and delegate formatting to SCM.
    return guess_next_dev_version(replace(version, tag=Version(f"{release}rc0")))


def get_package_version(root: str, write_to: str | None = None) -> str:
    """Select a release-family tag while retaining native SCM traceability."""
    options: dict[str, Any] = {"root": root, "fallback_root": root, "write_to": write_to}
    try:
        branch = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        )
    except FileNotFoundError:
        release = None
    else:
        release = _release_version(branch.stdout.strip()) if branch.returncode == 0 else None
    if release is None:
        return get_version(**options)

    tags = _version_tags(root)
    family_tags = {tag: version for tag, version in tags.items() if version.base_version == release}
    # Preserve the installed SCM version's native hash abbreviation and dirty
    # handling while replacing its tag filter with the selected exact tag.
    describe = [*DEFAULT_DESCRIBE, "--no-match"]
    if family_tags:
        describe.extend(("--match", _closest_tag(root, family_tags)))
    else:
        # With no family tag, use the usual valid-tag distance (or the full
        # commit count with no valid tags), formatted against a virtual rc0.
        if tags:
            for tag in tags:
                describe.extend(("--match", tag))
        else:
            describe.extend(("--exclude", "*"))
        options["version_scheme"] = partial(_untagged_release_version, release=release)

    # SCM 8 uses the legacy keyword; newer versions expose the nested public
    # configuration. Detect the supported API instead of pinning SCM versions.
    if "scm" in signature(get_version).parameters:
        options["scm"] = {"git": {"describe_command": describe}}
    else:
        options["git_describe_command"] = describe
    return get_version(**options)
