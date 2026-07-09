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
"""Detect vllm / vllm-ascend version skew at a bisected commit.

Each vllm-ascend commit pins the vLLM release it is built against in the
version-controlled file ``.github/vllm-release-tag.commit`` (e.g. ``v0.22.1``).
The bisect container has exactly one vLLM installed (paired with the bad/HEAD
commit). When a candidate commit pins a *different* vLLM tag than the one
installed, that commit cannot be validly tested in this container -- running it
would fail with a confusing collection/import error (pytest rc=4). We detect
this up front and report a clear SKIP instead.

Mirrors ``vllm_ascend.utils.vllm_version_is``: prefer the ``VLLM_VERSION`` env
var, else ``vllm.__version__``.
"""

import logging
import os
from pathlib import Path

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

VLLM_TAG_FILE = ".github/vllm-release-tag.commit"


def installed_vllm_version() -> str | None:
    """The vLLM version active in the container (env override, else import)."""
    env = os.getenv("VLLM_VERSION")
    if env:
        return env.strip()
    try:
        import vllm  # type: ignore[import-not-found]

        return vllm.__version__
    except Exception as exc:  # noqa: BLE001 - vllm may be absent/broken
        logger.warning("Could not import vllm to read its version: %s", exc)
        return None


def expected_vllm_tag(repo: Path) -> str | None:
    """The vLLM tag the *currently checked-out* commit is pinned to, or None."""
    path = repo / VLLM_TAG_FILE
    if not path.exists():
        return None
    tag = path.read_text(encoding="utf-8").strip()
    return tag or None


def expected_vllm_tag_at(repo: Path, commit: str) -> str | None:
    """The vLLM tag ``commit`` is pinned to, read without checking it out."""
    from tools.bisect import git_ops

    content = git_ops.file_at_commit(repo, commit, VLLM_TAG_FILE)
    return content.strip() if content else None


def _compare(expected: str | None, installed: str | None) -> tuple[bool, str]:
    if not expected:
        return True, f"no {VLLM_TAG_FILE} at this commit; skipping vllm compat check"
    if not installed:
        return True, "installed vllm version unknown; skipping vllm compat check"
    try:
        installed_base = Version(installed).base_version
        expected_base = Version(expected).base_version
    except InvalidVersion:
        return True, (
            f"cannot parse vllm versions (installed={installed!r}, "
            f"expected={expected!r}); set VLLM_VERSION to compare by hand. "
            "Skipping vllm compat check."
        )
    if installed_base == expected_base:
        return True, f"vllm matches (installed {installed} ~ pinned {expected})"
    return False, (
        f"vllm version mismatch: this commit pins {expected} but the container "
        f"has {installed}; cannot be validly tested here"
    )


def check_compatible_at(repo: Path, commit: str) -> tuple[bool, str]:
    """Like ``check_compatible`` but for a commit, without checking it out."""
    return _compare(expected_vllm_tag_at(repo, commit), installed_vllm_version())


def check_compatible(repo: Path) -> tuple[bool, str]:
    """Return (compatible, reason) for the checked-out commit vs installed vLLM.

    Compatible (True) is returned -- leniently -- when either version is unknown
    or unparsable (e.g. a local dev build): we'd rather let pytest run than skip
    on a guess. Only a confidently-detected tag mismatch returns False.

    The release segment (base_version) is compared so an optional leading 'v'
    (PEP 440) is ignored and a dev/local build of the same release still matches.
    """
    return _compare(expected_vllm_tag(repo), installed_vllm_version())
