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
"""Backward-compatible vLLM version helpers for the bisect tool."""

import logging
from pathlib import Path

from packaging.version import InvalidVersion, Version

from tools.bisect.version_compat import (
    VLLM_TAG_FILE,
    expected_vllm_version,
    expected_vllm_version_at,
    installed_vllm_version,
)

logger = logging.getLogger(__name__)


def expected_vllm_tag(repo: Path) -> str | None:
    return expected_vllm_version(repo)


def expected_vllm_tag_at(repo: Path, commit: str) -> str | None:
    return expected_vllm_version_at(repo, commit)


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
    return _compare(expected_vllm_tag_at(repo, commit), installed_vllm_version())


def check_compatible(repo: Path) -> tuple[bool, str]:
    return _compare(expected_vllm_tag(repo), installed_vllm_version())
