# SPDX-License-Identifier: Apache-2.0
"""Resource-aware compile job selection for source builds.

This module intentionally depends only on the Python standard library, apart
from an optional psutil lookup.  setup.py loads it by file path so importing it
does not initialize the vllm_ascend runtime package during a build.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

DEFAULT_BUILD_JOB_CAP = 8
BUILD_MEMORY_BYTES_PER_JOB = 4 * 1024**3
BUILD_MEMORY_RESERVE_BYTES = 2 * 1024**3
_UNLIMITED_CGROUP_THRESHOLD = 1 << 60
_CGROUP_MEMORY_FILES = (
    (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory.current"),
    ),
    (
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ),
)


class BuildJobPlan(NamedTuple):
    num_jobs: int
    source: str
    cpu_count: int
    available_memory_bytes: int | None


def detect_cpu_count() -> int:
    """Return the CPUs available to this process, including affinity limits."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count()
    return max(1, count or 1)


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _detect_cgroup_available_memory_bytes() -> int | None:
    for limit_path, usage_path in _CGROUP_MEMORY_FILES:
        limit = _read_integer(limit_path)
        if limit is None or limit >= _UNLIMITED_CGROUP_THRESHOLD:
            continue
        usage = _read_integer(usage_path)
        if usage is None:
            continue
        return max(0, limit - usage)
    return None


def _detect_host_available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * available_pages


def detect_available_memory_bytes() -> int | None:
    """Return the tightest visible host or cgroup memory availability."""
    candidates = [
        value
        for value in (
            _detect_host_available_memory_bytes(),
            _detect_cgroup_available_memory_bytes(),
        )
        if value is not None
    ]
    return min(candidates) if candidates else None


def default_build_jobs(
    cpu_count: int,
    available_memory_bytes: int | None,
) -> int:
    """Choose a conservative default when MAX_JOBS is not configured."""
    jobs = min(max(1, cpu_count), DEFAULT_BUILD_JOB_CAP)
    if available_memory_bytes is not None:
        usable_memory = max(
            0,
            available_memory_bytes - BUILD_MEMORY_RESERVE_BYTES,
        )
        memory_jobs = max(1, usable_memory // BUILD_MEMORY_BYTES_PER_JOB)
        jobs = min(jobs, memory_jobs)
    return max(1, jobs)


def resolve_build_jobs(max_jobs: str | None) -> BuildJobPlan:
    """Resolve an explicit MAX_JOBS value or an automatic safe default."""
    cpu_count = detect_cpu_count()
    if max_jobs is not None:
        try:
            num_jobs = int(max_jobs)
        except ValueError as error:
            raise ValueError(
                f"MAX_JOBS must be a positive integer, got {max_jobs!r}"
            ) from error
        if num_jobs <= 0:
            raise ValueError(
                f"MAX_JOBS must be a positive integer, got {max_jobs!r}"
            )
        return BuildJobPlan(
            num_jobs=num_jobs,
            source="MAX_JOBS",
            cpu_count=cpu_count,
            available_memory_bytes=None,
        )

    available_memory = detect_available_memory_bytes()
    return BuildJobPlan(
        num_jobs=default_build_jobs(cpu_count, available_memory),
        source="automatic",
        cpu_count=cpu_count,
        available_memory_bytes=available_memory,
    )
