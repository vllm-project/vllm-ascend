#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Pytest configuration for pull-request E2E tests.

Provides:
- ``@pytest.mark.requires_hardware("A2", "A3")`` — only run on matching
  NPU chip types. Multiple values = any of them.  Also accepts
  detailed SOC version strings such as ``"ascend910_9362"`` — a test
  marked with a specific SOC version will only run on hardware whose
  SOC version matches exactly.
- ``@pytest.mark.requires_npus(1, 2, 4)`` — only run when the visible
  NPU count is one of the specified values.

Tests that don't carry these markers run unconditionally (backward
compatible). Tests whose markers don't match the current environment
are skipped with a clear message.

CLI overrides:
- ``--requires-hardware A2,A3`` — override the hardware check, useful
  with ``--collect-only`` to preview what would run on a given chip
  without being on that hardware.
- ``--requires-npus 1,2`` — override the NPU count check.
"""

from __future__ import annotations

import pytest

from tests.e2e.pull_request.hardware_utils import (
    detect_device_type,
    detect_npu_count,
    detect_soc_version,
)

# CLI override state — populated by pytest_configure when
# --requires-hardware / --requires-npus are provided on the command line.
# When set, these replace the live hardware detection in filtering.
_CLI_HARDWARE: tuple[str, ...] | None = None
_CLI_NPUS: tuple[int, ...] | None = None


def pytest_addoption(parser):
    """Register CLI options for hardware-aware test selection."""
    parser.addoption(
        "--requires-hardware",
        action="store",
        default=None,
        metavar="TYPES",
        help="Comma-separated hardware types / SOC versions (e.g. A2,A3,ascend910_9362). "
        "When set, overrides live hardware detection for filtering.",
    )
    parser.addoption(
        "--requires-npus",
        action="store",
        default=None,
        metavar="COUNTS",
        help="Comma-separated NPU counts (e.g. 1,2,4). When set, overrides live NPU count detection for filtering.",
    )


def pytest_configure(config):
    """Register custom markers and parse CLI overrides."""
    global _CLI_HARDWARE, _CLI_NPUS

    # Parse CLI overrides
    hw = config.getoption("requires_hardware", default=None)
    if hw:
        _CLI_HARDWARE = tuple(v.strip() for v in hw.split(","))

    npus = config.getoption("requires_npus", default=None)
    if npus:
        _CLI_NPUS = tuple(int(v.strip()) for v in npus.split(","))

    # Register markers
    config.addinivalue_line(
        "markers",
        "requires_hardware(chip1, chip2, ...): "
        "Only run on specified NPU chip types (A2, A3, 310P, A5) or "
        "detailed SOC version strings (e.g. ascend910_9362). "
        "Multiple values mean any of them is sufficient — "
        "e.g. requires_hardware('A2', 'A3') runs on both A2 and A3, "
        "and requires_hardware('ascend910_9362') runs only on "
        "that specific SOC revision.",
    )
    config.addinivalue_line(
        "markers",
        "requires_npus(n1, n2, ...): "
        "Only run when the visible NPU count is at least the minimum "
        "specified value (e.g. requires_npus(1) runs on 1, 2, or 4 cards; "
        "requires_npus(4) runs on 4 or 8 cards). "
        "Multiple values mean any of them as the minimum.",
    )


# ── Collection-time filtering (drives --collect-only natively) ──────────────


def pytest_collection_modifyitems(config, items):
    """Remove tests whose hardware / NPU-count markers don't match the
    current environment (or CLI override when ``--requires-*`` is set).

    Because this runs at collection time, ``pytest --collect-only``
    natively shows only the tests that would actually execute — without
    needing a separate dry-run mode.
    """
    items[:] = [item for item in items if _matches_hardware(item) and _matches_npu_count(item)]


# ── Predicates (shared by collection-time filter and runtime safety net) ─────


def _matches_hardware(item) -> bool:
    """Return True if *item*'s ``requires_hardware`` marker (if any) is
    compatible with the current device (or CLI override).  Marker-less
    items always pass.

    Marker values are checked against:
    - The coarse device type (``A2``, ``A3``, ``310P``, ``A5``).
    - The detailed SOC version string (e.g. ``ascend910_9362``).
    If any value matches either, the test is considered compatible.
    """
    marker = item.get_closest_marker("requires_hardware")
    if marker is None:
        return True

    if _CLI_HARDWARE is not None:
        # CLI override mode: match against provided values
        return any(arg in _CLI_HARDWARE for arg in marker.args)

    device = detect_device_type()
    soc_version = detect_soc_version()
    return (
        device in marker.args
        or any(hasattr(arg, "name") and arg.name == device for arg in marker.args)
        or soc_version in marker.args
        or any(hasattr(arg, "name") and arg.name == soc_version for arg in marker.args)
    )


def _matches_npu_count(item) -> bool:
    """Return True if *item*'s ``requires_npus`` marker (if any) is
    compatible with the current NPU count (or CLI override).
    Marker-less items always pass.

    In live mode, uses "at least min(args)" semantics (e.g.
    ``requires_npus(4)`` runs on 4 or 8 cards).  In CLI override mode,
    uses exact set membership.
    """
    marker = item.get_closest_marker("requires_npus")
    if marker is None:
        return True

    if _CLI_NPUS is not None:
        # CLI override: exact match against provided values
        return any(arg in _CLI_NPUS for arg in marker.args)

    return detect_npu_count() >= min(marker.args)


# ── Runtime safety net (should not trigger after collection filtering) ───────


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip tests that somehow slip through collection-time filtering."""
    _skip_if_hardware_mismatch(item)
    _skip_if_npu_count_mismatch(item)


def _skip_if_hardware_mismatch(item):
    if _matches_hardware(item):
        return
    marker = item.get_closest_marker("requires_hardware")

    if _CLI_HARDWARE is not None:
        pytest.skip(
            f"@pytest.mark.requires_hardware{marker.args} does not match "
            f"CLI --requires-hardware={','.join(_CLI_HARDWARE)}. "
            f"Test is not applicable."
        )
    else:
        device = detect_device_type()
        soc_version = detect_soc_version()
        soc_info = f"{device}" if soc_version == "unknown" else f"{device} / {soc_version}"
        pytest.skip(
            f"@pytest.mark.requires_hardware{marker.args} does not match "
            f"current device {soc_info!r}. "
            f"Test is not applicable on this hardware."
        )


def _skip_if_npu_count_mismatch(item):
    if _matches_npu_count(item):
        return
    marker = item.get_closest_marker("requires_npus")

    if _CLI_NPUS is not None:
        pytest.skip(
            f"@pytest.mark.requires_npus{marker.args} does not match "
            f"CLI --requires-npus={','.join(str(n) for n in _CLI_NPUS)}. "
            f"Test is not applicable."
        )
    else:
        count = detect_npu_count()
        required_min = min(marker.args)
        pytest.skip(
            f"@pytest.mark.requires_npus{marker.args} requires at least "
            f"{required_min} NPU(s), but current count is {count}. "
            f"Test is not applicable on this configuration."
        )
