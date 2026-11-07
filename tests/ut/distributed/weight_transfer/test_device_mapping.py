# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical-to-physical device mapping at RLHF child-server boundaries."""

import pytest

from tests.e2e.rlhf.common import server_visible_devices


@pytest.mark.parametrize(
    ("visible", "index", "size", "expected"),
    [
        (None, 0, 1, "0"),
        (None, 6, 2, "6,7"),
        ("6,7", 0, 1, "6"),
        ("6,7", 1, 1, "7"),
        ("6,7", 0, 2, "6,7"),
        (" 7, 3, 6 ", 1, 2, "3,6"),
    ],
)
def test_server_visible_devices_respects_parent_allocation(monkeypatch, visible, index, size, expected):
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    assert server_visible_devices(index, size) == expected


@pytest.mark.parametrize(
    ("visible", "index", "size"),
    [
        ("6,7", -1, 1),
        ("6,7", 0, 0),
        ("6,7", 2, 1),
        ("6,7", 1, 2),
        ("6,6", 0, 1),
        ("6,x", 0, 1),
        ("", 0, 1),
        ("6,", 0, 1),
    ],
)
def test_server_visible_devices_rejects_invalid_allocation(monkeypatch, visible, index, size):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        server_visible_devices(index, size)
