# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.models.qwen4_exp.ple_layout import PLELayout


def test_320_byte_rows_do_not_cross_gib_mapping():
    layout = PLELayout(rows=320001536, row_bytes=320)
    boundary = layout.rows_per_mapping
    assert boundary == 3355443
    assert layout.offset(boundary - 1) + 320 == (1 << 30) - 64
    assert layout.offset(boundary) == 1 << 30
    assert list(layout.spans(boundary - 1, boundary + 1)) == [
        (boundary - 1, boundary, (1 << 30) - 384),
        (boundary, boundary + 1, 1 << 30),
    ]
    assert layout.offset(layout.rows - 1) + 320 <= layout.allocation_bytes


def test_invalid_row_and_range_fail_closed():
    layout = PLELayout(rows=5, row_bytes=3, mapping_bytes=8)
    assert list(layout.spans(1, 5)) == [(1, 2, 3), (2, 4, 8), (4, 5, 16)]
    with pytest.raises(ValueError):
        layout.offset(5)
    with pytest.raises(ValueError):
        list(layout.spans(-1, 2))
    with pytest.raises(ValueError):
        PLELayout(rows=1, row_bytes=9, mapping_bytes=8)
