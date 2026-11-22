# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the addresses formed by each lookback row, including masked lanes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from vllm_ascend.ops.triton import engram_lookback


@pytest.fixture
def host_kernel(monkeypatch):
    # Execute the JIT function body with integer offsets in place of device pointers.
    language = SimpleNamespace(
        program_id=Mock(),
        arange=lambda start, end: np.arange(start, end, dtype=np.int32),
        maximum=np.maximum,
        full=np.full,
        load=Mock(),
        store=Mock(),
    )
    monkeypatch.setattr(engram_lookback, "tl", language)
    kernel = engram_lookback._gather_lookback_kernel
    return getattr(kernel, "fn", kernel), language


@pytest.mark.parametrize("batch_idx", [1, 3])
def test_padding_row_only_writes_its_window(host_kernel, batch_idx):
    kernel, language = host_kernel
    language.program_id.return_value = batch_idx
    # One real request. Even computing an input address for a padding row is an error.
    unused = MagicMock()
    unused.dtype.element_ty = np.int32
    unused.__add__.side_effect = AssertionError("padding row formed an input address")
    kernel(0, unused, unused, unused, 8, 1, DEPTH=3, BLOCK_DEPTH=4)

    language.load.assert_not_called()
    language.store.assert_called_once()
    (addresses, values), kwargs = language.store.call_args
    mask = kwargs["mask"]
    np.testing.assert_array_equal(addresses[mask], batch_idx * 3 + np.arange(3))
    np.testing.assert_array_equal(values[mask], np.full(3, -1))


@pytest.mark.parametrize(
    ("slot", "computed", "stride"),
    [(0, 0, 8), (2, 0, 12), (2, 1, 12), (2, 2, 12), (2, 3, 12), (2, 4, 12), (2, 8, 12), (4096, 8, 1 << 20)],
)
def test_history_addresses_stay_in_the_mapped_row(host_kernel, slot, computed, stride):
    kernel, language = host_kernel
    language.program_id.return_value = 0
    language.load.side_effect = [np.int64(slot), computed, np.zeros(4, dtype=np.int32)]
    kernel(0, 0, 0, 0, stride, 1, DEPTH=3, BLOCK_DEPTH=4)

    # Check all token addresses before applying the load mask, including for an empty history.
    (addresses,), kwargs = language.load.call_args
    columns = addresses - slot * stride
    assert np.all((columns >= 0) & (columns < 8))
    valid = np.arange(4) < min(3, computed)
    np.testing.assert_array_equal(kwargs["mask"], valid)
    np.testing.assert_array_equal(columns[valid], np.arange(computed)[::-1][:3])
    assert kwargs["other"] == -1
