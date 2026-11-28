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


@pytest.fixture
def history_ptr():
    ptr = MagicMock()
    ptr.dtype.element_ty = np.int32
    ptr.__add__.side_effect = lambda offset: offset
    return ptr


def assert_empty_window_row(store, batch_idx):
    store.assert_called_once()
    (addresses, values), kwargs = store.call_args
    mask = kwargs["mask"]
    np.testing.assert_array_equal(addresses[mask], batch_idx * 3 + np.arange(3))
    np.testing.assert_array_equal(values[mask], np.full(3, -1))


@pytest.mark.parametrize("batch_idx", [1, 3])
def test_padding_row_only_writes_its_window(host_kernel, history_ptr, batch_idx):
    kernel, language = host_kernel
    language.program_id.return_value = batch_idx
    # One real request. Even computing an input address for a padding row is an error.
    history_ptr.__add__.side_effect = AssertionError("padding row formed an input address")
    kernel(0, history_ptr, history_ptr, history_ptr, 8, 1, DEPTH=3, BLOCK_DEPTH=4)
    language.load.assert_not_called()
    assert_empty_window_row(language.store, batch_idx)


@pytest.mark.parametrize(("batch_idx", "slot"), [(2, 4), (3, 3), (4, 0), (5, 5), (6, 2), (7, 1)])
def test_empty_history_in_reported_full_batch(host_kernel, history_ptr, batch_idx, slot):
    # These six real requests in #21392 had no history, including the first slot in storage.
    kernel, language = host_kernel
    language.program_id.return_value = batch_idx
    language.load.side_effect = [np.int64(slot), np.int32(0)]
    history_ptr.__add__.side_effect = AssertionError("empty slot formed a token-history address")
    kernel(0, 0, 8, history_ptr, 128 * 1024, 8, DEPTH=3, BLOCK_DEPTH=4)
    assert [call.args for call in language.load.call_args_list] == [(batch_idx,), (8 + slot,)]
    assert_empty_window_row(language.store, batch_idx)


@pytest.mark.parametrize(
    ("slot", "computed", "stride"),
    [(2, 1, 12), (2, 2, 12), (2, 3, 12), (2, 4, 12), (2, 8, 12), (4096, 8, 1 << 20)],
)
def test_history_addresses_stay_in_the_mapped_row(host_kernel, history_ptr, slot, computed, stride):
    kernel, language = host_kernel
    language.program_id.return_value = 0
    language.load.side_effect = [np.int64(slot), computed, np.zeros(4, dtype=np.int32)]
    kernel(0, 0, 0, history_ptr, stride, 1, DEPTH=3, BLOCK_DEPTH=4)

    # Check all token addresses before applying the load mask, including for short histories.
    (addresses,), kwargs = language.load.call_args
    columns = addresses - slot * stride
    assert np.all((columns >= 0) & (columns < 8))
    valid = np.arange(4) < min(3, computed)
    np.testing.assert_array_equal(kwargs["mask"], valid)
    np.testing.assert_array_equal(columns[valid], np.arange(computed)[::-1][:3])
    assert kwargs["other"] == -1
