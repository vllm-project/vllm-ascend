"""Native integer result validation preserves values and rejects malformed batches."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import (
    BackendResultError,
    integer_result,
    integer_results,
)


def test_integer_validation_preserves_addresses_byte_counts_and_error_codes():
    assert integer_results("allocation", ["address", "rejected", "failed"], [1000, 0, -7]) == (1000, 0, -7)
    assert integer_result("copied bytes", 64) == 64


@pytest.mark.parametrize("values", (None, [], [1, 2], [True], [1.5]))
def test_integer_validation_rejects_malformed_per_object_results(values):
    with pytest.raises(BackendResultError):
        integer_results("operation", ["key"], values)


def test_builtin_integer_batch_reuses_tuple_after_validation():
    values = (0, -7, 1 << 63)
    assert integer_results("native", ["a", "b", "c"], values) is values


def test_mixed_integer_batch_normalizes_numpy_values_without_changing_values():
    values = (0, np.int64(-7), np.uint64((1 << 64) - 1), 8)
    result = integer_results("native", ["a", "b", "c", "d"], iter(values))
    assert result == (0, -7, (1 << 64) - 1, 8)
    assert all(type(value) is int for value in result)


@pytest.mark.parametrize("invalid", (True, np.bool_(False), 1.5, "1", None))
def test_integer_batch_rejects_invalid_values_after_valid_and_converted_prefix(invalid):
    with pytest.raises(BackendResultError, match="non-integer"):
        integer_results("native", ["a", "b", "c"], [0, np.int64(7), invalid])
