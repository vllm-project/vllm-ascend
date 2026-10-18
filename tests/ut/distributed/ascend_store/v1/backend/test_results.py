"""Native integer result validation preserves values and rejects malformed batches."""

from __future__ import annotations

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
