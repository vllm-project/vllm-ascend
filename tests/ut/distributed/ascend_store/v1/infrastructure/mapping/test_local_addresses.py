"""Independent local-position evaluation preserves ordering and numeric boundaries."""

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.local_layout import (
    LocalMemorySegments,
)


@pytest.mark.parametrize(
    "strides, expected",
    [([16], [[132, 232], [100, 200], [132, 232]]), ([16, 32], [[132, 264], [100, 200], [132, 264]])],
)
def test_local_addresses_preserve_repeated_blocks_and_segment_order(strides, expected):
    segments = LocalMemorySegments(
        np.array([100, 200], dtype=np.uint64),
        np.array(strides, dtype=np.uint64),
        np.array([8], dtype=np.uint64),
    )
    ids = np.array([2, 0, 2], dtype=np.uint64)
    actual = segments.addresses(ids)
    assert actual.tolist() == expected
    assert actual.dtype == np.uint64
    assert segments.addresses([]).shape == (0, 2)
    actual[:] = 0
    assert segments.block_zero_segment_addresses.tolist() == [100, 200]
    assert ids.tolist() == [2, 0, 2]
    with pytest.raises(ValueError, match="one-dimensional"):
        segments.addresses([[2, 0]])


def test_local_addresses_keep_existing_uint64_wrap_semantics():
    segments = LocalMemorySegments(
        np.array([(1 << 64) - 8, 100], dtype=np.uint64),
        np.array([16], dtype=np.uint64),
        np.array([8], dtype=np.uint64),
    )
    assert segments.addresses([2, 0, 2]).tolist() == [[24, 132], [(1 << 64) - 8, 100], [24, 132]]
