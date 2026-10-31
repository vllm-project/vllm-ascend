# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import copy
import random
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
    _compact_transfer_args,
)


def coverage(addrs, sizes):
    """Independently compute the union of half-open byte ranges."""
    merged = []
    for start, end in sorted((addr, addr + size) for addr, size in zip(addrs, sizes, strict=True)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@pytest.mark.parametrize(
    "addrs,sizes,retained",
    [
        ([], [], []),
        ([100], [10], [0]),
        ([100, 100], [10, 10], [0]),
        ([100, 200], [10, 10], [0, 1]),
        ([100, 110], [10, 10], [0, 1]),
        ([100, 105], [10, 10], [0, 1]),
        ([100, 100], [10, 20], [1]),
        ([105, 100], [5, 20], [1]),
        ([100, 110, 100], [10, 5, 15], [2]),
        ([100, 120, 105], [10, 10, 20], [0, 1, 2]),
        ([200, 105, 100], [10, 5, 20], [0, 2]),
        ([100, 105, 110], [10, 10, 2], [0, 1]),
    ],
)
def test_containment_preserves_coverage_and_packed_order(addrs, sizes, retained):
    original = copy.deepcopy((addrs, sizes))
    actual = _compact_transfer_args(["any-model@any-group"], [addrs], [sizes])
    assert actual == ([[addrs[i] for i in retained]], [[sizes[i] for i in retained]])
    assert coverage(*original) == coverage(actual[0][0], actual[1][0])
    assert (addrs, sizes) == original
    assert _compact_transfer_args(["any-model@any-group"], *actual) == actual


@pytest.mark.parametrize("block_size", [32, 128, 256])
@pytest.mark.parametrize("layers", [1, 29, 32, 61])
def test_indexer_alias_regression(block_size, layers):
    # Indexer data and scale also appear as one enclosing view. Sizes are
    # fixtures for the reported regression, not conditions in the backend.
    sizes = [size * block_size // 32 for size in (19456, 4096, 128, 4224)]
    addrs = [
        value
        for base in range(0, layers * 400000, 400000)
        for value in (base, base + 200000, base + 200000 + sizes[1], base + 200000)
    ]
    result = _compact_transfer_args(["opaque-key"], [addrs], [sizes * layers])
    assert result == (
        [[addr for i, addr in enumerate(addrs) if i % 4 in (0, 3)]],
        [[size for i, size in enumerate(sizes * layers) if i % 4 in (0, 3)]],
    )
    assert coverage(addrs, sizes * layers) == coverage(result[0][0], result[1][0])


def test_batches_do_not_compare_addresses_between_keys():
    addrs, sizes = [[100, 100], [100], []], [[4, 8], [4], []]
    original = copy.deepcopy((addrs, sizes))
    assert _compact_transfer_args(["a", "b", "c"], addrs, sizes) == ([[100], [100], []], [[8], [4], []])
    assert (addrs, sizes) == original
    assert _compact_transfer_args([], [], []) == ([], [])


@pytest.mark.parametrize(
    "keys,addrs,sizes",
    [
        (["a"], [], []),
        ([], [[0]], []),
        (["a"], [[0]], []),
        (["a"], [[0]], [[1, 2]]),
        (["a"], [[0]], [[0]]),
        (["a"], [[0]], [[-1]]),
        (["a"], [[-1]], [[1]]),
    ],
)
def test_invalid_arguments_are_rejected(keys, addrs, sizes):
    with pytest.raises(ValueError):
        _compact_transfer_args(keys, addrs, sizes)


def test_randomized_ranges_preserve_coverage():
    rng = random.Random(20261022)
    for _ in range(1000):
        addrs = [rng.randrange(100) for _ in range(rng.randrange(30))]
        sizes = [rng.randrange(1, 40) for _ in addrs]
        original = copy.deepcopy((addrs, sizes))
        compact_addrs, compact_sizes = _compact_transfer_args(["key"], [addrs], [sizes])
        assert coverage(addrs, sizes) == coverage(compact_addrs[0], compact_sizes[0])
        assert (addrs, sizes) == original
        assert _compact_transfer_args(["key"], compact_addrs, compact_sizes) == (compact_addrs, compact_sizes)


def make_backend():
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend._lazy_init = False
    backend._store_initialized = True
    backend.store = MagicMock()
    return backend


def test_backend_put_get_roundtrip_with_reversed_allocations():
    source = bytearray((index * 17 + 13) % 256 for index in range(2000))
    destination = bytearray([255]) * len(source)
    # Destination allocations have the opposite address order. Packed order
    # must follow descriptors, not the addresses used to find containment.
    source_addrs = [100, 500, 508, 500, 1000, 1500, 1508, 1500]
    target_addrs = [1000, 1500, 1508, 1500, 100, 500, 508, 500]
    sizes = [13, 8, 3, 11] * 2
    objects = {}

    def put(keys, addresses, counts, direction):
        assert direction == MmcDirect.COPY_L2G.value
        for key, pointers, lengths in zip(keys, addresses, counts, strict=True):
            objects[key] = b"".join(
                source[pointer : pointer + length] for pointer, length in zip(pointers, lengths, strict=True)
            )
        return [0] * len(keys)

    def get(keys, addresses, counts, direction):
        assert direction == MmcDirect.COPY_G2L.value
        for key, pointers, lengths in zip(keys, addresses, counts, strict=True):
            offset = 0
            for pointer, length in zip(pointers, lengths, strict=True):
                destination[pointer : pointer + length] = objects[key][offset : offset + length]
                offset += length
            assert offset == len(objects[key])
        return [0] * len(keys)

    backend = make_backend()
    backend.store.batch_put_from_layers.side_effect = put
    backend.store.batch_get_into_layers.side_effect = get
    assert backend.put(["opaque-key"], [source_addrs], [sizes])
    assert backend.get(["opaque-key"], [target_addrs], [sizes]) == [0]
    assert len(objects["opaque-key"]) == 48
    expected = bytearray([255]) * len(source)
    for source_addr, target_addr, size in zip(source_addrs, target_addrs, sizes, strict=True):
        expected[target_addr : target_addr + size] = source[source_addr : source_addr + size]
    assert destination == expected, "lost data or copied padding/unrelated bytes"


def test_invalid_arguments_do_not_reach_store():
    backend = make_backend()
    assert backend.get(["key"], [[100]], [[0]]) is None
    assert backend.put(["key"], [[100]], [[0]]) is False
    backend.store.batch_get_into_layers.assert_not_called()
    backend.store.batch_put_from_layers.assert_not_called()
