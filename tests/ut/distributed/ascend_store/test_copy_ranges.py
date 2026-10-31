# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from unittest.mock import Mock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.copy_ranges import (
    copy_gva_ranges,
    copy_key_ranges,
    range_transfer_batches,
)


def test_legacy_packet_order_and_input_address_limit():
    copy = Mock(return_value=0)
    assert (
        copy_gva_ranges(copy, [100, 200, 300], [1000, 2000, 3000], [9, 5, 2], 0, max_addresses=2, max_packet_bytes=4)
        == 0
    )
    assert copy.call_args_list == [
        call([100, 200, 104, 204, 108], [1000, 2000, 1004, 2004, 1008], [4, 4, 4, 1, 1], 0),
        call([300], [3000], [2], 0),
    ]


@pytest.mark.parametrize("direction", [0, 1])
def test_bounded_multi_object_copy_preserves_bytes_and_padding(direction):
    source = ctypes.create_string_buffer(bytes(range(64)))
    target = ctypes.create_string_buffer(b"x" * 64)
    src, dst = ctypes.addressof(source), ctypes.addressof(target)
    calls = []

    def copy(gvas, buffers, sizes, operation):
        assert operation == direction
        assert sum(sizes) <= 9 and len(sizes) <= 3
        calls.append(sizes)
        for gva, buffer, size in zip(gvas, buffers, sizes, strict=True):
            ctypes.memmove(gva if operation == 0 else buffer, buffer if operation == 0 else gva, size)
        return 0

    offsets, lengths = [1, 20, 40], [13, 8, 17]
    source_ranges, target_ranges = ([src + offset for offset in offsets], [dst + offset for offset in offsets])
    assert (
        copy_gva_ranges(
            copy,
            target_ranges if direction == 0 else source_ranges,
            source_ranges if direction == 0 else target_ranges,
            lengths,
            direction,
            max_addresses=3,
            max_packet_bytes=5,
            max_batch_bytes=9,
        )
        == 0
    )
    expected = bytearray(b"x" * 64)
    for offset, length in zip(offsets, lengths, strict=True):
        expected[offset : offset + length] = source.raw[offset : offset + length]
    assert target.raw[:64] == expected
    assert len(calls) > 1


def test_first_native_failure_stops_later_submissions():
    copy = Mock(side_effect=[0, -3, 0])
    assert copy_gva_ranges(copy, [100], [1000], [12], 1, max_batch_bytes=4) == -3
    assert copy.call_count == 2


def test_small_objects_share_one_native_submission():
    copy = Mock(return_value=0)
    assert copy_gva_ranges(copy, [100, 200], [1000, 2000], [4, 4], 0, max_batch_bytes=16) == 0
    copy.assert_called_once_with([100, 200], [1000, 2000], [4, 4], 0)


def test_legacy_key_range_limits_preserve_rows_and_fast_path():
    keys, buffers, sizes, offsets = ["a", "b"], [[100], [200]], [[9], [2]], [[3], [7]]
    batch = range_transfer_batches(keys, buffers, sizes, offsets, 0, 0)[0]
    assert all(original is returned for original, returned in zip((keys, buffers, sizes, offsets), batch, strict=True))
    assert range_transfer_batches(keys, buffers, sizes, offsets, 1, 4) == [
        (["a"], [[100, 104, 108]], [[4, 4, 1]], [[3, 7, 11]]),
        (["b"], [[200]], [[2]], [[7]]),
    ]


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("return_byte_counts", [False, True])
def test_key_range_copy_preserves_object_offsets_and_byte_limits(write, return_byte_counts):
    objects = {key: ctypes.create_string_buffer(bytes(range(64)) if not write else b"x" * 64) for key in ("a", "b")}
    locals_ = {key: ctypes.create_string_buffer(bytes(range(64)) if write else b"x" * 64) for key in objects}
    regions = {"a": [(4, 2, 13), (25, 21, 7)], "b": [(7, 11, 11)]}
    calls = []

    def copy(keys, buffers, sizes, offsets):
        assert sum(map(sum, sizes)) <= 9
        assert sum(map(len, sizes)) <= 2
        calls.append(keys)
        for key, addrs, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
            for addr, size, start in zip(addrs, lengths, starts, strict=True):
                obj = ctypes.addressof(objects[key]) + start
                ctypes.memmove(obj if write else addr, addr if write else obj, size)
        return [sum(row) for row in sizes] if return_byte_counts else [0] * len(keys)

    copy_key_ranges(
        copy,
        list(objects),
        [[ctypes.addressof(locals_[key]) + local for local, _, _ in regions[key]] for key in objects],
        [[size for _, _, size in regions[key]] for key in objects],
        [[offset for _, offset, _ in regions[key]] for key in objects],
        max_ranges=2,
        max_bytes=9,
        operation="test range copy",
    )
    for key in objects:
        expected = bytearray(b"x" * 64)
        for local, offset, size in regions[key]:
            dst, src = (offset, local) if write else (local, offset)
            expected[dst : dst + size] = bytes(range(64))[src : src + size]
        assert (objects if write else locals_)[key].raw[:64] == expected
    assert len(calls) > 1


def test_key_range_failure_stops_before_later_batch():
    copy = Mock(side_effect=[[4], [-3], [4]])
    with pytest.raises(RuntimeError, match="range copy failed"):
        copy_key_ranges(copy, ["a"], [[100]], [[12]], [[0]], max_ranges=2, max_bytes=4, operation="range copy")
    assert copy.call_count == 2
