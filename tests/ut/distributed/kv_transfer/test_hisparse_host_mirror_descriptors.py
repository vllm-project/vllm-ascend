# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check mapped writeback descriptors against an independent byte-copy oracle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/hisparse/host_mirror_descriptors.py"
_spec = importlib.util.spec_from_file_location("hisparse_test_mirror_descriptors", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
prepare_snapshot = _module.prepare_snapshot

DEVICE = 0x120000000000
HOST = 0x100000000000
MAPPED = 0x140000000000


@pytest.mark.parametrize("count", [1, 8, 256, 257, 632, 2528])
def test_mapped_snapshot_preserves_exact_bytes_and_unwritten_neighbors(count):
    rng = np.random.default_rng(42)
    payload = rng.integers(0, 256, (count + 1) * 2048, dtype=np.uint8)
    original = rng.integers(0, 256, payload.size, dtype=np.uint8)
    src = DEVICE + np.arange(count, dtype=np.int64) * 2048 + 19
    dst = HOST + np.arange(count, dtype=np.int64) * 2048 + 31
    sizes = rng.choice(np.array([1, 17, 127, 656, 1312], dtype=np.int64), size=count)
    snapshot, eligible, reason = prepare_snapshot(
        src, dst, sizes, [(DEVICE, payload.size)], [(HOST, original.size, MAPPED)]
    )
    assert eligible and reason == "small_disjoint_extents"
    expected, actual = original.copy(), original.copy()
    for source, destination, size in zip(src, dst, sizes):
        expected[destination - HOST : destination - HOST + size] = payload[source - DEVICE : source - DEVICE + size]
    for source, destination, size in snapshot.T:
        actual[destination - MAPPED : destination - MAPPED + size] = payload[source - DEVICE : source - DEVICE + size]
    np.testing.assert_array_equal(actual, expected)
    saved = snapshot.copy()
    src[:] = dst[:] = sizes[:] = 0
    np.testing.assert_array_equal(snapshot, saved)


@pytest.mark.parametrize("overlap", [False, True])
def test_ineligible_writes_keep_native_dma(overlap):
    src = np.array([DEVICE, DEVICE + 1024], dtype=np.int64)
    dst = np.array([HOST, HOST + (17 if overlap else 131072)], dtype=np.int64)
    size = np.array([656, 656] if overlap else [83968, 83968], dtype=np.int64)
    _, eligible, reason = prepare_snapshot(src, dst, size, [(DEVICE, 1 << 20)], [(HOST, 1 << 20, MAPPED)])
    assert not eligible
    assert reason == ("ordered_overlapping_destinations" if overlap else "large_extent_keeps_native_DMA")


@pytest.mark.parametrize("kind", ["outside_source", "outside_destination", "ambiguous_source", "overflow"])
def test_bad_ownership_is_rejected_before_enqueue(kind):
    source, destination, length = DEVICE, HOST, 656
    allocations = [(DEVICE, 1 << 20)]
    if kind == "outside_source":
        source -= 1
    elif kind == "outside_destination":
        destination -= 1
    elif kind == "ambiguous_source":
        allocations *= 2
    else:
        destination = (1 << 63) - 2
    with pytest.raises(ValueError):
        prepare_snapshot(
            np.array([source], dtype=np.int64),
            np.array([destination], dtype=np.int64),
            np.array([length], dtype=np.int64),
            allocations,
            [(HOST, 1 << 20, MAPPED)],
        )


@pytest.mark.parametrize("count", [1, 8, 9, 16, 32, 64, 128, 311, 312, 624, 702])
def test_small_batch_fallback_preserves_exact_descriptor_snapshot(count):
    offsets = np.arange(count, dtype=np.int64) * 4096
    source, destination = DEVICE + offsets, HOST + offsets
    lengths = np.full(count, 2624, dtype=np.int64)
    matrix, eligible, reason = prepare_snapshot(
        source,
        destination,
        lengths,
        [(DEVICE, count * 4096)],
        [(HOST, count * 4096, MAPPED)],
        min_aiv_descriptors=312,
    )
    assert eligible == (count >= 312)
    assert reason == ("small_disjoint_extents" if eligible else "small_batch_keeps_native_DMA")
    np.testing.assert_array_equal(matrix, np.stack((source, MAPPED + offsets, lengths)))
    # A fallback must not rewrite the caller's CPU destinations to NPU aliases.
    np.testing.assert_array_equal(destination, HOST + offsets)


def test_small_batch_still_rejects_expired_or_invalid_ownership():
    with pytest.raises(ValueError, match="live Host registration"):
        prepare_snapshot(
            np.array([DEVICE], dtype=np.int64),
            np.array([HOST - 1], dtype=np.int64),
            np.array([656], dtype=np.int64),
            [(DEVICE, 4096)],
            [(HOST, 4096, MAPPED)],
            min_aiv_descriptors=312,
        )
