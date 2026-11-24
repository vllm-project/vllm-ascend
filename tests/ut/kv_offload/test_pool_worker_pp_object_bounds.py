# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All physical layer copies must fit the GVA object allocated for their key."""

import ctypes

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


def _worker(layer_entries, offset=42, total_layers=78, dcp_size=1, dcp_rank=0, block_key=False, pp_size=2):
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.pp_size = pp_size
    worker.pp_rank = int(offset > 0)
    worker.layerwise_key_layer_offset = offset
    worker.total_layers = total_layers
    worker.num_layers = len(layer_entries)
    worker.layerwise_key_layers = worker.num_layers - 1
    worker.num_kv_cache_groups = 1
    worker.use_layerwise_transfer = not block_key
    worker.use_block_key_layerwise = block_key
    worker.dcp_size = dcp_size
    worker.dcp_rank = dcp_rank
    worker.put_step = 8 if dcp_size > 1 else 1
    flat = [size for entries in layer_entries for size in entries]
    offsets = [0]
    for entries in layer_entries:
        offsets.append(offsets[-1] + len(entries))
    worker.group_num_layers = {0: len(layer_entries)}
    worker.group_block_len = {0: flat}
    worker.group_block_stride = {0: [size * 2 for size in flat]}
    worker.group_kv_caches_base_addr = {0: [10_000_000 + i * 1_000_000 for i in range(len(flat))]}
    worker.group_layer_cache_entry_offsets = {0: offsets}
    worker.token_database = worker
    worker.page_size_bytes = sum(flat)
    return worker


@pytest.mark.parametrize(
    ("layer_entries", "num_layers"),
    [
        ([[64, 16]] * 11, 45),
        ([[64, 16]] * 12, 46),
        ([[64, 16]] * 11, 46),
        ([[13, 7], [29]], 46),
    ],
    ids=["target-group", "mtp-group", "target-group-with-mtp", "unequal-entries"],
)
def test_unsharded_object_contains_group_payload_without_global_padding(layer_entries, num_layers):
    worker = _worker(layer_entries, offset=0, total_layers=45, pp_size=1)
    worker.num_layers = num_layers
    worker.put_step = 16
    builder = worker._build_group_layer_builders()[0]
    allocated = worker._global_group_alloc_size(0)
    assert allocated == sum(size for entries in layer_entries for size in entries)
    assert builder.page_size_bytes == allocated
    assert builder.layer_byte_offset == 0

    base = 1_000_000_000
    copied_bytes = 0
    for layer in range(len(layer_entries)):
        _, sizes, gvas = builder._build_transfer_arrays(np.array([1]), np.array([base]), layer)
        for size, gva in zip(sizes, gvas, strict=True):
            assert gva == base + copied_bytes
            copied_bytes += int(size)
            assert copied_bytes <= allocated
    assert copied_bytes == allocated


@pytest.mark.parametrize("offset", [0, 23])
@pytest.mark.parametrize(("dcp_size", "dcp_rank"), [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("put_step", [1, 16])
def test_pp_group_payload_is_packed_into_aligned_dcp_shards(offset, dcp_size, dcp_rank, put_step):
    mib = 1024 * 1024
    worker = _worker(
        [[3 * mib // 4, mib // 4]] * 7,
        offset=offset,
        total_layers=45,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    worker.num_layers = 23
    worker.put_step = put_step
    builder = worker._build_group_layer_builders()[0]
    shared_dcp = dcp_size > 1 and put_step > 1
    stride = (8 if shared_dcp else 7) * mib
    allocated = worker._global_group_alloc_size(0)
    assert allocated == stride * (dcp_size if shared_dcp else 1)
    assert builder.page_size_bytes == allocated
    assert builder.layer_byte_offset == (dcp_rank * stride if shared_dcp else 0)

    base = 1_000_000_000
    for layer in range(7):
        _, sizes, gvas = builder._build_transfer_arrays(np.array([1]), np.array([base]), layer)
        assert gvas.tolist() == [
            base + builder.layer_byte_offset + layer * mib,
            base + builder.layer_byte_offset + layer * mib + 3 * mib // 4,
        ]
        assert np.all(gvas + sizes <= base + builder.layer_byte_offset + stride)


@pytest.mark.parametrize(
    ("layer_entries", "offset"),
    [
        ([[83968]] * 37, 42),
        ([[83968]] * 39, 42),
        ([[83968, 128 + i] for i in range(36)] + [[83968, 17]], 42),
        ([[83968]] * 42, 0),
    ],
    ids=["one-mtp", "multiple-mtp", "unequal-cache-entries", "first-stage-control"],
)
@pytest.mark.parametrize("block_id", [0, 1])
def test_pp_physical_copies_fit_allocated_object(layer_entries, offset, block_id):
    worker = _worker(layer_entries, offset=offset)
    builder = worker._build_group_layer_builders()[0]
    allocated = worker._global_group_alloc_size(0)
    base = 1_000_000_000
    copied_bytes = 0
    for layer, entries in enumerate(layer_entries):
        addrs, sizes, gvas = builder._build_transfer_arrays(np.array([block_id]), np.array([base]), layer)
        assert len(gvas) == len(entries)
        assert np.all(gvas >= base)
        assert np.all(gvas + sizes <= base + allocated)
        start = worker.group_layer_cache_entry_offsets[0][layer]
        expected = [
            worker.group_kv_caches_base_addr[0][start + i] + block_id * worker.group_block_stride[0][start + i]
            for i in range(len(entries))
        ]
        assert addrs.tolist() == expected
        copied_bytes += int(sizes.sum())
    assert copied_bytes == sum(worker.group_block_len[0])
    assert worker.layerwise_key_layers == len(layer_entries)
    assert builder.page_size_bytes == allocated
    assert builder.layer_byte_offset == 0
    assert allocated == copied_bytes


@pytest.mark.parametrize("dcp_rank", range(4))
def test_dcp_stride_aligns_stage_payload_including_mtp(dcp_rank):
    mib = 1024 * 1024
    worker = _worker([[mib]] * 37, dcp_size=4, dcp_rank=dcp_rank)
    builder = worker._build_group_layer_builders()[0]
    allocated = worker._global_group_alloc_size(0)
    assert allocated == 4 * 38 * mib
    stride = allocated // 4
    base = 1_000_000_000
    for layer in range(37):
        _, sizes, gvas = builder._build_transfer_arrays(np.array([0]), np.array([base]), layer)
        assert np.all(gvas >= base + dcp_rank * stride)
        assert np.all(gvas + sizes <= base + (dcp_rank + 1) * stride)
    assert builder.layer_byte_offset == dcp_rank * stride


def test_multigroup_pp_objects_pack_only_each_local_payload():
    worker = _worker([[32]] * 16, offset=32, total_layers=61)
    worker.num_layers = 32
    worker.num_kv_cache_groups = 2
    worker.group_num_layers = {0: 16, 1: 16}
    worker.group_block_len[1] = [64] * 16
    worker.group_block_stride[1] = [128] * 16
    worker.group_kv_caches_base_addr[1] = [30_000_000 + i * 1000 for i in range(16)]
    worker.group_layer_cache_entry_offsets[1] = list(range(17))
    builders = worker._build_group_layer_builders()
    for group, size in enumerate([32, 64]):
        assert builders[group].page_size_bytes == 16 * size
        assert builders[group].layer_byte_offset == 0
    assert worker.layerwise_key_layers == 32


def test_block_key_layout_keeps_stage_local_object():
    worker = _worker([[83968]] * 37, block_key=True)
    builder = worker._build_group_layer_builders()[0]
    assert builder.page_size_bytes == 37 * 83968
    assert builder.layer_byte_offset == 0


@pytest.mark.parametrize("total_layers", [None, 0])
def test_missing_global_metadata_preserves_local_fallback(total_layers):
    worker = _worker([[13, 7], [29]], total_layers=total_layers)
    assert worker._global_group_alloc_size(0) == 49


def test_pp_allocation_uses_exact_entry_lengths():
    worker = _worker([[2], [1]], offset=0, total_layers=2)
    assert worker._global_group_alloc_size(0) == 3


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_pp_payload_round_trips_between_different_local_blocks(dcp_size):
    """Stage-local objects restore every entry without overwriting HBM gaps."""
    entries = [[13, 7], [29]]
    transfers = []
    for rank in range(dcp_size):
        writer = _worker(entries, offset=23, total_layers=45, dcp_size=dcp_size, dcp_rank=rank)
        reader = _worker(entries, offset=23, total_layers=45, dcp_size=dcp_size, dcp_rank=rank)
        lengths = writer.group_block_len[0]
        source = [np.arange(6 * size, dtype=np.uint8) + rank * 50 + i * 7 for i, size in enumerate(lengths)]
        restored = [np.full_like(data, 255) for data in source]
        writer.group_kv_caches_base_addr[0] = [data.ctypes.data for data in source]
        reader.group_kv_caches_base_addr[0] = [data.ctypes.data for data in restored]
        transfers.append(
            (writer._build_group_layer_builders()[0], reader._build_group_layer_builders()[0], source, restored)
        )

    allocated = writer._global_group_alloc_size(0)
    pool = np.full(allocated, 255, dtype=np.uint8)
    base = pool.ctypes.data
    for writer_builder, _, source, _ in transfers:
        for layer in range(len(entries)):
            addrs, sizes, gvas = writer_builder._build_transfer_arrays(np.array([1]), np.array([base]), layer)
            assert np.all(gvas >= base)
            assert np.all(gvas + sizes <= base + allocated)
            for addr, size, gva in zip(addrs, sizes, gvas, strict=True):
                ctypes.memmove(int(gva), int(addr), int(size))

    for _, reader_builder, source, restored in transfers:
        for layer in range(len(entries)):
            addrs, sizes, gvas = reader_builder._build_transfer_arrays(np.array([2]), np.array([base]), layer)
            assert np.all(gvas >= base)
            assert np.all(gvas + sizes <= base + allocated)
            for addr, size, gva in zip(addrs, sizes, gvas, strict=True):
                ctypes.memmove(int(addr), int(gva), int(size))
        for data, result, size in zip(source, restored, lengths, strict=True):
            expected = np.full_like(result, 255)
            expected[4 * size : 5 * size] = data[2 * size : 3 * size]
            np.testing.assert_array_equal(result, expected)
    assert allocated == (2 * 1024 * 1024 * dcp_size if dcp_size > 1 else sum(lengths))
