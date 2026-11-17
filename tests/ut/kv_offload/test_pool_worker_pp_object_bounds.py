# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All physical layer copies must fit the GVA object allocated for their key."""

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


def _worker(layer_entries, offset=42, total_layers=78, dcp_size=1, dcp_rank=0, block_key=False, kvpp=False, pp_size=2):
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
    worker.use_kvpp = kvpp
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


@pytest.mark.parametrize("rank", range(16))
@pytest.mark.parametrize(("pp_size", "offset"), [(1, 0), (2, 0), (2, 42)])
@pytest.mark.parametrize("block_id", [0, 1])
def test_kvpp_gva_uses_owner_indices_and_payload(rank, pp_size, offset, block_id):
    count = 4 + int(rank < 14)
    first = rank * 4 + min(rank, 14)
    entries = [[4096, 8192] for _ in range(count)]
    worker = _worker(entries, offset=offset, total_layers=156, kvpp=True, pp_size=pp_size)
    worker.num_layers = 78
    worker.physical_layer_to_group_layers = {layer: [(0, i)] for i, layer in enumerate(range(first, first + count))}
    builder = worker._build_group_layer_builders()[0]
    allocated = worker._global_group_alloc_size(0)
    payload_bytes = count * 12288
    byte_origin = 0 if pp_size == 1 else offset * 12288
    assert allocated == (payload_bytes if pp_size == 1 else 156 * 12288)
    assert builder.layer_byte_offset == byte_origin
    assert worker._final_layer_for_transfer(78) == first + count - 1

    base = 1_000_000_000
    copied_bytes = 0
    for layer in range(78):
        groups = worker._groups_for_layerwise_transfer(layer)
        if not first <= layer < first + count:
            assert groups == []
            continue
        assert groups == [(0, layer - first)]
        _, dense = groups[0]
        addrs, sizes, gvas = builder._build_transfer_arrays(np.array([block_id]), np.array([base]), dense)
        start = dense * 2
        assert addrs.tolist() == [
            worker.group_kv_caches_base_addr[0][start + i] + block_id * worker.group_block_stride[0][start + i]
            for i in range(2)
        ]
        assert gvas.tolist() == [base + byte_origin + dense * 12288, base + byte_origin + dense * 12288 + 4096]
        assert np.all(gvas + sizes <= base + allocated)
        copied_bytes += int(sizes.sum())
    assert copied_bytes == payload_bytes


def test_non_kvpp_gva_keeps_layer_fallback_and_completion():
    worker = _worker([[4096, 8192]], offset=0)
    worker.physical_layer_to_group_layers = {}
    assert worker._groups_for_layerwise_transfer(7) == [(0, 7)]
    assert worker._final_layer_for_transfer(8) == 7


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
    # The byte origin is unchanged; only an insufficient object grows.
    assert builder.layer_byte_offset == offset * (copied_bytes // len(layer_entries))


@pytest.mark.parametrize("dcp_rank", range(4))
def test_dcp_stride_covers_extended_stage_before_alignment(dcp_rank):
    mib = 1024 * 1024
    worker = _worker([[mib]] * 37, dcp_size=4, dcp_rank=dcp_rank)
    builder = worker._build_group_layer_builders()[0]
    allocated = worker._global_group_alloc_size(0)
    # 78 MiB formerly aligned exactly; the physical stage needs 79 MiB.
    assert allocated == 4 * 80 * mib
    stride = allocated // 4
    base = 1_000_000_000
    for layer in range(37):
        _, sizes, gvas = builder._build_transfer_arrays(np.array([0]), np.array([base]), layer)
        assert np.all(gvas >= base + dcp_rank * stride)
        assert np.all(gvas + sizes <= base + (dcp_rank + 1) * stride)
    assert builder.layer_byte_offset == dcp_rank * stride + 42 * mib


def test_multigroup_layout_with_sufficient_capacity_is_unchanged():
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
        assert builders[group].page_size_bytes == 61 * size
        assert builders[group].layer_byte_offset == 32 * size
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


def test_existing_round_up_rule_is_preserved():
    worker = _worker([[2], [1]], offset=0, total_layers=2)
    assert worker._global_group_alloc_size(0) == 4
