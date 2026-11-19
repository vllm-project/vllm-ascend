# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise actual pool address generation against interleaved CPU storage."""

import ctypes
from types import SimpleNamespace

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

NUM_BLOCKS = 3
TOKENS_PER_PHYSICAL_BLOCK = 2


def make_worker(caches, *, layerwise=False):
    worker = object.__new__(KVPoolWorker)
    worker.num_blocks = NUM_BLOCKS
    worker.use_layerwise = layerwise
    worker.kv_caches = caches
    worker.hf_config = SimpleNamespace(num_hidden_layers=2)
    worker.num_layers = 2
    worker._global_to_local_layer = {}
    for name in (
        "group_kv_caches_base_addr",
        "group_block_len",
        "group_block_stride",
        "group_layer_cache_entry_offsets",
        "group_cache_entry_segments",
        "group_num_layers",
    ):
        setattr(worker, name, {})
    worker._infer_cache_group_metadata(0, list(caches))
    return worker


def make_database(worker, scale):
    database = ChunkedTokenDatabase([], [scale * TOKENS_PER_PHYSICAL_BLOCK], None)
    database.set_group_buffers(
        worker.group_kv_caches_base_addr,
        worker.group_block_len,
        worker.group_block_stride,
        group_num_layers=worker.group_num_layers,
        group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        group_cache_entry_segments=worker.group_cache_entry_segments,
    )
    return database


def make_components(scale, padding, dtype):
    # Each physical slot contains [NoPE][RoPE][padding]. Use a nonzero
    # storage offset and guard slots to catch out-of-region writes as well.
    nope_width, rope_width = 4, 2
    nope_elements = TOKENS_PER_PHYSICAL_BLOCK * nope_width
    rope_elements = TOKENS_PER_PHYSICAL_BLOCK * rope_width
    slot_elements = nope_elements + rope_elements + padding
    slots = NUM_BLOCKS * scale
    backing = torch.full(((slots + 2) * slot_elements,), -1, dtype=dtype)
    components = tuple(
        torch.as_strided(
            backing,
            (slots, TOKENS_PER_PHYSICAL_BLOCK, 1, width),
            (slot_elements, width, width, 1),
            storage_offset=slot_elements + component_offset,
        )
        for width, component_offset in ((nope_width, 0), (rope_width, nope_elements))
    )
    return backing, components


@pytest.mark.parametrize("scale", [1, 3, 45])
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pool_roundtrip_preserves_components_and_guards(scale, padding, dtype):
    source, source_components = make_components(scale, padding, dtype)
    destination, destination_components = make_components(scale, padding, dtype)
    for component, view in enumerate(source_components):
        view.copy_(torch.arange(view.numel()).reshape(view.shape).remainder(97) + component * 100)
    layer = "model.layers.0.self_attn.attn"
    source_worker = make_worker({layer: source_components})
    destination_worker = make_worker({layer: destination_components})
    source_db = make_database(source_worker, scale)
    destination_db = make_database(destination_worker, scale)
    selected_block = 1
    start = selected_block * scale * TOKENS_PER_PHYSICAL_BLOCK
    end = start + scale * TOKENS_PER_PHYSICAL_BLOCK
    addresses, sizes, _ = source_db.prepare_value(start, end, [0, selected_block])
    restored_addresses, restored_sizes, _ = destination_db.prepare_value(start, end, [0, selected_block])
    assert sizes == restored_sizes
    # Model a backend PUT followed by GET, rather than tensor indexing, so a
    # wrong contiguous byte span reproduces the production corruption.
    payload = [ctypes.string_at(address, size) for address, size in zip(addresses, sizes)]
    for address, data in zip(restored_addresses, payload):
        ctypes.memmove(address, data, len(data))
    expected, expected_components = make_components(scale, padding, dtype)
    selected = slice(selected_block * scale, (selected_block + 1) * scale)
    for original, restored, reference in zip(source_components, destination_components, expected_components):
        assert torch.equal(restored[selected], original[selected])
        reference[selected].copy_(original[selected])
    assert torch.equal(destination, expected), "Padding, guard slots and unselected blocks must remain untouched"
    assert source_worker.group_num_layers[0] == 1
    assert source_worker.group_layer_cache_entry_offsets[0] == [0, len(addresses)]
    assert sum(sizes) == sum(view[selected].numel() * view.element_size() for view in source_components)


def test_dense_scaled_cache_keeps_one_descriptor():
    cache = torch.zeros(NUM_BLOCKS * 3, 2, 4)
    worker = make_worker({"model.layers.0.attn": (cache,)})
    assert worker.group_kv_caches_base_addr[0] == [cache.data_ptr()]
    assert worker.group_block_len[0] == [cache[0].numel() * cache.element_size() * 3]


def test_registration_covers_last_physical_payload():
    _, components = make_components(45, 3, torch.float32)
    worker = make_worker({"model.layers.0.attn": components})
    for cache in components:
        _, _, span, _ = worker._get_cache_block_metadata(cache)
        expected = ((cache.shape[0] - 1) * cache.stride(0) + cache[0].numel()) * cache.element_size()
        assert span == expected


def test_layer_offsets_cover_different_segment_counts():
    _, components = make_components(3, 3, torch.float32)
    dense = torch.zeros(NUM_BLOCKS * 3, 2, 4)
    worker = make_worker({"model.layers.1.attn": components, "model.layers.0.attn": (dense,)})
    assert worker.group_num_layers[0] == 2
    assert worker.group_layer_cache_entry_offsets[0] == [0, 1, 7]


def test_layerwise_descriptors_preserve_physical_blocks():
    _, components = make_components(3, 3, torch.float32)
    worker = make_worker({"model.layers.0.attn": components}, layerwise=True)
    assert worker.group_layer_cache_entry_offsets[0] == [0, 6]
    assert worker.group_cache_entry_segments[0] == [(0, 3), (1, 3), (2, 3)] * 2


@pytest.mark.parametrize("layerwise", [False, True])
def test_segmented_transfer_rejects_noncontiguous_inner_block(layerwise):
    cache = torch.zeros(NUM_BLOCKS * 3, 2, 8)[..., ::2]
    with pytest.raises(ValueError, match="contiguous data inside each physical block"):
        make_worker({"model.layers.0.attn": (cache,)}, layerwise=layerwise)
