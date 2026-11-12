# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the selected logical block must survive a raw-byte round trip."""

import ctypes

import numpy as np
import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_segmented_cache_transfer import make_database, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData


def components(scale, padding, dtype, tokens=2, nope=8, rope=1):
    slot = tokens * (nope + rope) + padding
    backing = torch.full(((3 * scale + 2) * slot,), -1, dtype=dtype)
    views = tuple(
        torch.as_strided(backing, (3 * scale, tokens, 1, width), (slot, width, width, 1), storage_offset=slot + offset)
        for width, offset in [(nope, 0), (rope, tokens * nope)]
    )
    return backing, views


def addresses(database, mode, scale, tokens, token_count=None, layer_id=0, token_offset=0):
    start = scale * tokens
    start += token_offset
    end = start + (scale * tokens if token_count is None else token_count)
    if mode == "bulk":
        addr, size, _ = database.prepare_value(start, end, [0, 1])
        return addr, size, None
    if mode == "legacy":
        addr, size, _ = database.prepare_value_layer(start, end, [0, 1], layer_id)
        return addr, size, None
    builder = LayerBatchBuilder(database, sum(database.group_block_len[0]), database.group_num_layers["kv"][0])
    shared = SharedBlockData(
        block_ids_arr=np.array([1], dtype=np.int64),
        block_gvas_arr=np.array([4096], dtype=np.int64) if mode == "gva" else None,
        req_ids=["regression"],
        is_last_chunks=[True],
        block_keys=["regression-key"] if mode == "block_key" else None,
    )
    result = builder.build_addrs(shared, layer_id)
    if mode == "gva":
        return result.addr_array.tolist(), result.size_array.tolist(), result.gvas_array.tolist()
    return result.all_buffers[0], result.all_sizes[0], result.all_offsets[0]


def roundtrip(mode, scale, padding, dtype, tokens=2, nope=8, rope=1, token_count=None, token_offset=0):
    source, source_views = components(scale, padding, dtype, tokens, nope, rope)
    dest, dest_views = components(scale, padding, dtype, tokens, nope, rope)
    expected, expected_views = components(scale, padding, dtype, tokens, nope, rope)
    for i, view in enumerate(source_views):
        view.copy_(torch.arange(view.numel()).reshape(view.shape).remainder(97) + i * 100)

    def database(views):
        worker = make_worker({"model.layers.0.self_attn.attn": views}, layerwise=mode != "bulk")
        # The helper uses 2 tokens/physical block. Here the database's logical
        # block size is explicit, including the 5760/128 hardware-shaped case.
        db = make_database(worker, scale)
        db.block_size = [scale * tokens]
        return db

    src_db, dst_db = database(source_views), database(dest_views)
    addr, sizes, offsets = addresses(src_db, mode, scale, tokens, token_count, token_offset=token_offset)
    dst_addr, dst_sizes, dst_offsets = addresses(dst_db, mode, scale, tokens, token_count, token_offset=token_offset)
    assert sizes == dst_sizes
    assert offsets == dst_offsets
    # Serialize every segment before restoring, just as a pool PUT then GET.
    payload = [ctypes.string_at(a, n) for a, n in zip(addr, sizes)]
    for a, data in zip(dst_addr, payload):
        ctypes.memmove(a, data, len(data))
    count = scale * tokens if token_count is None else token_count
    for sv, ev in zip(source_views, expected_views):
        for token in range(token_offset, token_offset + count):
            slot, offset = divmod(token, tokens)
            ev[scale + slot, offset].copy_(sv[scale + slot, offset])
    # Compare restored payload first, then guards, padding and unselected blocks.
    bad_slots = [
        i
        for i in range(scale)
        if any(not torch.equal(d[scale + i], e[scale + i]) for d, e in zip(dest_views, expected_views))
    ]
    assert not bad_slots, f"{mode}: incorrect physical slots {bad_slots}; descriptors={len(sizes)}"
    assert torch.equal(dest, expected), f"{mode}: modified padding/guards/unselected data"


@pytest.mark.parametrize("mode", ["bulk", "gva", "block_key", "legacy"])
@pytest.mark.parametrize("scale", [1, 3, 45])
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_interleaved_full_block(mode, scale, padding, dtype):
    roundtrip(mode, scale, padding, dtype)


@pytest.mark.parametrize("mode", ["bulk", "gva", "block_key", "legacy"])
def test_5760_token_block_128_token_kernel_bf16(mode):
    roundtrip(mode, 45, 0, torch.bfloat16, tokens=128, nope=512, rope=64)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["bulk", "legacy"])
def test_partial_block(mode, count, padding, dtype):
    roundtrip(mode, 3, padding, dtype, token_count=count)


@pytest.mark.parametrize("offset,count", [(1, 1), (1, 3), (2, 3), (3, 3), (5, 1), (0, 0)])
@pytest.mark.parametrize("mode", ["bulk", "legacy"])
def test_token_interval(mode, offset, count):
    roundtrip(mode, 3, 3, torch.bfloat16, token_count=count, token_offset=offset)


@pytest.mark.parametrize("mode", ["gva", "block_key", "legacy"])
@pytest.mark.parametrize("segmented_layer", [0, 1])
@pytest.mark.parametrize("layer_id", [0, 1])
def test_mixed_layer_descriptor_counts(mode, segmented_layer, layer_id):
    def make(fill):
        backing, views = components(3, 3, torch.bfloat16)
        dense = torch.full((9, 2, 8), fill, dtype=torch.bfloat16)
        if fill != -1:
            for i, view in enumerate(views):
                view.copy_(torch.arange(view.numel()).reshape(view.shape).remainder(97) + i * 100)
        caches = {f"model.layers.{i}.attn": views if i == segmented_layer else (dense,) for i in range(2)}
        worker = make_worker(caches, layerwise=True)
        return (backing, dense), caches, make_database(worker, 3)

    src_storage, src_views, src_db = make(7)
    dst_storage, dst_views, dst_db = make(-1)
    expected_storage, expected_views, _ = make(-1)
    src_addr, sizes, offsets = addresses(src_db, mode, 3, 2, layer_id=layer_id)
    dst_addr, dst_sizes, dst_offsets = addresses(dst_db, mode, 3, 2, layer_id=layer_id)
    assert sizes == dst_sizes
    assert offsets == dst_offsets
    for addr, data in zip(dst_addr, [ctypes.string_at(a, n) for a, n in zip(src_addr, sizes)]):
        ctypes.memmove(addr, data, len(data))
    layer = f"model.layers.{layer_id}.attn"
    for original, expected in zip(src_views[layer], expected_views[layer]):
        expected[3:6].copy_(original[3:6])
    assert all(torch.equal(actual, expected) for actual, expected in zip(dst_storage, expected_storage))


@pytest.mark.parametrize("mode", ["bulk", "gva", "block_key", "legacy"])
def test_dense_scaled_control(mode):
    source = torch.arange(3 * 45 * 2 * 8, dtype=torch.float32).reshape(3 * 45, 2, 8)
    dest = torch.full_like(source, -1)

    def db(tensor):
        return make_database(make_worker({"model.layers.0.attn": (tensor,)}, layerwise=mode != "bulk"), 45)

    a, n, _ = addresses(db(source), mode, 45, 2)
    b, m, _ = addresses(db(dest), mode, 45, 2)
    assert n == m
    for dst, data in zip(b, [ctypes.string_at(p, s) for p, s in zip(a, n)]):
        ctypes.memmove(dst, data, len(data))
    expected = torch.full_like(source, -1)
    expected[45:90].copy_(source[45:90])
    assert torch.equal(dest, expected)
