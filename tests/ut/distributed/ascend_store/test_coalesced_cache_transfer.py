# SPDX-License-Identifier: Apache-2.0
"""Bulk coalescing must reduce descriptors without touching adjacent bytes."""

import ctypes

import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_layerwise_strided_roundtrip import components
from tests.ut.distributed.ascend_store.test_segmented_cache_transfer import make_database, make_worker


def bulk_worker(views, scale, tokens, **overrides):
    caches = {"model.layers.0.attn": views}
    worker = make_worker(caches)
    worker._extra_config = {}
    worker.grouped_block_size = [scale * tokens]
    worker.cache_transfer_granularity = scale * tokens
    worker.physical_layer_to_group_layers = {}
    for key, value in overrides.items():
        setattr(worker, key, value)
    worker._infer_cache_group_metadata(0, list(caches))
    return worker


@pytest.mark.parametrize("scale", [3, 45])
@pytest.mark.parametrize("source_padding,dest_padding", [(0, 0), (3, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_combined_bulk_roundtrip(scale, source_padding, dest_padding, dtype, monkeypatch):
    tokens, nope, rope = 128, 512, 64
    source, source_views = components(scale, source_padding, dtype, tokens, nope, rope)
    dest, dest_views = components(scale, dest_padding, dtype, tokens, nope, rope)
    expected, expected_views = components(scale, dest_padding, dtype, tokens, nope, rope)
    for index, view in enumerate(source_views):
        view.copy_(torch.arange(view.numel()).reshape(view.shape).remainder(97) + index * 100)
    for src, exp in zip(source_views, expected_views):
        exp[scale : 2 * scale].copy_(src[scale : 2 * scale])

    databases = []
    for views, padding in [(source_views, source_padding), (dest_views, dest_padding)]:
        worker = bulk_worker(views, scale, tokens)
        expected_count = scale if padding else 1
        assert len(worker.group_block_len[0]) == expected_count
        assert sum(worker.group_block_len[0]) == scale * tokens * (nope + rope) * views[0].element_size()
        # Prove that the combined branch, rather than the segmented fallback,
        # constructs the descriptors during normal metadata registration.
        monkeypatch.setattr(worker, "_get_cache_transfer_segments", lambda _: pytest.fail("unexpected fallback"))
        worker._infer_cache_group_metadata(0, list(worker.kv_caches))
        db = make_database(worker, scale)
        db.block_size = [scale * tokens]
        databases.append(db)

    src_addr, src_size, _ = databases[0].prepare_value(scale * tokens, 2 * scale * tokens, [0, 1])
    dst_addr, dst_size, _ = databases[1].prepare_value(scale * tokens, 2 * scale * tokens, [0, 1])
    payload = b"".join(ctypes.string_at(address, size) for address, size in zip(src_addr, src_size))
    cursor = 0
    for address, size in zip(dst_addr, dst_size):
        ctypes.memmove(address, payload[cursor : cursor + size], size)
        cursor += size
    assert cursor == len(payload)
    assert torch.equal(dest, expected), "copied padding, guards, or an unselected block"
    assert torch.equal(source_views[0][scale : 2 * scale], dest_views[0][scale : 2 * scale])
    assert torch.equal(source_views[1][scale : 2 * scale], dest_views[1][scale : 2 * scale])


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_layerwise": True},
        {"_extra_config": {"discard_partial_chunks": False}},
        {"_extra_config": {"save_decode_cache": True}},
        {"tp_mismatch": True},
        {"dcp_size": 2},
        {"use_kvpp": True},
        {"cache_transfer_granularity": 5},
        {"group_uses_align_state": [True]},
        {"_extra_config": None},
    ],
)
def test_unsupported_transfer_modes_keep_segmented_layout(overrides):
    _, views = components(3, 3, torch.bfloat16)
    worker = bulk_worker(views, 3, 2, **overrides)
    expected = [segment for cache in views for segment in worker._get_cache_transfer_segments(cache)]
    actual = list(zip(worker.group_kv_caches_base_addr[0], worker.group_block_len[0], worker.group_block_stride[0]))
    assert actual == expected
    assert worker.group_cache_entry_segments[0] == [(0, 3), (1, 3), (2, 3)] * 2


@pytest.mark.parametrize("layout", ["separate_storage", "reversed", "gap", "different_shape", "different_dtype"])
def test_incompatible_components_do_not_coalesce(layout):
    _, views = components(3, 3, torch.float32)
    first, second = views
    if layout == "separate_storage":
        _, other = components(3, 3, torch.float32)
        second = other[1]
    elif layout == "reversed":
        first, second = second, first
    elif layout == "gap":
        second = torch.as_strided(second, second.shape, second.stride(), second.storage_offset() + 1)
    elif layout == "different_shape":
        second = second[:, :1]
    else:
        second = second.to(torch.bfloat16)
    worker = bulk_worker((first, second), 3, 2)
    assert worker._get_combined_cache_transfer_segments((first, second), 0) is None


def test_dense_or_single_component_retains_existing_descriptors():
    dense = torch.zeros(9, 2, 8)
    worker = bulk_worker((dense,), 3, 2)
    assert worker.group_kv_caches_base_addr[0] == [dense.data_ptr()]
    assert worker.group_block_len[0] == [3 * dense[0].numel() * dense.element_size()]


def test_ratio_one_retains_existing_descriptors():
    _, views = components(1, 0, torch.bfloat16)
    worker = bulk_worker(views, 1, 2)
    assert len(worker.group_block_len[0]) == 2


def test_mixed_layer_descriptor_offsets():
    _, views = components(3, 0, torch.bfloat16)
    worker = bulk_worker(views, 3, 2)
    worker.kv_caches["model.layers.1.attn"] = (torch.zeros(9, 2, 8),)
    worker._infer_cache_group_metadata(0, list(worker.kv_caches))
    assert worker.group_layer_cache_entry_offsets[0] == [0, 1, 2]
    assert worker.group_num_layers[0] == 2
