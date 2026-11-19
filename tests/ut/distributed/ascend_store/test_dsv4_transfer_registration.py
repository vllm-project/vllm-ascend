# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport registration, byte restoration and old pool-format isolation."""

import copy
import ctypes
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.models.test_dsv4_layerwise import make_layout
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker


def make_indexer_views(scale_bytes=4, padding=0):
    # A5 has a 4096-byte data payload followed by 128 bytes of float scales.
    payload = 32 * (128 + scale_bytes)
    stride = payload + padding
    raw = torch.arange(4 * stride, dtype=torch.int64).to(torch.uint8)
    data = raw.as_strided((4, 32, 1, 128), (stride, 128, 128, 1))
    dtype = torch.float32 if scale_bytes == 4 else torch.float16
    scale = raw.view(dtype).as_strided((4, 32, 1, 1), (stride // scale_bytes, 1, 1, 1), 4096 // scale_bytes)
    full = raw.as_strided((4, 32, 1, 128 + scale_bytes), (stride, 128 + scale_bytes, 128 + scale_bytes, 1))
    return data, scale, full


def register_transport(caches, *, dsv4=True, gva=True):
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(dsv4_cache_format="format") if dsv4 else SimpleNamespace()
    worker.use_layerwise_transfer = gva
    worker.use_recoverssm = False
    worker._set_num_blocks = Mock()
    worker._register_dspark_prefix_cache = Mock()
    worker._register_kv_caches = Mock()
    worker.register_kv_caches(caches)
    return worker._register_kv_caches.call_args.args[0]


def test_layerwise_registration_preserves_kernel_views_and_other_groups():
    indexer = make_indexer_views()
    swa = torch.zeros(4, 32, 1, 512, dtype=torch.bfloat16)
    state = torch.zeros(4, 2, 1, 2048, dtype=torch.float32)
    caches = {
        "model.layers.2.self_attn.indexer.k_cache": indexer,
        "model.layers.2.self_attn.swa_cache": swa,
        "model.layers.2.self_attn.compressor.state_cache": state,
    }
    transferred = register_transport(caches)
    assert transferred is not caches
    assert caches["model.layers.2.self_attn.indexer.k_cache"] is indexer
    assert len(indexer) == 3
    data, scale = transferred["model.layers.2.self_attn.indexer.k_cache"]
    assert data is indexer[0] and scale is indexer[1]
    assert transferred["model.layers.2.self_attn.swa_cache"] is swa
    assert transferred["model.layers.2.self_attn.compressor.state_cache"] is state


def test_saved_indexer_restores_full_kernel_view_after_source_reuse():
    name = "model.layers.2.self_attn.indexer.k_cache"
    source = make_indexer_views()
    expected = source[2].clone()
    views = register_transport({name: source})[name]
    # Immutable CPU copies stand in for the external byte store. Mutating the
    # source models the layer's HBM being reused before a second request loads.
    pages = [torch.cat([view[page].view(torch.uint8).reshape(-1) for view in views]) for page in range(4)]
    assert all(page.numel() == 4224 for page in pages)
    source[2].zero_()
    destination = make_indexer_views()
    destination[2].zero_()
    for page, payload in enumerate(pages):
        offset = 0
        for view in register_transport({name: destination})[name]:
            target = view[page].view(torch.uint8).reshape(-1)
            target.copy_(payload[offset : offset + target.numel()])
            offset += target.numel()
    assert torch.equal(destination[2], expected)
    assert torch.count_nonzero(source[2]) == 0


def test_non_dsv4_registration_keeps_existing_three_component_behavior():
    caches = {"other_model.cache": make_indexer_views()}
    assert register_transport(caches, dsv4=False) is caches


def test_block_key_layerwise_store_keeps_ordinary_wire_offsets():
    caches = {"model.layers.2.self_attn.indexer.k_cache": make_indexer_views()}
    assert register_transport(caches, gva=False) is caches


def test_group_metadata_and_pool_size_count_indexer_payload_once():
    caches = {
        "model.layers.2.self_attn.attn": torch.zeros(4, 32, 1, 512, dtype=torch.bfloat16),
        "model.layers.2.self_attn.indexer.k_cache": make_indexer_views(),
    }
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_caches = register_transport(caches)
    worker.kv_cache_config = None
    worker.hf_config = SimpleNamespace(num_hidden_layers=43)
    worker.num_blocks, worker.num_layers, worker.total_layers = 4, 43, 43
    worker.use_layerwise, worker.use_kvpp = True, False
    worker.put_step, worker.dcp_size, worker.pp_rank = 8, 1, 0
    worker._global_to_local_layer = {}
    worker.group_kv_caches_base_addr, worker.group_block_len, worker.group_block_stride = {}, {}, {}
    worker.group_layer_cache_entry_offsets, worker.group_cache_entry_segments, worker.group_num_layers = {}, {}, {}
    worker._infer_cache_group_metadata(0, list(caches))
    assert worker.group_block_len[0] == [32768, 4096, 128]
    assert worker.group_block_stride[0] == [32768, 4224, 4224]
    assert worker.group_layer_cache_entry_offsets[0] == [0, 3]
    assert worker._global_group_alloc_size(0) == 43 * (32768 + 4096 + 128)


def test_invalid_indexer_alias_is_rejected_before_transport_registration():
    data, scale, full = make_indexer_views()
    with pytest.raises(ValueError, match="not disjoint"):
        register_transport({"model.layers.2.self_attn.indexer.k_cache": (data, scale, full.clone())})


def test_ordinary_store_preserves_legacy_registration_and_object_offsets():
    caches = {"model.layers.2.self_attn.indexer.k_cache": make_indexer_views()}
    worker = SimpleNamespace(
        use_recoverssm=False,
        kv_cache_config=SimpleNamespace(dsv4_cache_format="v2"),
        _set_num_blocks=Mock(),
        _register_kv_caches=Mock(),
    )
    BaseKVPoolWorker.register_kv_caches(worker, caches)
    registered = worker._register_kv_caches.call_args.args[0]
    assert registered is caches
    assert sum(t[0].numel() * t.element_size() for t in next(iter(registered.values()))) == 8448

    # A following member detects shifts in the object layout in either
    # direction, even when the indexer itself appears to round-trip.
    def encode(views, trailer):
        return b"".join(ctypes.string_at(v.data_ptr(), v[0].numel() * v.element_size()) for v in views) + trailer

    def decode(obj, views):
        cursor = 0
        for view in views:
            length = view[0].numel() * view.element_size()
            ctypes.memmove(view.data_ptr(), obj[cursor : cursor + length], length)
            cursor += length
        return obj[cursor:]

    indexer = next(iter(caches.values()))
    indexer[0].fill_(31)
    indexer[1].fill_(7)
    expected = indexer[2][0].clone()
    for old_writer in (True, False):
        target = make_indexer_views()
        target[2].zero_()
        old_order = tuple(indexer) if old_writer else target
        if old_writer:
            obj = encode(old_order, b"next")
            BaseKVPoolWorker.register_kv_caches(worker, {"indexer": target})
            readers = worker._register_kv_caches.call_args.args[0]["indexer"]
        else:
            obj = encode(next(iter(registered.values())), b"next")
            readers = old_order
        assert decode(obj, readers) == b"next"
        assert torch.equal(target[2][0], expected)


@pytest.mark.parametrize("scale_bytes", [2, 4])
@pytest.mark.parametrize("padding", [0, 64])
@pytest.mark.parametrize("first,last", [(0, 16), (16, 32), (0, 32)])
def test_subpage_copies_only_selected_data_and_scale_rows(scale_bytes, padding, first, last):
    name = "model.layers.2.self_attn.indexer.k_cache"
    source = make_indexer_views(scale_bytes, padding)
    destination = make_indexer_views(scale_bytes, padding)
    source[0].fill_(17)
    source[1].fill_(5)
    length = source[2].untyped_storage().nbytes()
    ctypes.memset(destination[2].data_ptr(), 0xA5, length)
    expected = bytearray([0xA5] * length)
    stride = 4096 + 32 * scale_bytes + padding
    for start, end in ((first * 128, last * 128), (4096 + first * scale_bytes, 4096 + last * scale_bytes)):
        offset = stride + start
        expected[offset : stride + end] = ctypes.string_at(source[2].data_ptr() + offset, end - start)

    def segments(views):
        database = ChunkedTokenDatabase([], [32], None)
        members = register_transport({name: views})[name]
        database.group_kv_caches_base_addr[0] = [v.data_ptr() for v in members]
        database.group_block_len[0] = [v[0].numel() * v.element_size() for v in members]
        database.group_block_stride[0] = [v.stride(0) * v.element_size() for v in members]
        return database.prepare_value(first, last, [1])[:2]

    src, sizes = segments(source)
    dst, dest_sizes = segments(destination)
    assert sizes == dest_sizes
    for a, b, size in zip(src, dst, sizes, strict=True):
        ctypes.memmove(b, a, size)
    assert ctypes.string_at(destination[2].data_ptr(), length) == expected


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
def test_v1_objects_are_isolated_but_resident_and_rotating_v2_keys_match(model, cache_format):
    keys = []
    for buffers, blocks, independent in ((None, 16, []), (1, 8, []), (3, 32, [0, 1]), (4, 64, [0, 1, 4])):
        runtime, groups, _ = make_layout(model, cache_format=cache_format, buffers=buffers, independent=independent)
        if buffers is None:
            runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
        config = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=blocks)
        # Reconstruct the published v1 format identity, independently of v2.
        old_groups = [
            [
                (name, type(spec).__name__, asdict(spec))
                for name in group.layer_names
                for spec in (group.kv_cache_spec.kv_cache_specs[name],)
            ]
            for group in config.kv_cache_groups
        ]
        legacy = copy.copy(config)
        payload = json.dumps(("dsv4-components-v1", old_groups), default=str, sort_keys=True, separators=(",", ":"))
        legacy.dsv4_cache_format = hashlib.sha256(payload.encode()).hexdigest()[:24]
        assert legacy.dsv4_cache_format != config.dsv4_cache_format
        pair = []
        for item in (legacy, config):
            bound = MemcacheBackend.bind_layerwise_keys(
                vllm_config=runtime,
                kv_cache_config=item,
                model_name=model,
                use_hybrid=True,
                grouped_block_size=[128] * len(groups),
            )
            pair.append([bound.make_hit_check_keys(group, "same-prefix-hash", 1) for group in range(len(groups))])
        assert all(old != new for old, new in zip(*pair))
        keys.append(pair[1])
    assert all(key_set == keys[0] for key_set in keys[1:])
