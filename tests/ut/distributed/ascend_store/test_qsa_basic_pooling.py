# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bulk QSA pooling contracts shared by memcache and Mooncake."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from tests.ut.distributed.ascend_store import test_pool_scheduler as scheduler_tests
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class BufferStore:
    """Copy bytes into independent objects; never retain source addresses."""

    def __init__(self, buffers):
        self.objects = {}
        self.bounds = [(array.ctypes.data, array.ctypes.data + array.nbytes) for array in buffers]
        self.get_calls = 0

    def _check_range(self, address, size):
        assert size > 0
        assert any(start <= address and address + size <= end for start, end in self.bounds)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            for address, size in zip(pointers, lengths, strict=True):
                self._check_range(address, size)
            self.objects[key] = b"".join(ctypes.string_at(p, n) for p, n in zip(pointers, lengths, strict=True))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.get_calls += 1
        results = []
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            value = self.objects.get(key)
            if value is None:
                results.append(-1)
                continue
            assert len(value) == sum(lengths)
            offset = 0
            for address, size in zip(pointers, lengths, strict=True):
                self._check_range(address, size)
                ctypes.memmove(address, value[offset : offset + size], size)
                offset += size
            results.append(offset)
        return results

    def batch_put_from_layers(self, keys, addresses, sizes, direction):
        return self.batch_put_from_multi_buffers(keys, addresses, sizes, None)

    def batch_get_into_layers(self, keys, addresses, sizes, direction):
        return [0 if result >= 0 else result for result in self.batch_get_into_multi_buffers(keys, addresses, sizes)]


def make_backend(backend_name, store):
    backend_type = MooncakeBackend if backend_name == "mooncake" else MemcacheBackend
    backend = backend_type.__new__(backend_type)
    backend.store = store
    backend._lazy_init = False
    backend._store_initialized = True
    backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=True)
    return backend


def make_six_region_database(backing, padding):
    payloads = [32, 48, 64, 16, 80, 96]
    strides = [payload + padding for payload in payloads]
    offsets = []
    cursor = 64
    for stride in strides:
        offsets.append(cursor)
        cursor += stride * 5 + 64
    assert cursor <= backing.nbytes
    database = ChunkedTokenDatabase([KeyMetadata("next", 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [backing.ctypes.data + offset for offset in offsets]},
        {0: payloads},
        {0: strides},
    )
    return database, offsets, strides, payloads


@pytest.mark.parametrize("backend_name", ["memcache", "mooncake"])
@pytest.mark.parametrize("padding", [0, 64])
@pytest.mark.parametrize("tokens", [8, 16])
def test_six_region_restore_uses_external_bytes_and_preserves_other_owners(backend_name, padding, tokens):
    source = np.full(8192, 90, dtype=np.uint8)
    destination = np.full_like(source, 90)
    database, offsets, strides, payloads = make_six_region_database(source, padding)
    target, _, _, _ = make_six_region_database(destination, padding)
    for region, (offset, stride, payload) in enumerate(zip(offsets, strides, payloads, strict=True)):
        for block in range(5):
            start = offset + block * stride
            source[start : start + payload] = (np.arange(payload) + region * 17 + block * 29) % 256

    native = BufferStore([source, destination])
    backend = make_backend(backend_name, native)
    keys = ["block-a", "block-b", "block-c"]
    source_order, target_order = [4, 1, 3], [0, 2, 4]
    source_rows = [database.prepare_value(0, tokens, [], block_id=block) for block in source_order]
    target_rows = [target.prepare_value(0, tokens, [], block_id=block) for block in target_order]
    expected = destination.copy()
    for source_block, target_block in zip(source_order, target_order, strict=True):
        for offset, stride, payload in zip(offsets, strides, payloads, strict=True):
            size = payload * tokens // 16
            src, dst = offset + source_block * stride, offset + target_block * stride
            expected[dst : dst + size] = source[src : src + size]

    assert backend.put(keys, [row[0] for row in source_rows], [row[1] for row in source_rows]) == [True] * 3
    for addresses, sizes, _ in source_rows + target_rows:
        for address, size in zip(addresses, sizes, strict=True):
            ctypes.memset(address, 165, size)
    assert not np.array_equal(destination, expected)
    assert backend.get(keys, [row[0] for row in target_rows], [row[1] for row in target_rows]) == [0] * 3
    assert native.get_calls == 1
    np.testing.assert_array_equal(destination, expected)


@pytest.fixture
def worker_factory():
    case = unittest.TestCase()

    def create(backend_name, group_count=2):
        groups = [
            KVCacheGroupSpec(
                [f"model.layers.{index}.attn"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype="uint8",
                ),
            )
            for index in range(group_count)
        ]
        with patch.object(KVPoolWorker, "_build_cache_coordinator", return_value=None):
            worker = make_worker(
                case,
                kv_role="kv_both",
                extra_config={"backend": backend_name, "load_async": False},
                kv_cache_config=SimpleNamespace(num_blocks=8, kv_cache_groups=groups),
            )
        worker.token_database.set_group_buffers({0: [1000], 1: [2000]}, {0: [16], 1: [16]})
        return worker

    yield create
    case.doCleanups()


@pytest.mark.parametrize(
    "backend_name,layerwise",
    [("memcache", False), ("mooncake", False), ("mooncake", True)],
)
@pytest.mark.parametrize(
    "prompt_tokens,boundary",
    [(9216, 7680), (9217, 9216), (10087, 9216), (16384, 15360)],
)
def test_private_state_lookup_queries_the_restorable_boundary(backend_name, prompt_tokens, boundary, layerwise):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
    config = scheduler_tests.TestKVPoolScheduler()._make_config(block_size=1536, extra_config={"backend": backend_name})
    with patch(module + ".importlib") as imports, patch(module + ".LookupKeyClient") as client:
        imports.import_module.return_value = MagicMock()
        scheduler_cls = LayerwiseKVPoolScheduler if layerwise else scheduler_tests.KVPoolScheduler
        scheduler = scheduler_cls(config, use_layerwise=layerwise)
        scheduler.cacheable_group_ids = [0]
        scheduler.original_block_size = [1536, 4]
        scheduler.kv_cache_group_ids = [0]
        scheduler.use_eagle = False
        client.return_value.closed = False
        client.return_value.lookup.side_effect = lambda token_len, *args, **kwargs: token_len
        request = SimpleNamespace(
            request_id="private-tail",
            prompt_token_ids=[0] * prompt_tokens,
            num_tokens=prompt_tokens,
            block_hashes=[b"h"] * (prompt_tokens // 1536),
        )
        with patch.object(
            scheduler,
            "_get_block_key_layerwise_hit_tokens",
            side_effect=lambda req, token_len, computed: token_len,
            create=True,
        ) as layer_lookup:
            assert scheduler.get_num_new_matched_tokens(request, 0) == (boundary, False)
        if layerwise:
            assert layer_lookup.call_args.args[1] == boundary
        else:
            assert client.return_value.lookup.call_args.args[0] == boundary
        assert scheduler.load_specs[request.request_id].kvpool_cached_tokens == boundary


def test_mooncake_dense_read_failure_keeps_existing_block_fallback(worker_factory):
    worker = worker_factory("mooncake", group_count=1)
    worker.m_store.get.return_value = [-800]
    request = ReqMeta(
        req_id="dense",
        token_len_chunk=16,
        block_ids=[7],
        block_hashes=["h0"],
        load_spec=LoadSpec(0, 16, can_load=True, token_len=16),
        can_save=True,
    )
    metadata = AscendConnectorMetadata(set())
    metadata.add_request(request)
    worker.start_load_kv(metadata)
    assert worker.get_block_ids_with_load_errors() == {7}
    assert worker.get_failed_requests() == set()


def test_mooncake_preserves_partial_get_errors():
    native = MagicMock()
    native.batch_get_into_multi_buffers.return_value = [514, -800]
    backend = make_backend("mooncake", native)
    assert backend.get(["ok", "missing"], [[100], [200]], [[514], [514]]) == [0, -800]


@pytest.mark.parametrize("backend_name", ["memcache", "mooncake"])
def test_packed_qsa_gdn_pages_restore_disjoint_live_block_ids(backend_name):
    """Restore actual shared-page views without touching another live owner."""
    import torch
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec, MLAAttentionSpec

    from vllm_ascend.models.qwen4_exp.cache_config import build_qwen4_kv_cache_layout, make_slab_view

    main = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, head_size_v=8, dtype=torch.float16)
    raw = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=4, head_size_v=0, dtype=torch.float16)
    compressed = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=4, dtype=torch.float16, tokens_per_state=4)
    gdn = MambaSpec(block_size=128, shapes=((2, 4), (4, 4)), dtypes=(torch.float16, torch.float32))
    ple = MambaSpec(block_size=128, shapes=((2, 4),), dtypes=(torch.float16,))
    groups = [
        KVCacheGroupSpec(["model.layers.0.self_attn.attn"], main),
        KVCacheGroupSpec(["model.layers.0.self_attn.indexer.raw_key_cache"], raw),
        KVCacheGroupSpec(["model.layers.0.self_attn.indexer.compressed_key_cache"], compressed),
        KVCacheGroupSpec(["model.layers.1.linear_attn"], gdn),
        KVCacheGroupSpec(["model.ple"], ple),
    ]
    blocks = 5
    layout = build_qwen4_kv_cache_layout(groups, blocks)
    private_offset = layout.slot_backing_size + 64
    size = private_offset + blocks * 16 + 64
    source, destination = (np.full(size, 90, dtype=np.uint8) for _ in range(2))

    def views(array):
        backing = torch.from_numpy(array)

        def view(dtype, shape, offset, stride=None):
            return make_slab_view(
                backing,
                dtype=dtype,
                num_blocks=blocks,
                item_shape=shape,
                storage_offset=offset,
                block_stride=stride,
            )

        shared = layout.region("r0")
        return {
            0: [
                view(torch.float16, (128, 1, 8), 0, shared.block_stride),
                view(torch.float16, (128, 1, 8), 2048, shared.block_stride),
                view(torch.float16, (32, 1, 4), layout.region("r5").offset),
            ],
            1: [view(torch.float16, (8, 1, 4), layout.region("r4").offset)],
            2: [
                view(torch.float16, (2, 4), layout.region("r1").offset, shared.block_stride),
                view(torch.float32, (4, 4), layout.region("r2").offset, shared.block_stride),
            ],
            3: [view(torch.float16, (2, 4), private_offset)],
        }

    source_views, target_views = views(source), views(destination)
    # QSA and GDN deliberately alias physical pages. Their live block IDs
    # must be disjoint, as enforced by the shared scheduler block pool.
    mappings = {0: ([1, 4], [0, 2]), 1: ([1, 4], [0, 2]), 2: ([0, 3], [1, 4]), 3: ([0, 3], [1, 4])}
    for group_id, (source_ids, _) in mappings.items():
        for tensor_id, tensor in enumerate(source_views[group_id]):
            for block_id in source_ids:
                tensor[block_id].fill_(group_id * 11 + tensor_id + block_id + 1)

    def database(cache_views):
        result = ChunkedTokenDatabase([KeyMetadata("next", 0, 0, 0)], [128, 8, 128, 128], None)
        result.set_group_buffers(
            {group: [tensor.data_ptr() for tensor in tensors] for group, tensors in cache_views.items()},
            {
                group: [tensor[0].numel() * tensor.element_size() for tensor in tensors]
                for group, tensors in cache_views.items()
            },
            {
                group: [tensor.stride(0) * tensor.element_size() for tensor in tensors]
                for group, tensors in cache_views.items()
            },
        )
        return result

    source_db, target_db = database(source_views), database(target_views)
    keys, source_rows, target_rows = [], [], []
    expected = destination.copy()
    for group_id, (source_ids, target_ids) in mappings.items():
        for index, (source_id, target_id) in enumerate(zip(source_ids, target_ids, strict=True)):
            keys.append(f"{group_id}-{index}")
            tokens = 8 if group_id == 1 else 128
            src = source_db.prepare_value(group_id, tokens, [], block_id=source_id)
            dst = target_db.prepare_value(group_id, tokens, [], block_id=target_id)
            source_rows.append(src)
            target_rows.append(dst)
            for src_addr, dst_addr, length in zip(src[0], dst[0], src[1], strict=True):
                offset = dst_addr - destination.ctypes.data
                expected[offset : offset + length] = np.frombuffer(ctypes.string_at(src_addr, length), dtype=np.uint8)

    store = BufferStore([source, destination])
    backend = make_backend(backend_name, store)
    assert backend.put(keys, [row[0] for row in source_rows], [row[1] for row in source_rows]) == [True] * len(keys)
    source.fill(165)
    for row in target_rows:
        for address, length in zip(row[0], row[1], strict=True):
            ctypes.memset(address, 136, length)
    assert backend.get(keys, [row[0] for row in target_rows], [row[1] for row in target_rows]) == [0] * len(keys)
    np.testing.assert_array_equal(destination, expected)
