# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerPoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


def config_for(prefix="", backend="mooncake"):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"cache_prefix": prefix, "backend": backend}),
    )


class TestFixedCachePrefix(unittest.TestCase):
    def test_default_and_literal_values(self):
        self.assertEqual(metadata.get_cache_prefix({}), "")
        for prefix in ("", "release-v2", "AUTO", " auto ", "namespace@with:markers"):
            with self.subTest(prefix=prefix):
                self.assertEqual(metadata.get_cache_prefix({"cache_prefix": prefix}), prefix)

    def test_upstream_literal_string_conversion(self):
        for value, expected in ((None, "None"), (False, "False"), (0, "0"), ([], "[]"), ({}, "{}")):
            with self.subTest(value=value):
                self.assertEqual(metadata.get_cache_prefix({"cache_prefix": value}), expected)

    def test_non_mooncake_rejects_nonempty_prefix(self):
        for backend in ("memcache", "yuanrong"):
            for prefix in ("release-v2", "auto", None, False, 0):
                with self.subTest(backend=backend, prefix=prefix), self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                    metadata.get_cache_prefix({"cache_prefix": prefix, "backend": backend})
            for extra in ({"backend": backend}, {"backend": backend, "cache_prefix": ""}):
                self.assertEqual(metadata.get_cache_prefix(extra), "")

    def test_auto_is_rejected_instead_of_becoming_a_literal_namespace(self):
        with self.assertRaisesRegex(ValueError, "auto.*not supported.*fixed prefix"):
            metadata.get_cache_prefix({"cache_prefix": "auto"})


class TestCachePrefixKeys(unittest.TestCase):
    def test_default_keys_and_namespaced_object_fast_and_layer_paths(self):
        old = "model@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@abcd"
        for prefix in ("", "release-v1", "release-v2"):
            with self.subTest(prefix=prefix):
                metadata = KeyMetadata("model", 0, 0, 0, cache_prefix=prefix)
                database = ChunkedTokenDatabase([metadata], [16], None)
                expected = f"{prefix}@{old}" if prefix else old
                key = database._make_key_by_hash("abcd")
                self.assertEqual(key.to_string(), expected)
                self.assertEqual(database._get_key_prefix(0) + "abcd", expected)
                self.assertEqual(list(database.process_token_key_strings(16, ["abcd"]))[0][2], expected)
                self.assertEqual(
                    list(database.process_token_key_strings_with_block_ids(16, ["abcd"], [1]))[0][2], expected
                )
                layer = key.split_layers(2)[1]
                unprefixed = LayerPoolKey(replace(metadata, cache_prefix=""), "abcd", 1).to_string()
                self.assertEqual(layer.to_string(), f"{prefix}@{unprefixed}" if prefix else unprefixed)
                self.assertNotEqual(key, PoolKey(replace(metadata, cache_prefix="other"), "abcd"))
                self.assertEqual(hash(key), hash(PoolKey(metadata, "abcd")))

    def test_pp_and_tp_rewriting_does_not_modify_prefix(self):
        prefix = "release@pp_rank:0@head_or_tp_rank:0@dcp:0"
        worker = make_worker(self, extra_config={"cache_prefix": prefix})
        key = worker.token_database._make_key_by_hash("hash").to_string()
        self.assertEqual(
            worker._make_sub_key_str(key, 3),
            prefix + "@" + key[len(prefix) + 1 :].replace("head_or_tp_rank:0", "head_or_tp_rank:3"),
        )
        database = worker.token_database
        database.partitions = [1, 1]
        database.group_num_layers = {"kv": {0: 2}}
        rewritten, _, _ = database.decode_adaptor_prefill_pp([key], [[10, 20, 30, 40]], [[1, 1, 1, 1]])
        self.assertTrue(rewritten[1].startswith(prefix + "@"))
        self.assertIn("@pp_rank:1@group:", rewritten[1][len(prefix) + 1 :])

    def test_worker_binding_and_layerwise_lookup_namespace(self):
        for layerwise in (False, True):
            workers = [
                make_worker(self, use_layerwise=layerwise, extra_config={"cache_prefix": prefix})
                for prefix in ("", "v1", "v2")
            ]
            keys = [
                worker._make_layerwise_full_key(0, "hash")
                if layerwise
                else worker.token_database._make_key_by_hash("hash").to_string()
                for worker in workers
            ]
            self.assertEqual(keys[1], "v1@" + keys[0])
            self.assertEqual(keys[2], "v2@" + keys[0])
            if layerwise:
                self.assertEqual(workers[1].layerwise_keys.make_hit_check_keys(0, "hash", 1), [keys[1]])

    def test_memcache_default_keys_and_unsupported_backend_startup(self):
        config = config_for("", "memcache")
        config.parallel_config = SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1)
        builder = MemcacheBackend.bind_layerwise_keys(
            vllm_config=config, kv_cache_config=None, model_name="model", use_hybrid=False, grouped_block_size=[16]
        )
        self.assertEqual(builder.make_full_key(0, "hash", 0, 0), "model@hash@0")
        self.assertEqual(builder.make_partial_key("model", "r1", 0, 1, 16, 0), "model@partial@r1@0@1@16@0")
        for backend in ("memcache", "yuanrong"):
            for prefix in ("v1", "auto"):
                with self.subTest(backend=backend, prefix=prefix), self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                    make_worker(self, extra_config={"backend": backend, "cache_prefix": prefix})
                with self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                    KVPoolScheduler(config_for(prefix, backend), False)

    def test_context_parallel_layerwise_keys_preserve_release_namespace(self):
        for pcp_size, dcp_size in ((2, 1), (1, 2)):
            with self.subTest(pcp_size=pcp_size, dcp_size=dcp_size):
                worker = make_worker(
                    self,
                    use_layerwise=True,
                    use_mla=True,
                    tp_size=2,
                    pcp_size=pcp_size,
                    dcp_size=dcp_size,
                    extra_config={"cache_prefix": "release-v1"},
                )
                key = worker._make_layerwise_full_key(0, "hash")
                candidates = worker.layerwise_keys.make_hit_check_keys(0, "hash", 1)
                self.assertIn(key, candidates)
                self.assertTrue(all(item.startswith("release-v1@") for item in candidates))
                if dcp_size > 1:
                    self.assertIn("@dcp_size:2@dcp_rank:0", key)

    def test_auto_is_rejected_by_scheduler_and_bulk_and_layerwise_workers(self):
        for layerwise in (False, True):
            with self.subTest(layerwise=layerwise), self.assertRaisesRegex(ValueError, "auto.*not supported"):
                make_worker(self, use_layerwise=layerwise, extra_config={"cache_prefix": "auto"})
        with self.assertRaisesRegex(ValueError, "auto.*not supported"):
            KVPoolScheduler(config_for("auto"), False)

    def test_reset_is_unfiltered_for_fixed_namespaces(self):
        for prefix in ("", "release-v1", "release-v2"):
            backend = MooncakeBackend.__new__(MooncakeBackend)
            backend._store_initialized = True
            backend.store = MagicMock()
            worker = make_worker(self, extra_config={"cache_prefix": prefix})
            worker.m_store = backend
            worker.kv_send_thread = MagicMock()
            worker.reset_store()
            worker.kv_send_thread.request_queue.join.assert_called_once_with()
            backend.store.remove_all.assert_called_once_with(force=True)


class ByteStore:
    """CPU stand-in for the Mooncake SDK's scatter/gather storage boundary."""

    requires_exists_before_put = True

    def __init__(self):
        self.objects = {}
        self.calls: dict[str, list[str]] = {"exists": [], "put": [], "get": []}

    def set_device(self):
        pass

    def exists(self, keys):
        self.calls["exists"].extend(keys)
        return [int(key in self.objects) for key in keys]

    def put(self, keys, addrs, sizes):
        self.calls["put"].extend(keys)
        for key, row, lengths in zip(keys, addrs, sizes, strict=True):
            self.objects[key] = [ctypes.string_at(addr, size) for addr, size in zip(row, lengths, strict=True)]
        return [True] * len(keys)

    def get(self, keys, addrs, sizes):
        self.calls["get"].extend(keys)
        for key, row, lengths in zip(keys, addrs, sizes, strict=True):
            for addr, size, data in zip(row, lengths, self.objects[key], strict=True):
                assert len(data) == size
                ctypes.memmove(addr, data, size)
        return [0] * len(keys)

    def remove_all(self, *, force):
        assert force is True
        self.objects.clear()

    # Mirror the SDK boundary so the actual MooncakeBackend also participates.
    batch_is_exist = exists
    batch_get_into_multi_buffers = get

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        return [0 if value else -1 for value in self.put(keys, addrs, sizes)]


class TestCachePrefixBulkRoundtrip(unittest.TestCase):
    def test_lookup_put_get_and_global_reset(self):
        # Run real worker lookup and send/receive handlers against CPU bytes.
        # Separate arrays and remapped destination blocks prevent local reuse
        # from hiding a wrong namespace or an incorrect scatter/gather offset.
        store = ByteStore()
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend._store_initialized, backend.store = True, store
        backend._lazy_init = False
        backend._prefetch_conf = SimpleNamespace(enabled=False)
        backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=True)
        hashes = [b"h0", b"h1"]
        for prefix in ("", "release-v1", "release-v2"):
            with self.subTest(prefix=prefix):
                worker = make_worker(self, extra_config={"cache_prefix": prefix})
                source = np.arange(48, dtype=np.uint8).reshape(3, 16)
                target = np.full_like(source, 255)
                writer_db = ChunkedTokenDatabase(worker.metadata, [16], None)
                reader_db = ChunkedTokenDatabase(worker.metadata, [16], None)
                for db, array in ((writer_db, source), (reader_db, target)):
                    db.set_group_buffers({0: [array.ctypes.data]}, {0: [16]}, {0: [16]}, group_num_layers={0: 1})
                worker.token_database = reader_db
                worker.m_store = backend
                self.assertEqual(worker.lookup(32, hashes), 0)
                sender = KVCacheStoreSendingThread(backend, writer_db, 16, 0, group_uses_align_state=[False])
                request = ReqMeta(req_id="write", token_len_chunk=32, block_ids=[1, 2], block_hashes=hashes)
                sender.add_stored_request(request)
                sender.request_queue.put(request)
                sender._handle_request(request)
                self.assertEqual(sender.request_queue.unfinished_tasks, 0)
                self.assertFalse(sender.save_disabled.is_set())
                self.assertEqual(worker.lookup(32, hashes), 32)
                receiver = KVCacheStoreRecvingThread(backend, reader_db, 16, 0)
                load = ReqMeta(
                    req_id="read",
                    token_len_chunk=32,
                    block_ids=[2, 1],
                    block_hashes=hashes,
                    load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32),
                )
                receiver.request_queue.put(load)
                receiver._handle_request(load)
                self.assertFalse(receiver._invalid_block_ids)
                np.testing.assert_array_equal(target[[2, 1]], source[1:])
                self.assertTrue(np.all(target[0] == 255))
                expected = list(store.calls["put"][-2:])
                self.assertEqual(expected, [reader_db._make_key_by_hash(value.hex()).to_string() for value in hashes])
                self.assertEqual(store.calls["get"][-2:], expected)
                self.assertTrue(all(key in store.calls["exists"] for key in expected))
                self.assertTrue(all(key.startswith(prefix + "@") for key in expected) if prefix else True)
        self.assertEqual(len(store.objects), 6)
        worker.m_store = backend
        worker.kv_send_thread = MagicMock()
        worker.reset_store()
        self.assertEqual(store.objects, {})

    def test_other_backend_empty_and_omitted_keep_bulk_keys(self):
        for backend in ("memcache", "yuanrong"):
            for extra in ({"backend": backend}, {"backend": backend, "cache_prefix": ""}):
                worker = make_worker(self, extra_config=extra)
                self.assertEqual(worker.cache_prefix, "")
                self.assertEqual(
                    worker.token_database._get_key_prefix(0),
                    "llama-7b@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@",
                )
