# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

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


@dataclass
class CacheIdentity:
    block_size: int = 32
    cache_dtype: str = "bfloat16"

    def compute_hash(self):
        return f"{self.block_size}:{self.cache_dtype}"


def config_for(prefix="", backend="mooncake"):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"cache_prefix": prefix, "backend": backend}),
        model_config=SimpleNamespace(compute_hash=lambda: "model-v1", hf_config=SimpleNamespace(_commit_hash="hf-v1")),
        cache_config=CacheIdentity(),
    )


class TestCachePrefixResolution(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(VLLM_IMAGE_TAG="image-v1", VLLM_BUILD_COMMIT="commit-v1")
        env_patch = patch.object(metadata, "envs", self.env)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        # Keep these tests independent of a vLLM installation; also inspect
        # the exact factors passed to the maintained upstream hash helper.
        hash_patch = patch.object(
            metadata,
            "hash_factors",
            side_effect=lambda factors: hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest(),
        )
        self.hash_factors = hash_patch.start()
        self.addCleanup(hash_patch.stop)

    def test_default_and_literal_values_do_not_require_build_identity(self):
        self.env.VLLM_IMAGE_TAG = self.env.VLLM_BUILD_COMMIT = ""
        for prefix in ("", "release-v2", "AUTO"):
            with self.subTest(prefix=prefix):
                self.assertEqual(metadata.resolve_cache_prefix(config_for(prefix)), prefix)
        self.hash_factors.assert_not_called()

    def test_non_mooncake_rejects_nonempty_prefix_before_accessing_model(self):
        for backend in ("memcache", "yuanrong"):
            for prefix in ("release-v2", "auto"):
                config = config_for(prefix, backend)
                del config.model_config
                with self.subTest(backend=backend, prefix=prefix), self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                    metadata.resolve_cache_prefix(config)
            self.assertEqual(metadata.resolve_cache_prefix(config_for("", backend)), "")
        self.hash_factors.assert_not_called()

    def test_upstream_string_coercion_and_omitted_default(self):
        for value, expected in ((None, "None"), (False, "False"), (0, "0"), ([], "[]"), ({}, "{}")):
            with self.subTest(value=value):
                self.assertEqual(metadata.resolve_cache_prefix(config_for(value)), expected)
                for backend in ("memcache", "yuanrong"):
                    with self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                        metadata.resolve_cache_prefix(config_for(value, backend))
        config = config_for()
        del config.kv_transfer_config.kv_connector_extra_config["cache_prefix"]
        self.assertEqual(metadata.resolve_cache_prefix(config), "")
        self.hash_factors.assert_not_called()

    def test_auto_factors_and_image_precedence(self):
        config = config_for("auto")
        original = metadata.resolve_cache_prefix(config)
        self.assertTrue(original.startswith("auto-"))
        self.hash_factors.assert_called_with(
            {"build": "image-v1", "model": "model-v1", "hf_commit": "hf-v1", "cache": "32:bfloat16"}
        )
        self.env.VLLM_BUILD_COMMIT = "commit-v2"
        self.assertEqual(original, metadata.resolve_cache_prefix(config))
        self.env.VLLM_IMAGE_TAG = "image-v2"
        self.assertNotEqual(original, metadata.resolve_cache_prefix(config))

    def test_compile_factors_identity_matches_current_upstream_hashing(self):
        config = config_for("auto")
        config.model_config.compile_factors = lambda: {"model": "model-v1"}
        config.model_config.compute_hash = MagicMock(side_effect=AssertionError("legacy hash must not be used"))
        config.cache_config = SimpleNamespace(
            block_size=32,
            compile_factors=lambda: {"block_size": 32, "cache_dtype": "bfloat16"},
            compute_hash=MagicMock(side_effect=AssertionError("legacy hash must not be used")),
        )
        expected_model = hashlib.sha256(json.dumps({"model": "model-v1"}, sort_keys=True).encode()).hexdigest()
        expected_cache = hashlib.sha256(
            json.dumps({"block_size": 32, "cache_dtype": "bfloat16"}, sort_keys=True).encode()
        ).hexdigest()
        self.assertTrue(metadata.resolve_cache_prefix(config).startswith("auto-"))
        self.hash_factors.assert_called_with(
            {"build": "image-v1", "model": expected_model, "hf_commit": "hf-v1", "cache": expected_cache}
        )
        config.model_config.compute_hash.assert_not_called()
        config.cache_config.compute_hash.assert_not_called()

    def test_commit_fallback_and_missing_build(self):
        config = config_for("auto")
        for image in ("", "local/vllm-openai:dev"):
            self.env.VLLM_IMAGE_TAG = image
            self.assertTrue(metadata.resolve_cache_prefix(config).startswith("auto-"))
            self.assertEqual(self.hash_factors.call_args.args[0]["build"], "commit-v1")
            for commit in ("", "unknown"):
                self.env.VLLM_BUILD_COMMIT = commit
                with self.assertRaisesRegex(ValueError, "VLLM_IMAGE_TAG or VLLM_BUILD_COMMIT"):
                    metadata.resolve_cache_prefix(config)
            self.env.VLLM_BUILD_COMMIT = "commit-v1"

    def test_model_hf_and_cache_identity_changes_isolate(self):
        config = config_for("auto")
        original = metadata.resolve_cache_prefix(config)
        for field in ("model", "hf", "cache"):
            other = deepcopy(config)
            if field == "model":
                other.model_config.compute_hash = lambda: "model-v2"
            elif field == "hf":
                other.model_config.hf_config._commit_hash = "hf-v2"
            else:
                other.cache_config.cache_dtype = "int8"
            with self.subTest(field=field):
                self.assertNotEqual(original, metadata.resolve_cache_prefix(other))
        config.scheduler_config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=4096)
        self.assertEqual(original, metadata.resolve_cache_prefix(config))

    def test_resolved_groups_normalize_scheduler_worker_without_mutation(self):
        spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype="bfloat16")
        uniform = UniformTypeKVCacheSpecs.from_specs({"layer.1": spec.copy_with_new_block_size(32)})
        groups = SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec), KVCacheGroupSpec(["layer.1"], uniform)]
        )
        worker_config = config_for("auto")
        scheduler_config = deepcopy(worker_config)
        scheduler_config.cache_config.block_size = 16
        self.assertEqual(
            metadata.resolve_cache_prefix(worker_config, groups),
            metadata.resolve_cache_prefix(scheduler_config, groups),
        )
        self.assertEqual(worker_config.cache_config.block_size, 32)
        self.assertEqual(scheduler_config.cache_config.block_size, 16)


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

    def test_auto_prefix_is_used_by_bulk_and_layerwise_workers(self):
        with (
            patch.object(metadata, "envs", SimpleNamespace(VLLM_IMAGE_TAG="image-v1", VLLM_BUILD_COMMIT="")),
            patch.object(metadata, "hash_factors", return_value="fingerprint"),
        ):
            for layerwise in (False, True):
                worker = make_worker(self, use_layerwise=layerwise, extra_config={"cache_prefix": "auto"})
                self.assertEqual(worker.cache_prefix, "auto-fingerprint")
                self.assertEqual(worker.metadata[0].cache_prefix, worker.cache_prefix)
                if layerwise:
                    self.assertTrue(worker._make_layerwise_full_key(0, "hash").startswith("auto-fingerprint@"))

    def test_reset_is_unfiltered_for_every_prefix_mode(self):
        for prefix in ("", "release-v1", "auto-0123"):
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
