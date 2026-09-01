#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
    HybridCacheC128Config,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
    get_block_hashes,
    resolve_hybrid_cache_c128_config,
)

_GROUPED_BLOCK_HASH_DOMAIN = b"vllm-ascend-grouped-block-hash-v1\0"
_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES = 4


def _expected_grouped_hash(*block_hashes):
    hasher = hashlib.sha256()
    hasher.update(_GROUPED_BLOCK_HASH_DOMAIN)
    hasher.update(len(block_hashes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
    for block_hash in block_hashes:
        hash_bytes = block_hash.encode("utf-8") if isinstance(block_hash, str) else bytes(block_hash)
        hasher.update(len(hash_bytes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
        hasher.update(hash_bytes)
    return hasher.digest()


class TestKeyMetadata(unittest.TestCase):
    def test_fields(self):
        meta = KeyMetadata(
            model_name="llama",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        self.assertEqual(meta.model_name, "llama")
        self.assertEqual(meta.head_or_tp_rank, 0)
        self.assertEqual(meta.pcp_rank, 0)
        self.assertEqual(meta.dcp_rank, 0)
        self.assertEqual(meta.pp_rank, 0)


class TestPoolKey(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 1, 2, 3, 0)

    def test_hash_equal(self):
        k1 = PoolKey(self.meta, "abc123")
        k2 = PoolKey(self.meta, "abc123")
        self.assertEqual(hash(k1), hash(k2))

    def test_hash_diff(self):
        k1 = PoolKey(self.meta, "abc123")
        k2 = PoolKey(self.meta, "def456")
        self.assertNotEqual(hash(k1), hash(k2))

    def test_to_string(self):
        k = PoolKey(self.meta, "hash1")
        s = k.to_string()
        self.assertIn("llama", s)
        self.assertIn("@pcp2", s)
        self.assertIn("@dcp3", s)
        self.assertIn("@head_or_tp_rank:1", s)
        self.assertIn("@pp_rank:0", s)
        self.assertIn("hash1", s)

    def test_default_key_string_is_unchanged(self):
        k = PoolKey(self.meta, "hash1")
        self.assertEqual(
            k.to_string(),
            "llama@pcp2@dcp3@head_or_tp_rank:1@pp_rank:0@group:0"
            "@cache_role:kv@cache_family:default@hash1",
        )

    def test_transfer_namespace_and_range_are_part_of_key(self):
        transfer_meta = KeyMetadata(
            "llama",
            1,
            2,
            3,
            0,
            transfer_namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            slot_start=4,
            slot_end=8,
        )
        key = PoolKey(transfer_meta, "hash1")
        self.assertIn("@transfer:hybrid_c128_chunk_v1@range:4_8@hash1", key.to_string())
        self.assertNotEqual(hash(key), hash(PoolKey(self.meta, "hash1")))

    def test_split_layers(self):
        k = PoolKey(self.meta, "hash1")
        layers = k.split_layers(3)
        self.assertEqual(len(layers), 3)
        for i, lk in enumerate(layers):
            self.assertIsInstance(lk, LayerPoolKey)
            self.assertEqual(lk.layer_id, i)
            self.assertEqual(lk.chunk_hash, "hash1")


class TestLayerPoolKey(unittest.TestCase):
    def test_hash(self):
        meta = KeyMetadata("model", 0, 0, 0, 0)
        k1 = LayerPoolKey(meta, "h1", 0)
        k2 = LayerPoolKey(meta, "h1", 1)
        self.assertNotEqual(hash(k1), hash(k2))

    def test_to_string_contains_layer_id(self):
        meta = KeyMetadata("model", 0, 0, 0, 0)
        k = LayerPoolKey(meta, "h1", 5)
        s = k.to_string()
        self.assertIn("@layer_id:5", s)
        self.assertIn("model", s)
        self.assertTrue(s.endswith("@h1"))


class TestChunkedTokenDatabase(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 0, 0, 0, 0)
        self.db = ChunkedTokenDatabase([self.meta], block_size=[16], partitions=None)
        self.db.set_group_buffers({0: [1000, 2000]}, {0: [160, 320]}, group_num_layers={0: 1})

    def test_make_key_by_hash(self):
        key = self.db._make_key_by_hash("abc")
        self.assertIsInstance(key, PoolKey)
        self.assertEqual(key.chunk_hash, "abc")

    def test_process_tokens_empty(self):
        result = list(self.db.process_tokens(32, []))
        self.assertEqual(result, [])

    def test_process_tokens_with_str_hashes(self):
        hashes = ["aaa", "bbb"]
        result = list(self.db.process_tokens(32, hashes))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0)  # start
        self.assertEqual(result[0][1], 16)  # end
        self.assertEqual(result[1][0], 16)
        self.assertEqual(result[1][1], 32)

    def test_process_tokens_with_bytes_hashes(self):
        hashes = [b"\xaa\xbb", b"\xcc\xdd"]
        result = list(self.db.process_tokens(32, hashes))
        self.assertEqual(len(result), 2)

    def test_process_tokens_with_mask(self):
        hashes = ["a", "b", "c"]
        result = list(self.db.process_tokens(48, hashes, mask_num=16))
        # first chunk (start=0 < mask_num=16) should be skipped
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 16)

    def test_process_tokens_with_tail_clipped_block_ids_maps_tail_chunks(self):
        db = ChunkedTokenDatabase([self.meta], block_size=[128], partitions=None)
        hashes = [bytes([idx % 251]) * 32 for idx in range(128)]

        result = list(
            db.process_tokens_with_block_ids(
                128 * 128,
                hashes,
                [1000, 1001, 1002, 1003],
            )
        )

        self.assertEqual(
            [start for start, _, _, _ in result],
            [124 * 128, 125 * 128, 126 * 128, 127 * 128],
        )
        self.assertEqual(
            [block_id for _, _, _, block_id in result],
            [1000, 1001, 1002, 1003],
        )

    def test_disabled_transfer_chunks_preserve_compressed_block_mapping(self):
        db = ChunkedTokenDatabase(
            [self.meta],
            block_size=[8],
            partitions=None,
            hash_block_size=8,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [80]},
            group_cache_families={0: "c4"},
        )
        db.cache_coordinator = MagicMock()
        db.cache_coordinator.transfer_value_block_range.side_effect = AssertionError(
            "disabled transfer path must preserve legacy block-id mapping"
        )

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                64,
                [f"h{index}" for index in range(8)],
                [7, 11],
            )
        )

        self.assertEqual([chunk.raw_start for chunk in chunks], [0, 8])
        self.assertEqual([chunk.block_id for chunk in chunks], [7, 11])
        addrs, sizes, block_id = db.prepare_transfer_value(chunks[1], [7, 11])
        self.assertEqual((addrs, sizes, block_id), ([1000 + 11 * 80], [80], 11))

    def test_process_tokens_token_len_shorter_than_all_blocks(self):
        hashes = ["a", "b", "c", "d"]
        # token_len=32 means only first 2 blocks valid
        result = list(self.db.process_tokens(32, hashes))
        self.assertEqual(len(result), 2)

    def test_process_tokens_rehashes_grouped_hashes(self):
        db = ChunkedTokenDatabase([self.meta], block_size=[16], partitions=None, hash_block_size=8)
        result = list(db.process_tokens(32, ["a", "b", "c", "d"]))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][2].chunk_hash, _expected_grouped_hash("a", "b").hex())
        self.assertEqual(len(result[0][2].chunk_hash), 64)

    def test_get_block_hashes_rehashes_grouped_str_hashes(self):
        result = get_block_hashes(["a", "b", "c", "d"], group_block_size=32, hash_block_size=16)
        self.assertEqual(
            result,
            [
                _expected_grouped_hash("a", "b"),
                _expected_grouped_hash("c", "d"),
            ],
        )

    def test_get_block_hashes_rehashes_grouped_bytes_hashes(self):
        result = get_block_hashes([b"a", b"b", b"c", b"d"], group_block_size=32, hash_block_size=16)
        self.assertEqual(
            result,
            [
                _expected_grouped_hash(b"a", b"b"),
                _expected_grouped_hash(b"c", b"d"),
            ],
        )
        self.assertEqual(len(result[0]), 32)

    def test_prepare_value(self):
        addr, size, block_id = self.db.prepare_value(0, 16, [5, 6, 7])
        self.assertEqual(block_id, 5)
        self.assertEqual(len(addr), 2)
        self.assertEqual(addr[0], 1000 + 5 * 160)
        self.assertEqual(addr[1], 2000 + 5 * 320)
        self.assertEqual(size[0], 160)
        self.assertEqual(size[1], 320)

    def test_prepare_value_partial_block(self):
        addr, size, block_id = self.db.prepare_value(0, 8, [5])
        self.assertEqual(size[0], 80)  # 160/16*8
        self.assertEqual(size[1], 160)  # 320/16*8

    def test_prepare_value_uses_block_id_override(self):
        addr, size, block_id = self.db.prepare_value(64, 80, [5], block_id=99)
        self.assertEqual(block_id, 99)
        self.assertEqual(addr[0], 1000 + 99 * 160)
        self.assertEqual(addr[1], 2000 + 99 * 320)
        self.assertEqual(size[0], 160)
        self.assertEqual(size[1], 320)

    def test_prepare_value_layer(self):
        addr, size, block_id = self.db.prepare_value_layer(0, 16, [5, 6], layer_id=0)
        self.assertEqual(block_id, 5)
        self.assertEqual(len(addr), 2)
        # layer_id=0, entries_per_layers=2 => group_addrs[0] and group_addrs[1]
        self.assertEqual(addr[0], 1000 + 5 * 160)
        self.assertEqual(addr[1], 2000 + 5 * 320)

    def test_decode_adaptor_prefill_pp_no_partitions(self):
        key, addr, size = self.db.decode_adaptor_prefill_pp(["k1"], [[1, 2]], [[10, 20]])
        self.assertEqual(key, ["k1"])

    def test_decode_adaptor_prefill_pp_single_partition(self):
        db = ChunkedTokenDatabase([self.meta], [16], partitions=[4])
        key, addr, size = db.decode_adaptor_prefill_pp(["k1"], [[1, 2]], [[10, 20]])
        self.assertEqual(key, ["k1"])

    def test_decode_adaptor_prefill_pp_multi_partition(self):
        db = ChunkedTokenDatabase([self.meta], [16], partitions=[2, 2])
        db.set_group_buffers({0: [1000, 2000]}, {0: [160, 320]})
        keys = ["k1@pp_rank:0"]
        addrs = [[1, 2, 3, 4, 5, 6, 7, 8]]
        sizes = [[10, 20, 30, 40, 50, 60, 70, 80]]
        new_keys, new_addrs, new_sizes = db.decode_adaptor_prefill_pp(keys, addrs, sizes)
        self.assertEqual(len(new_keys), 2)
        self.assertIn("@pp_rank:0", new_keys[0])
        self.assertIn("@pp_rank:1", new_keys[1])

    def test_hybrid_c128_transfer_chunks_use_non_cumulative_c128_ranges(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[128, 128],
            partitions=None,
            use_hybrid=True,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [1280], 1: [1280]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = [f"h{index}" for index in range(132)]

        chunks = list(db.process_transfer_chunks(16896, hashes, kv_cache_group_id=1))

        self.assertEqual(len(chunks), 33)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (0, 4))
        self.assertEqual((chunks[1].value_start, chunks[1].value_end), (4, 8))
        self.assertEqual((chunks[31].value_start, chunks[31].value_end), (124, 128))
        self.assertEqual((chunks[32].value_start, chunks[32].value_end), (0, 4))
        self.assertEqual(chunks[31].target_block_index, 0)
        self.assertEqual(chunks[32].target_block_index, 1)
        self.assertIn("@range:0_4", chunks[0].key.to_string())
        self.assertIn("@range:4_8", chunks[1].key.to_string())

    def test_hybrid_c128_non_c128_chunk_prepares_one_complete_block(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[128, 128],
            partitions=None,
            use_hybrid=True,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [1280], 1: [1280]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = ["h0", "h1", "h2", "h3"]
        chunk = next(iter(db.process_transfer_chunks_with_block_ids(512, hashes, [7], kv_cache_group_id=0)))

        addrs, sizes, block_id = db.prepare_transfer_value(chunk, [7], kv_cache_group_id=0)

        self.assertEqual(block_id, 7)
        self.assertEqual(addrs, [1000 + 7 * 1280])
        self.assertEqual(sizes, [1280])
        self.assertNotIn("@range:", chunk.key.to_string())

    def test_hybrid_c128_block_size_64_uses_two_authoritative_slots_per_chunk(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=256,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=64,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[64, 64],
            partitions=None,
            use_hybrid=True,
            hash_block_size=64,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [640], 1: [640]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = [f"h{index}" for index in range(8)]

        chunks = list(db.process_transfer_chunks(512, hashes, kv_cache_group_id=1))

        self.assertEqual(len(chunks), 2)
        self.assertEqual([(chunk.value_start, chunk.value_end) for chunk in chunks], [(0, 2), (2, 4)])
        self.assertEqual([chunk.target_block_index for chunk in chunks], [0, 0])
        self.assertIn("@range:0_2", chunks[0].key.to_string())
        self.assertIn("@range:2_4", chunks[1].key.to_string())

    def test_hybrid_c128_block_ids_map_across_20k_prefix_pages(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(160)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                20 * 1024,
                hashes,
                [10, 11],
            )
        )

        self.assertEqual(len(chunks), 40)
        self.assertEqual([chunk.block_id for chunk in chunks[:32]], [10] * 32)
        self.assertEqual([chunk.block_id for chunk in chunks[32:]], [11] * 8)
        self.assertEqual((chunks[32].value_start, chunks[32].value_end), (0, 4))
        self.assertEqual(chunks[32].target_block_index, 1)

    def test_hybrid_c128_tail_block_ids_and_mask_num_map_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                32 * 1024,
                hashes,
                [21],
                mask_num=17 * 1024,
            )
        )

        self.assertEqual(len(chunks), 30)
        self.assertEqual(chunks[0].raw_start, 17 * 1024)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (8, 12))
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 21 for chunk in chunks))

    def test_hybrid_c128_skip_null_blocks_keeps_valid_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                32 * 1024,
                hashes,
                [0, 22],
                skip_null_blocks=True,
            )
        )

        self.assertEqual(len(chunks), 32)
        self.assertEqual(chunks[0].raw_start, 16 * 1024)
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 22 for chunk in chunks))


class TestHybridCacheC128Config(unittest.TestCase):
    @staticmethod
    def _config(enabled=True, *, backend="mooncake", load_async=False):
        extra = {
            "backend": backend,
            "load_async": load_async,
            "hybrid_cache_c128_chunk": enabled,
        }
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(),
                hf_config=SimpleNamespace(),
            ),
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        )

    def _resolve(self, config, **overrides):
        kwargs = {
            "use_layerwise": False,
            "group_block_sizes": [128, 128],
            "group_cache_families": ["c4", "c128"],
            "hash_block_size": 128,
            "discard_partial_chunks": True,
        }
        kwargs.update(overrides)
        return resolve_hybrid_cache_c128_config(config, **kwargs)

    def test_missing_option_preserves_default_path(self):
        config = self._config()
        del config.kv_transfer_config.kv_connector_extra_config["hybrid_cache_c128_chunk"]
        self.assertFalse(self._resolve(config).enabled)

    def test_false_option_preserves_default_path(self):
        self.assertFalse(self._resolve(self._config(False)).enabled)

    def test_block_size_128_uses_512_tokens(self):
        resolved = self._resolve(self._config())
        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.chunk_tokens, 512)
        self.assertEqual(resolved.c128_group_id, 1)
        self.assertEqual(resolved.c128_slots_per_page, 128)

    def test_block_size_64_uses_256_tokens(self):
        resolved = self._resolve(
            self._config(),
            group_block_sizes=[64, 64],
            group_cache_families=["c4", "c128"],
            hash_block_size=64,
        )
        self.assertEqual(resolved.chunk_tokens, 256)
        self.assertEqual(resolved.c128_slots_per_page, 64)

    def test_load_async_is_supported(self):
        resolved = self._resolve(self._config(load_async=True))
        self.assertTrue(resolved.enabled)

    def test_invalid_values_fail_fast(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._resolve(self._config(value))

    def test_unsupported_execution_modes_fail_fast(self):
        cases = (
            (self._config(backend="memcache"), {}),
            (self._config(), {"use_layerwise": True}),
            (self._config(), {"discard_partial_chunks": False}),
            (self._config(), {"group_block_sizes": [256, 128]}),
            (self._config(), {"group_block_sizes": [128], "group_cache_families": ["default"]}),
        )
        for config, overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._resolve(config, **overrides)


class TestLoadSpec(unittest.TestCase):
    def test_fields(self):
        spec = LoadSpec(vllm_cached_tokens=10, kvpool_cached_tokens=20, can_load=True)
        self.assertEqual(spec.vllm_cached_tokens, 10)
        self.assertEqual(spec.kvpool_cached_tokens, 20)
        self.assertTrue(spec.can_load)
        self.assertEqual(spec.token_len, 0)

    def test_token_len_default(self):
        spec = LoadSpec(0, 0, False, token_len=128)
        self.assertEqual(spec.token_len, 128)


class TestRequestTracker(unittest.TestCase):
    def test_from_new_request(self):
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.block_ids = [10, 20, 30]
        new_req.prompt_token_ids = list(range(100))

        tracker = RequestTracker.from_new_request(new_req, num_tokens_to_compute=48)
        self.assertEqual(tracker.req_id, "req-1")
        self.assertEqual(tracker.token_len, 48)
        self.assertEqual(tracker.allocated_block_ids, [10, 20, 30])
        self.assertEqual(len(tracker.token_ids), 48)
        self.assertEqual(tracker.num_saved_tokens, 0)

    def test_from_new_request_nested_block_ids(self):
        new_req = MagicMock()
        new_req.req_id = "req-2"
        new_req.block_ids = [[10, 20], [30, 40]]
        new_req.prompt_token_ids = list(range(32))

        tracker = RequestTracker.from_new_request(new_req, num_tokens_to_compute=32)
        self.assertEqual(tracker.allocated_block_ids, [10, 20])

    def test_update_with_list(self):
        tracker = RequestTracker(req_id="r1", token_len=16, allocated_block_ids=[1, 2])
        tracker.update([3, 4])
        self.assertEqual(tracker.allocated_block_ids, [1, 2, 3, 4])

    def test_update_with_tuple(self):
        tracker = RequestTracker(req_id="r1", token_len=16, allocated_block_ids=[1])
        tracker.update(([5, 6], [7, 8]))
        self.assertEqual(tracker.allocated_block_ids, [1, 5, 6])

    def test_update_with_empty(self):
        tracker = RequestTracker(req_id="r1", token_len=16, allocated_block_ids=[1])
        tracker.update([])
        self.assertEqual(tracker.allocated_block_ids, [1])

    def test_update_invalid_type(self):
        tracker = RequestTracker(req_id="r1", token_len=16, allocated_block_ids=[1])
        with self.assertRaises(ValueError):
            tracker.update("invalid")  # type: ignore[arg-type]

    def test_update_mamba_with_tuple(self):
        tracker = RequestTracker(
            req_id="r1", token_len=16, allocated_block_ids_by_group=[[1], [2], [3], [4]], block_sizes=[16] * 4
        )
        tracker.update(([5, 6], [0, 7], [0, 8], [0, 9]))
        self.assertEqual(tracker.allocated_block_ids_by_group[0], [1, 5, 6])
        self.assertEqual(tracker.allocated_block_ids_by_group[1], [2, 0, 7])
        self.assertEqual(tracker.allocated_block_ids_by_group[2], [3, 0, 8])
        self.assertEqual(tracker.allocated_block_ids_by_group[3], [4, 0, 9])

    def test_update_mamba_mtp_with_tuple_chunk2(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids_by_group=[
                [1, 2],
                [0, 3, 4, 5, 6],
                [0, 7, 8, 9, 10],
                [0, 11, 12, 13, 14],
            ],
            mamba_group_ids=[1, 2, 3],
            num_speculative_blocks=3,
            block_sizes=[16] * 4,
        )

        tracker.update(([15, 16], [4, 17], [8, 18], [12, 19]), 32)
        self.assertEqual(tracker.allocated_block_ids_by_group[0], [1, 2, 15, 16])
        self.assertEqual(tracker.allocated_block_ids_by_group[1], [0, 3, 0, 5, 6, 4, 17])
        self.assertEqual(tracker.allocated_block_ids_by_group[2], [0, 7, 0, 9, 10, 8, 18])
        self.assertEqual(tracker.allocated_block_ids_by_group[3], [0, 11, 0, 13, 14, 12, 19])

    def test_update_mamba_mtp_with_tuple_chunk8(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=128,
            allocated_block_ids_by_group=[
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12],
                [0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16],
                [0, 0, 0, 0, 0, 0, 0, 17, 18, 19, 20],
            ],
            mamba_group_ids=[1, 2, 3],
            num_speculative_blocks=3,
            block_sizes=[16] * 4,
        )

        tracker.update(
            (
                [21, 22, 23, 24, 25, 26, 27, 28],
                [0, 0, 0, 0, 10, 11, 12, 29],
                [0, 0, 0, 0, 14, 15, 16, 30],
                [0, 0, 0, 0, 18, 19, 20, 31],
            ),
            128,
        )
        self.assertEqual(
            tracker.allocated_block_ids_by_group[0], [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28]
        )
        self.assertEqual(
            tracker.allocated_block_ids_by_group[1], [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 29]
        )
        self.assertEqual(
            tracker.allocated_block_ids_by_group[2], [0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 14, 15, 16, 30]
        )
        self.assertEqual(
            tracker.allocated_block_ids_by_group[3], [0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 31]
        )


class TestReqMeta(unittest.TestCase):
    def test_from_request_tracker_basic_save(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
            token_ids=list(range(32)),
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, block_hashes=[b"h1", b"h2"])
        self.assertIsNotNone(meta)
        self.assertEqual(meta.req_id, "r1")
        self.assertTrue(meta.can_save)
        self.assertEqual(meta.token_len_chunk, 32)
        self.assertIsNone(meta.load_spec)

    def test_from_request_tracker_skip_save(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, skip_save=True)
        self.assertIsNone(meta)

    def test_from_request_tracker_with_load_spec(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, load_spec=load_spec, skip_save=True)
        self.assertIsNotNone(meta)
        self.assertIsNotNone(meta.load_spec)

    def test_from_request_tracker_load_spec_cannot_load(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=False)
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, load_spec=load_spec, skip_save=True)
        # can_load=False => load_spec set to None in meta,
        # but skip_save+load_spec input is not None, so meta is still created
        self.assertIsNotNone(meta)
        self.assertIsNone(meta.load_spec)
        self.assertFalse(meta.can_save)

    def test_from_request_tracker_partial_tokens_discarded(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=20,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, discard_partial_chunks=True)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 16)

    def test_from_request_tracker_no_discard(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=20,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, discard_partial_chunks=False)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 20)

    def test_from_request_tracker_already_saved(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16)

        # num_saved_tokens=32, chunk_boundary=ceil(33/16)*16=48 > 32
        # so skip_save, and no load_spec => None
        self.assertIsNone(meta)

    def test_from_request_tracker_with_original_block_size(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, original_block_size=8)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.original_block_size, 8)


class TestAscendConnectorMetadata(unittest.TestCase):
    def test_add_request(self):
        meta = AscendConnectorMetadata(unfinished_request_ids=set(), preempted_req_ids=set())
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[],
        )
        meta.add_request(req)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].req_id, "r1")


class TestLayerMultiBlockReqMeta(unittest.TestCase):
    def test_fields(self):
        meta = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[0, 16],
            ends=[16, 32],
            block_ids=[0, 1],
            layer_id=2,
        )
        self.assertEqual(meta.req_id, "r1")
        self.assertEqual(meta.layer_id, 2)
        self.assertTrue(meta.is_last_chunk)
        self.assertIsNone(meta.current_event)


if __name__ == "__main__":
    unittest.main()
