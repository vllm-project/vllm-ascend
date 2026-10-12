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

import unittest
from dataclasses import replace
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    HybridCacheC128Config,
    HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
    TransferChunkWithBlockId,
    aggregate_c128_page_chunks,
    get_block_hashes,
    get_group_block_size,
    get_group_cache_family,
    infer_cache_transfer_granularity,
    infer_dcp_mismatch_info,
    infer_group_block_sizes,
    masked_block_runs,
    resolve_hybrid_cache_c128_config,
    uses_hybrid_kv_cache,
)


class TestCacheLayoutHelpers(unittest.TestCase):
    def test_uses_hybrid_kv_cache(self):
        groups = [
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32)),
        ]
        scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
        self.assertTrue(uses_hybrid_kv_cache(scheduler_config, groups))
        self.assertFalse(uses_hybrid_kv_cache(scheduler_config, None))

    def test_uses_hybrid_kv_cache_disabled(self):
        groups = [
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32)),
        ]
        scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=True)
        self.assertFalse(uses_hybrid_kv_cache(scheduler_config, groups))

    def test_infer_group_block_sizes(self):
        groups = [
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32)),
        ]
        self.assertEqual(infer_group_block_sizes(8, groups), [16, 32])
        self.assertEqual(infer_group_block_sizes(8, None), [8])

    def test_get_group_cache_family(self):
        self.assertEqual(get_group_cache_family(["c1", "c2"], 1), "c2")
        self.assertEqual(get_group_cache_family(["c1"], 3), "default")

    def test_get_group_block_size(self):
        self.assertEqual(get_group_block_size([16, 32], 1), 32)

    def test_get_group_block_size_out_of_range(self):
        self.assertEqual(get_group_block_size([16, 32], 5), 16)

    def test_infer_cache_transfer_granularity(self):
        self.assertEqual(infer_cache_transfer_granularity([16, 32], 32, [0, 1]), 32)


class TestMaskedBlockRuns(unittest.TestCase):
    def test_none_mask_returns_single_run(self):
        self.assertEqual(masked_block_runs(None, 1, 5), [(1, 5)])

    def test_sparse_mask_splits_into_runs(self):
        mask = [False, True, False, True, True]
        self.assertEqual(masked_block_runs(mask, 0, 5), [(1, 2), (3, 5)])

    def test_empty_range_returns_no_runs(self):
        self.assertEqual(masked_block_runs([True, True], 2, 2), [])
        self.assertEqual(masked_block_runs(None, 3, 1), [])

    def test_blocks_beyond_mask_length_are_allowed(self):
        self.assertEqual(masked_block_runs([False, True], 0, 4), [(1, 4)])

    def test_all_disallowed_mask_returns_no_runs(self):
        self.assertEqual(masked_block_runs([False, False], 0, 2), [])


class TestKeyMetadata(unittest.TestCase):
    def test_fields(self):
        meta = KeyMetadata(
            model_name="llama",
            head_or_tp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        self.assertEqual(meta.model_name, "llama")
        self.assertEqual(meta.head_or_tp_rank, 0)
        self.assertEqual(meta.dcp_rank, 0)
        self.assertEqual(meta.pp_rank, 0)


class TestPoolKey(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 1, 3, 0)

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
        self.assertEqual(
            s,
            "llama@dcp:3@head_or_tp_rank:1@pp_rank:0@group:0@cache_role:kv@cache_family:default@hash1",
        )

    def test_cache_partitions_use_distinct_keys(self):
        key = PoolKey(self.meta, "hash1")
        for field, value in (
            ("model_name", "other-model"),
            ("head_or_tp_rank", 2),
            ("dcp_rank", 0),
            ("pp_rank", 1),
            ("kv_cache_group_id", 1),
            ("cache_role", "state"),
            ("cache_family", "swa"),
        ):
            with self.subTest(field=field):
                other = PoolKey(replace(self.meta, **{field: value}), "hash1")
                self.assertNotEqual(key.to_string(), other.to_string())
                self.assertNotEqual(key, other)

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
        meta = KeyMetadata("model", 0, 0, 0)
        k1 = LayerPoolKey(meta, "h1", 0)
        k2 = LayerPoolKey(meta, "h1", 1)
        self.assertNotEqual(hash(k1), hash(k2))

    def test_to_string_contains_layer_id(self):
        meta = KeyMetadata("model", 0, 0, 0)
        k = LayerPoolKey(meta, "h1", 5)
        s = k.to_string()
        self.assertIn("@dcp:0", s)
        self.assertIn("@layer_id:5", s)
        self.assertIn("model", s)
        self.assertTrue(s.endswith("@h1"))


class TestChunkedTokenDatabase(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 0, 0, 0)
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
            db.process_token_key_strings_with_block_ids(
                128 * 128,
                hashes,
                [1000, 1001, 1002, 1003],
            )
        )

        self.assertEqual(
            [start for start, _, _, _, _ in result],
            [124 * 128, 125 * 128, 126 * 128, 127 * 128],
        )
        self.assertEqual(
            [block_id for _, _, _, _, block_id in result],
            [1000, 1001, 1002, 1003],
        )

    def test_process_tokens_token_len_shorter_than_all_blocks(self):
        hashes = ["a", "b", "c", "d"]
        # token_len=32 means only first 2 blocks valid
        result = list(self.db.process_tokens(32, hashes))
        self.assertEqual(len(result), 2)

    def test_process_tokens_selects_terminal_group_hash(self):
        db = ChunkedTokenDatabase([self.meta], block_size=[16], partitions=None, hash_block_size=8)
        result = list(db.process_tokens(32, ["a", "b", "c", "d"]))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][2].chunk_hash, "b")

    def test_key_strings_match_pool_keys(self):
        hashes = ["aaa", "bbb", "ccc"]
        pool_keys = list(self.db.process_tokens(40, hashes))
        self.assertEqual(
            list(self.db.process_token_key_strings(40, hashes)),
            [
                (start, end, key.to_string(), hash_val)
                for (start, end, key), hash_val in zip(pool_keys, hashes, strict=True)
            ],
        )

        block_ids = [5, 6]
        self.assertEqual(
            list(self.db.process_token_key_strings_with_block_ids(32, hashes, block_ids)),
            [
                (start, end, key.to_string(), hash_val, block_id)
                for (start, end, key), hash_val, block_id in zip(pool_keys[:2], hashes[:2], block_ids, strict=True)
            ],
        )

    def test_direct_keys_preserve_multigroup_layerwise_key_semantics(self):
        group_metadata = [
            KeyMetadata("llama", 0, 0, 0),
            KeyMetadata("llama", 1, 0, 0),
        ]
        db = ChunkedTokenDatabase(group_metadata, block_size=[16, 64], partitions=None, hash_block_size=16)
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [160], 1: [320]},
            group_cache_families={0: "c1", 1: "c2"},
            group_num_layers={0: 2, 1: 2},
        )
        hashes = ["a", "b", "c", "d"]

        pool_key_result = list(db.process_tokens(64, hashes, kv_cache_group_id=1))
        direct_key_result = list(db.process_token_key_strings(64, hashes, kv_cache_group_id=1))

        self.assertEqual(len(pool_key_result), 1)
        self.assertEqual(
            direct_key_result[0][:3],
            (pool_key_result[0][0], pool_key_result[0][1], pool_key_result[0][2].to_string()),
        )
        layer_key = pool_key_result[0][2].split_layers(2)[1]
        self.assertIn("@group:1@cache_role:kv@cache_family:c2@layer_id:1", layer_key.to_string())

    def test_key_strings_pre_shard_after_filtering(self):
        hashes = ["a", "b", "c", "d"]
        store_mask = [True, False, True, True]
        result = list(
            self.db.process_token_key_strings_with_block_ids(
                64,
                hashes,
                [10, 11, 12, 13],
                chunk_filter=lambda start: store_mask[start // 16],
                shard_rank=1,
                shard_size=2,
            )
        )
        self.assertEqual([(start, end, block_id) for start, end, _, _, block_id in result], [(32, 48, 12)])

    def test_compressed_keys_use_logical_spans_and_values_use_physical_rows(self):
        db = ChunkedTokenDatabase(
            [self.meta],
            block_size=[512],
            partitions=None,
            hash_block_size=128,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1024]},
            group_cache_families={0: "c4"},
            group_num_layers={0: 1},
        )
        hashes = [bytes([idx]) * 32 for idx in range(8)]

        chunks = list(
            db.process_tokens(
                1024,
                hashes,
                kv_cache_group_id=0,
                cache_family="c4",
            )
        )
        self.assertEqual(
            [(start, end) for start, end, _ in chunks],
            [(0, 512), (512, 1024)],
        )

        first_addr, first_size, first_block = db.prepare_value(
            0,
            512,
            [5, 6],
            kv_cache_group_id=0,
        )
        second_addr, second_size, second_block = db.prepare_value(
            512,
            1024,
            [5, 6],
            kv_cache_group_id=0,
        )
        self.assertEqual((first_block, second_block), (5, 6))
        self.assertEqual(
            (first_addr, second_addr),
            ([1000 + 5 * 1024], [1000 + 6 * 1024]),
        )
        self.assertEqual((first_size, second_size), ([1024], [1024]))

    def test_get_block_hashes_selects_terminal_str_hashes(self):
        result = get_block_hashes(["a", "b", "c", "d"], group_block_size=32, hash_block_size=16)
        self.assertEqual(list(result), ["b", "d"])

    def test_get_block_hashes_selects_terminal_byte_hashes(self):
        result = get_block_hashes([b"a", b"b", b"c", b"d"], group_block_size=32, hash_block_size=16)
        self.assertEqual(list(result), [b"b", b"d"])

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
        addrs = [[1, 2, 3,4, 5, 6, 7, 8]]
        sizes = [[10, 20, 30, 40, 50, 60, 70, 80]]
        new_keys, new_addrs, new_sizes = db.decode_adaptor_prefill_pp(keys, addrs, sizes)
        self.assertEqual(len(new_keys), 2)
        self.assertIn("@pp_rank:0", new_keys[0])
        self.assertIn("@pp_rank:1", new_keys[1])


class TestPoolKeyTransferNamespace(unittest.TestCase):
    def test_default_key_string_is_unchanged(self):
        meta = KeyMetadata("llama", 0, 0, 0, kv_cache_group_id=2, cache_family="c4")
        key = PoolKey(meta, "hash")
        self.assertNotIn("@transfer:", key.to_string())
        self.assertNotIn("@range:", key.to_string())

    def test_transfer_namespace_and_range_are_part_of_key(self):
        plain = KeyMetadata("llama", 0, 0, 0, kv_cache_group_id=2, cache_family="c128")
        extended = KeyMetadata(
            "llama",
            0,
            0,
            0,
            kv_cache_group_id=2,
            cache_family="c128",
            transfer_namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            slot_start=0,
            slot_end=4,
        )
        key_plain = PoolKey(plain, "hash")
        key_extended = PoolKey(extended, "hash")
        self.assertNotEqual(hash(key_plain), hash(key_extended))
        self.assertIn(f"@transfer:{HYBRID_CACHE_C128_TRANSFER_NAMESPACE}", key_extended.to_string())
        self.assertIn("@range:0_4", key_extended.to_string())


class TestHybridC128TransferChunks(unittest.TestCase):
    """C128 hybrid transfer chunk geometry.

    Block sizes are logical raw-token spans: a c4 group with 128 physical
    slots maps 512 raw tokens per page and a c128 group with 128 physical
    slots maps 16384 raw tokens per page.
    """

    @staticmethod
    def _database(
        block_sizes: list[int],
        families: dict[int, str],
        config: HybridCacheC128Config,
        hash_block_size: int,
        block_lens: dict[int, list[int]],
    ) -> ChunkedTokenDatabase:
        metadata = [KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=gid) for gid in range(len(block_sizes))]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=block_sizes,
            partitions=None,
            hash_block_size=hash_block_size,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {gid: [1000 + gid * 1000] for gid in range(len(block_sizes))},
            block_lens,
            group_cache_families=families,
        )
        return db

    def test_transfer_chunks_use_non_cumulative_c128_ranges(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[512, 128 * 128],
            families={0: "c4", 1: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280], 1: [1280]},
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

    def test_non_c128_chunk_prepares_one_complete_block(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[512, 128 * 128],
            families={0: "c4", 1: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280], 1: [1280]},
        )
        hashes = ["h0", "h1", "h2", "h3"]
        chunk = next(iter(db.process_transfer_chunks_with_block_ids(512, hashes, [7], kv_cache_group_id=0)))

        addrs, sizes, block_id = db.prepare_transfer_value(chunk, [7], kv_cache_group_id=0)

        self.assertEqual(block_id, 7)
        self.assertEqual(addrs, [1000 + 7 * 1280])
        self.assertEqual(sizes, [1280])
        self.assertNotIn("@range:", chunk.key.to_string())

    def test_c128_page_load_transfers_complete_physical_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[512, 128 * 128],
            families={0: "c4", 1: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280], 1: [1280]},
        )
        hashes = [f"h{index}" for index in range(132)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                16896,
                hashes,
                [3, 4],
                kv_cache_group_id=1,
            )
        )
        self.assertEqual(len(chunks), 33)
        page_chunk = aggregate_c128_page_chunks(chunks, config.c128_slots_per_page)[-1]

        addrs, sizes, block_id = db.prepare_transfer_value(page_chunk, [3, 4], kv_cache_group_id=1)

        # The complete physical page of block 4 is restored in one transfer.
        self.assertEqual(block_id, 4)
        self.assertEqual(addrs, [2000 + 4 * 1280])
        self.assertEqual(sizes, [1280])

    def test_block_size_64_uses_two_authoritative_slots_per_chunk(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=256,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=64,
        )
        db = self._database(
            block_sizes=[256, 64 * 128],
            families={0: "c4", 1: "c128"},
            config=config,
            hash_block_size=64,
            block_lens={0: [640], 1: [640]},
        )
        hashes = [f"h{index}" for index in range(8)]

        chunks = list(db.process_transfer_chunks(512, hashes, kv_cache_group_id=1))

        self.assertEqual(len(chunks), 2)
        self.assertEqual([(chunk.value_start, chunk.value_end) for chunk in chunks], [(0, 2), (2, 4)])
        self.assertEqual([chunk.target_block_index for chunk in chunks], [0, 0])
        self.assertIn("@range:0_2", chunks[0].key.to_string())
        self.assertIn("@range:2_4", chunks[1].key.to_string())

    def test_block_ids_map_across_20k_prefix_pages(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[128 * 128],
            families={0: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280]},
        )
        hashes = [f"h{index}" for index in range(160)]

        chunks = list(db.process_transfer_chunks_with_block_ids(20 * 1024, hashes, [10, 11]))

        self.assertEqual(len(chunks), 40)
        self.assertEqual([chunk.block_id for chunk in chunks[:32]], [10] * 32)
        self.assertEqual([chunk.block_id for chunk in chunks[32:]], [11] * 8)
        self.assertEqual((chunks[32].value_start, chunks[32].value_end), (0, 4))
        self.assertEqual(chunks[32].target_block_index, 1)

    def test_tail_block_ids_and_mask_num_map_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[128 * 128],
            families={0: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280]},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(32 * 1024, hashes, [21], mask_num=17 * 1024)
        )

        self.assertEqual(len(chunks), 30)
        self.assertEqual(chunks[0].raw_start, 17 * 1024)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (8, 12))
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 21 for chunk in chunks))

    def test_skip_null_blocks_keeps_valid_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = self._database(
            block_sizes=[128 * 128],
            families={0: "c128"},
            config=config,
            hash_block_size=128,
            block_lens={0: [1280]},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(32 * 1024, hashes, [0, 22], skip_null_blocks=True)
        )

        self.assertEqual(len(chunks), 32)
        self.assertEqual(chunks[0].raw_start, 16 * 1024)
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 22 for chunk in chunks))

    def test_aggregate_c128_page_chunks_keeps_tail_and_full_pages(self):
        def chunk(block_index: int, block_id: int, value_start: int, value_end: int) -> TransferChunkWithBlockId:
            return TransferChunkWithBlockId(
                raw_start=0,
                raw_end=0,
                value_start=value_start,
                value_end=value_end,
                target_block_index=block_index,
                key=PoolKey(KeyMetadata("m", 0, 0, 0), f"h{block_index}-{value_start}"),
                block_id=block_id,
                block_ids=(block_id,),
            )

        chunks = [
            chunk(0, 10, 0, 4),
            chunk(0, 10, 4, 8),
            chunk(0, 10, 124, 128),
            chunk(1, 11, 0, 4),
            chunk(1, 11, 4, 8),
        ]

        pages = aggregate_c128_page_chunks(chunks, slots_per_page=128)

        # Page 0 is complete (value_end == slots_per_page) and page 1 is the
        # tail page, so one representative chunk per page survives.
        self.assertEqual(len(pages), 2)
        self.assertEqual({page.block_id for page in pages}, {10, 11})
        self.assertTrue(all(page.value_end == 128 or page.target_block_index == 1 for page in pages))

    def test_disabled_fallback_wraps_native_token_chunks(self):
        db = ChunkedTokenDatabase(
            [KeyMetadata("m", 0, 0, 0)],
            block_size=[16],
            partitions=None,
            hash_block_size=16,
        )
        db.set_group_buffers({0: [1000]}, {0: [160]})

        chunks = list(db.process_transfer_chunks_with_block_ids(32, ["a", "b"], [4, 5]))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].block_id, 4)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (0, 16))
        self.assertEqual(chunks[1].block_id, 5)
        self.assertEqual(chunks[1].raw_start, 16)

        addrs, sizes, block_id = db.prepare_transfer_value(chunks[0], [4, 5])
        self.assertEqual(block_id, 4)
        self.assertEqual(sizes, [160])


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
            "group_block_sizes": [512, 128 * 128],
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
            group_block_sizes=[256, 64 * 128],
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
            (self._config(), {"group_block_sizes": [256, 128 * 128]}),
            (self._config(), {"group_block_sizes": [512], "group_cache_families": ["default"]}),
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

    def test_update_mamba_uses_per_group_speculative_counts(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids_by_group=[[1, 2], [3, 4], [5, 6]],
            num_speculative_blocks_by_group={1: 1, 2: 0},
            block_sizes=[16] * 3,
        )

        tracker.update(([7], [4, 8], [9]), 32)

        self.assertEqual(tracker.allocated_block_ids_by_group[0], [1, 2, 7])
        self.assertEqual(tracker.allocated_block_ids_by_group[1], [0, 0, 4, 8])
        self.assertEqual(tracker.allocated_block_ids_by_group[2], [0, 6, 9])

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
            num_speculative_blocks_by_group={1: 3, 2: 3, 3: 3},
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
            num_speculative_blocks_by_group={1: 3, 2: 3, 3: 3},
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

    def test_from_request_tracker_can_load_suppresses_save(self):
        # Port of vllm-project/vllm#43371: a ReqMeta must never carry both a
        # save AND a load. When the request can load from the KV pool, force
        # skip_save so the same req_id is not queued into both the send and
        # recv threads (which double delayed-free and can crash the scheduler
        # with `assert req_id in self.requests`).
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        meta = ReqMeta.from_request_tracker(tracker, cache_transfer_granularity=16, load_spec=load_spec)
        self.assertIsNotNone(meta)
        self.assertIsNotNone(meta.load_spec)
        self.assertFalse(meta.can_save)

    def test_from_request_tracker_partial_tokens_discarded(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=20,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, cache_transfer_granularity=16, discard_partial_chunks=True, block_hashes=[b"h0"]
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 16)

    def test_from_request_tracker_keeps_intermediate_partial_prefill(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=20,
            allocated_block_ids=[0, 1],
            num_saved_tokens=16,
        )
        meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=16,
            block_hashes=[b"h0"],
            save_partial_block=True,
        )

        self.assertIsNotNone(meta)
        self.assertTrue(meta.can_save)
        self.assertEqual(meta.save_start_token, 16)
        self.assertEqual(meta.save_end_token, 16)
        self.assertEqual(meta.target_token_len, 20)

    def test_from_request_tracker_keeps_partial_decode_step(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=33,
            allocated_block_ids=[0, 1, 2],
            num_saved_tokens=32,
            num_prompt_tokens=32,
        )
        meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=16,
            block_hashes=[b"h0", b"h1"],
            save_partial_block=True,
        )

        self.assertIsNotNone(meta)
        self.assertTrue(meta.can_save)
        self.assertEqual(meta.save_start_token, 32)
        self.assertEqual(meta.save_end_token, 32)
        self.assertEqual(meta.target_token_len, 33)

    def test_from_request_tracker_defers_c8_boundary_without_hash(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=128,
            allocated_block_ids=list(range(8)),
            num_saved_tokens=0,
        )

        partial_meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=128,
            block_hashes=[f"h{i}".encode() for i in range(7)],
            save_partial_block=True,
            hash_block_size=16,
        )

        self.assertIsNotNone(partial_meta)
        self.assertTrue(partial_meta.can_save)
        self.assertEqual(partial_meta.save_end_token, 0)
        self.assertEqual(tracker.num_saved_tokens, 0)

        full_meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=128,
            block_hashes=[f"h{i}".encode() for i in range(8)],
            save_partial_block=True,
            hash_block_size=16,
        )

        self.assertIsNotNone(full_meta)
        self.assertTrue(full_meta.can_save)
        self.assertEqual(full_meta.save_end_token, 128)
        self.assertEqual(tracker.num_saved_tokens, 128)

    def test_from_request_tracker_no_discard(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=20,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, cache_transfer_granularity=16, discard_partial_chunks=False, block_hashes=[b"h0"]
        )
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
        # Provide block_hashes (2 full blocks for token_len=32 / granularity=16)
        # so the boundary_without_hash short-circuit does not zero out the save
        # length and skip; this exercises the original_block_size propagation.
        meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=16,
            original_block_size=8,
            block_hashes=[b"h0", b"h1"],
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.original_block_size, 8)


class TestAscendConnectorMetadata(unittest.TestCase):
    def test_add_request(self):
        meta = AscendConnectorMetadata(preempted_req_ids=set())
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[],
        )
        meta.add_request(req)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].req_id, "r1")

    def test_released_req_ids_seed_from_preempted(self):
        meta = AscendConnectorMetadata(preempted_req_ids={"p1"})
        self.assertEqual(meta.released_req_ids, {"p1"})
        self.assertEqual(AscendConnectorMetadata(preempted_req_ids=None).released_req_ids, set())
        self.assertEqual(AscendConnectorMetadata(preempted_req_ids=set()).released_req_ids, set())


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


class TestInferDcpMismatchInfo(unittest.TestCase):
    def test_same_dcp_returns_false(self):
        self.assertFalse(infer_dcp_mismatch_info("kv_consumer", {"prefill_dcp_size": 2}, 2))
        self.assertFalse(infer_dcp_mismatch_info("kv_producer", {"decode_dcp_size": 8}, 8, 1))

    def test_missing_peer_key_returns_false(self):
        # single-group path: peer topology absent -> local layout authoritative
        self.assertFalse(infer_dcp_mismatch_info("kv_consumer", {}, 2, 1))

    def test_consumer_prefill_dcp_mismatch_detected(self):
        self.assertTrue(infer_dcp_mismatch_info("kv_consumer", {"prefill_dcp_size": 8}, 2))

    def test_producer_decode_dcp_mismatch_detected(self):
        self.assertTrue(infer_dcp_mismatch_info("kv_producer", {"decode_dcp_size": 2}, 8))

    def test_pcp_mismatch_detected(self):
        self.assertTrue(infer_dcp_mismatch_info("kv_consumer", {"prefill_dcp_size": 2, "prefill_pcp_size": 4}, 2, 1))

    def test_non_mapping_extra_config_returns_false(self):
        self.assertFalse(infer_dcp_mismatch_info("kv_consumer", object(), 2, 1))

    def test_kv_both_returns_false(self):
        self.assertFalse(infer_dcp_mismatch_info("kv_both", {"prefill_dcp_size": 8}, 2, 1))

    def test_invalid_peer_value_treated_as_local(self):
        self.assertFalse(infer_dcp_mismatch_info("kv_consumer", {"prefill_dcp_size": "bad"}, 2, 1))
