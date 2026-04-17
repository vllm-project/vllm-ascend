#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

import unittest
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
)


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
        self.assertIn("@5", s)
        self.assertIn("model", s)


class TestChunkedTokenDatabase(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 0, 0, 0, 0)
        self.db = ChunkedTokenDatabase(self.meta, block_size=16, partitions=None)
        self.db.set_kv_caches_base_addr([1000, 2000])
        self.db.set_block_len([160, 320])

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

    def test_process_tokens_token_len_shorter_than_all_blocks(self):
        hashes = ["a", "b", "c", "d"]
        # token_len=32 means only first 2 blocks valid
        result = list(self.db.process_tokens(32, hashes))
        self.assertEqual(len(result), 2)

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

    def test_prepare_value_layer(self):
        addr, size = self.db.prepare_value_layer(0, 16, [5, 6], layer_id=0)
        self.assertEqual(len(addr), 2)
        # layer_id=0 => kv_caches_base_addr[0*2] and [0*2+... index mod length]
        self.assertEqual(addr[0], 1000 + 5 * 160)
        self.assertEqual(addr[1], 1000 + 5 * 320)

    def test_decode_adaptor_prefill_pp_no_partitions(self):
        key, addr, size = self.db.decode_adaptor_prefill_pp(["k1"], [[1, 2]], [[10, 20]])
        self.assertEqual(key, ["k1"])

    def test_decode_adaptor_prefill_pp_single_partition(self):
        db = ChunkedTokenDatabase(self.meta, 16, partitions=[4])
        key, addr, size = db.decode_adaptor_prefill_pp(["k1"], [[1, 2]], [[10, 20]])
        self.assertEqual(key, ["k1"])

    def test_decode_adaptor_prefill_pp_multi_partition(self):
        db = ChunkedTokenDatabase(self.meta, 16, partitions=[2, 2])
        db.set_kv_caches_base_addr([1000, 2000])
        db.set_block_len([160, 320])
        keys = ["k1@pp_rank:0"]
        addrs = [[1, 2, 3, 4, 5, 6, 7, 8]]
        sizes = [[10, 20, 30, 40, 50, 60, 70, 80]]
        new_keys, new_addrs, new_sizes = db.decode_adaptor_prefill_pp(keys, addrs, sizes)
        self.assertEqual(len(new_keys), 2)
        self.assertIn("@pp_rank:0", new_keys[0])
        self.assertIn("@pp_rank:1", new_keys[1])


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


class TestReqMeta(unittest.TestCase):
    def test_from_request_tracker_basic_save(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=32,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
            token_ids=list(range(32)),
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, block_hashes=[b"h1", b"h2"]
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.req_id, "r1")
        self.assertTrue(meta.can_save)
        self.assertEqual(meta.token_len_chunk, 32)
        self.assertIsNone(meta.load_spec)

    def test_from_request_tracker_skip_save(self):
        tracker = RequestTracker(
            req_id="r1", token_len=32, allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, skip_save=True
        )
        self.assertIsNone(meta)

    def test_from_request_tracker_with_load_spec(self):
        tracker = RequestTracker(
            req_id="r1", token_len=32, allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, load_spec=load_spec, skip_save=True
        )
        self.assertIsNotNone(meta)
        self.assertIsNotNone(meta.load_spec)

    def test_from_request_tracker_load_spec_cannot_load(self):
        tracker = RequestTracker(
            req_id="r1", token_len=32, allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=False)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, load_spec=load_spec, skip_save=True
        )
        # can_load=False => load_spec set to None in meta,
        # but skip_save+load_spec input is not None, so meta is still created
        self.assertIsNotNone(meta)
        self.assertIsNone(meta.load_spec)
        self.assertFalse(meta.can_save)

    def test_from_request_tracker_partial_tokens_discarded(self):
        tracker = RequestTracker(
            req_id="r1", token_len=20, allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, discard_partial_chunks=True
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 16)

    def test_from_request_tracker_no_discard(self):
        tracker = RequestTracker(
            req_id="r1", token_len=20, allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, discard_partial_chunks=False
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 20)

    def test_from_request_tracker_already_saved(self):
        tracker = RequestTracker(
            req_id="r1", token_len=32, allocated_block_ids=[0, 1],
            num_saved_tokens=32,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16
        )
        # num_saved_tokens=32, chunk_boundary=ceil(33/16)*16=48 > 32
        # so skip_save, and no load_spec => None
        self.assertIsNone(meta)

    def test_from_request_tracker_with_original_block_size(self):
        tracker = RequestTracker(
            req_id="r1", token_len=32, allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=16, original_block_size=8
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.original_block_size, 8)


class TestAscendConnectorMetadata(unittest.TestCase):
    def test_add_request(self):
        meta = AscendConnectorMetadata(
            unfinished_request_ids=set(), preempted_req_ids=set()
        )
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
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
