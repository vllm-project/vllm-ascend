import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

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
    def test_creation(self):
        meta = KeyMetadata(
            model_name="test_model",
            head_or_tp_rank=0,
            pcp_rank=1,
            dcp_rank=2,
            pp_rank=3,
        )
        self.assertEqual(meta.model_name, "test_model")
        self.assertEqual(meta.head_or_tp_rank, 0)
        self.assertEqual(meta.pcp_rank, 1)
        self.assertEqual(meta.dcp_rank, 2)
        self.assertEqual(meta.pp_rank, 3)


class TestPoolKey(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("model", 0, 1, 2, 3)

    def test_hash(self):
        key1 = PoolKey(self.meta, "abc123")
        key2 = PoolKey(self.meta, "abc123")
        self.assertEqual(hash(key1), hash(key2))

    def test_hash_different(self):
        key1 = PoolKey(self.meta, "abc123")
        key2 = PoolKey(self.meta, "def456")
        self.assertNotEqual(hash(key1), hash(key2))

    def test_to_string(self):
        key = PoolKey(self.meta, "chunkhash")
        result = key.to_string()
        self.assertIn("model", result)
        self.assertIn("@pcp1", result)
        self.assertIn("@dcp2", result)
        self.assertIn("@head_or_tp_rank:0", result)
        self.assertIn("@pp_rank:3", result)
        self.assertIn("@chunkhash", result)

    def test_split_layers(self):
        key = PoolKey(self.meta, "hash1")
        layers = key.split_layers(3)
        self.assertEqual(len(layers), 3)
        for i, layer_key in enumerate(layers):
            self.assertIsInstance(layer_key, LayerPoolKey)
            self.assertEqual(layer_key.layer_id, i)
            self.assertEqual(layer_key.chunk_hash, "hash1")

    def test_usable_in_set(self):
        key1 = PoolKey(self.meta, "a")
        key2 = PoolKey(self.meta, "a")
        s = {key1, key2}
        self.assertEqual(len(s), 1)


class TestLayerPoolKey(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("model", 1, 0, 0, 0)

    def test_hash(self):
        key1 = LayerPoolKey(self.meta, "hash", 0)
        key2 = LayerPoolKey(self.meta, "hash", 0)
        self.assertEqual(hash(key1), hash(key2))

    def test_hash_different_layer(self):
        key1 = LayerPoolKey(self.meta, "hash", 0)
        key2 = LayerPoolKey(self.meta, "hash", 1)
        self.assertNotEqual(hash(key1), hash(key2))

    def test_to_string(self):
        key = LayerPoolKey(self.meta, "hash", 5)
        result = key.to_string()
        self.assertIn("@head_or_tp_rank:1", result)
        self.assertIn("@hash@5", result)

    def test_usable_in_set(self):
        key1 = LayerPoolKey(self.meta, "a", 0)
        key2 = LayerPoolKey(self.meta, "a", 1)
        s = {key1, key2}
        self.assertEqual(len(s), 2)


class TestChunkedTokenDatabase(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("model", 0, 0, 0, 0)
        self.db = ChunkedTokenDatabase(self.meta, block_size=4, partitions=None)
        self.db.set_kv_caches_base_addr([1000, 2000])
        self.db.set_block_len([64, 64])

    def test_make_key_by_hash(self):
        key = self.db._make_key_by_hash("abc")
        self.assertIsInstance(key, PoolKey)
        self.assertEqual(key.chunk_hash, "abc")

    def test_set_kv_caches_base_addr(self):
        db = ChunkedTokenDatabase(self.meta, 4, None)
        db.set_kv_caches_base_addr([100, 200])
        self.assertEqual(db.kv_caches_base_addr, [100, 200])

    def test_set_block_len(self):
        db = ChunkedTokenDatabase(self.meta, 4, None)
        db.set_block_len([32, 64])
        self.assertEqual(db.block_len, [32, 64])

    def test_prepare_value(self):
        # block_size=4, block_ids=[0,1,2], start=0, end=4
        addr_list, size_list, block_id = self.db.prepare_value(0, 4, [0, 1, 2])
        self.assertEqual(block_id, 0)
        # addr = base_addr + block_id * block_len
        self.assertEqual(addr_list, [1000 + 0, 2000 + 0])
        # size = block_len / block_size * (end - start) = 64/4*4 = 64
        self.assertEqual(size_list, [64, 64])

    def test_prepare_value_partial_block(self):
        # start=0, end=2 (half block)
        addr_list, size_list, block_id = self.db.prepare_value(0, 2, [0, 1])
        self.assertEqual(block_id, 0)
        # size = 64/4*2 = 32
        self.assertEqual(size_list, [32, 32])

    def test_prepare_value_second_block(self):
        # start=4, end=8, block_ids=[0,1,2]
        addr_list, size_list, block_id = self.db.prepare_value(4, 8, [0, 1, 2])
        self.assertEqual(block_id, 1)
        # addr = base_addr + 1 * 64
        self.assertEqual(addr_list, [1064, 2064])

    def test_prepare_value_layer(self):
        # layer_id=0, start=0, end=4, block_ids=[0,1]
        addr_list, size_list = self.db.prepare_value_layer(0, 4, [0, 1], layer_id=0)
        # length=2, addr = kv_caches_base_addr[0*2] + block_id*block_len[0]
        self.assertEqual(len(addr_list), 2)
        self.assertEqual(addr_list[0], 1000 + 0 * 64)  # base_addr[0] + block_id*block_len[0]
        self.assertEqual(addr_list[1], 2000 + 0 * 64)  # base_addr[1] + block_id*block_len[1]

    def test_prepare_value_layer_second_layer(self):
        # 4 base addrs for 2 layers * 2 kv
        db = ChunkedTokenDatabase(self.meta, block_size=4, partitions=None)
        db.set_kv_caches_base_addr([1000, 2000, 3000, 4000])
        db.set_block_len([64, 64])
        addr_list, size_list = db.prepare_value_layer(0, 4, [0, 1], layer_id=1)
        # layer_id=1, length=2, addr = kv_caches_base_addr[1*2+i] + block_id*block_len[i]
        self.assertEqual(addr_list[0], 3000 + 0 * 64)
        self.assertEqual(addr_list[1], 4000 + 0 * 64)

    def test_process_tokens_empty_hashes(self):
        results = list(self.db.process_tokens(16, []))
        self.assertEqual(results, [])

    def test_process_tokens_string_hashes(self):
        results = list(self.db.process_tokens(8, ["hash0", "hash1"]))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # start
        self.assertEqual(results[0][1], 4)  # end
        self.assertEqual(results[1][0], 4)
        self.assertEqual(results[1][1], 8)

    def test_process_tokens_bytes_hashes(self):
        # BlockHash with .hex() method
        class FakeBlockHash:
            def __init__(self, val):
                self._val = val

            def hex(self):
                return self._val

        hashes = [FakeBlockHash("aa"), FakeBlockHash("bb")]
        results = list(self.db.process_tokens(8, hashes))
        self.assertEqual(len(results), 2)

    def test_process_tokens_with_mask(self):
        # mask_num=4 means skip first block
        results = list(self.db.process_tokens(8, ["h0", "h1"], mask_num=4))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 4)

    def test_process_tokens_token_len_less_than_block(self):
        # token_len=2, block_size=4 -> one partial block
        results = list(self.db.process_tokens(2, ["h0"]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[0][1], 2)

    def test_process_tokens_exceeds_token_len(self):
        # More hashes than tokens can fill
        results = list(self.db.process_tokens(4, ["h0", "h1", "h2"]))
        # Only first block fits within token_len=4
        self.assertEqual(len(results), 1)

    def test_decode_adaptor_prefill_pp_no_partitions(self):
        key, addr, size = self.db.decode_adaptor_prefill_pp(
            ["k1"], [[1, 2]], [[10, 20]]
        )
        self.assertEqual(key, ["k1"])
        self.assertEqual(addr, [[1, 2]])
        self.assertEqual(size, [[10, 20]])

    def test_decode_adaptor_prefill_pp_single_partition(self):
        db = ChunkedTokenDatabase(self.meta, 4, partitions=[3])
        key, addr, size = db.decode_adaptor_prefill_pp(
            ["k1"], [[1, 2]], [[10, 20]]
        )
        self.assertEqual(key, ["k1"])

    def test_decode_adaptor_prefill_pp_multi_partition(self):
        db = ChunkedTokenDatabase(self.meta, 4, partitions=[2, 3])
        # addr has 10 elements: (2+3)*2 = 10 (k and v for each layer)
        key = ["k@pp_rank:0@end"]
        addr = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        size = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        new_key, new_addr, new_size = db.decode_adaptor_prefill_pp(key, addr, size)
        self.assertEqual(len(new_key), 2)
        self.assertIn("@pp_rank:0", new_key[0])
        self.assertIn("@pp_rank:1", new_key[1])
        # First partition: 2 layers * 2 = 4 elements
        self.assertEqual(len(new_addr[0]), 4)
        # Second partition: remaining
        self.assertEqual(len(new_addr[1]), 6)


class TestLoadSpec(unittest.TestCase):
    def test_creation(self):
        spec = LoadSpec(vllm_cached_tokens=10, kvpool_cached_tokens=50, can_load=True)
        self.assertEqual(spec.vllm_cached_tokens, 10)
        self.assertEqual(spec.kvpool_cached_tokens, 50)
        self.assertTrue(spec.can_load)
        self.assertEqual(spec.token_len, 0)

    def test_token_len_default(self):
        spec = LoadSpec(0, 0, False)
        self.assertEqual(spec.token_len, 0)


class TestRequestTracker(unittest.TestCase):
    def test_creation(self):
        tracker = RequestTracker(
            req_id="req1",
            token_len=100,
            allocated_block_ids=[0, 1, 2],
            num_saved_tokens=0,
        )
        self.assertEqual(tracker.req_id, "req1")
        self.assertEqual(tracker.token_len, 100)
        self.assertEqual(tracker.allocated_block_ids, [0, 1, 2])

    def test_from_new_request_flat_block_ids(self):
        new_req = MagicMock()
        new_req.req_id = "req1"
        new_req.block_ids = [10, 20, 30]
        new_req.prompt_token_ids = list(range(50))

        tracker = RequestTracker.from_new_request(new_req, num_tokens_to_compute=30)
        self.assertEqual(tracker.req_id, "req1")
        self.assertEqual(tracker.token_len, 30)
        self.assertEqual(tracker.allocated_block_ids, [10, 20, 30])
        self.assertEqual(len(tracker.token_ids), 30)
        self.assertEqual(tracker.num_saved_tokens, 0)

    def test_from_new_request_nested_block_ids(self):
        new_req = MagicMock()
        new_req.req_id = "req2"
        new_req.block_ids = [[10, 20], [30, 40]]
        new_req.prompt_token_ids = list(range(50))

        tracker = RequestTracker.from_new_request(new_req, num_tokens_to_compute=20)
        self.assertEqual(tracker.allocated_block_ids, [10, 20])

    def test_update_with_list(self):
        tracker = RequestTracker("req1", 10, [0, 1], 0)
        tracker.update([2, 3])
        self.assertEqual(tracker.allocated_block_ids, [0, 1, 2, 3])

    def test_update_with_tuple(self):
        tracker = RequestTracker("req1", 10, [0], 0)
        tracker.update(([1, 2],))
        self.assertEqual(tracker.allocated_block_ids, [0, 1, 2])

    def test_update_with_empty(self):
        tracker = RequestTracker("req1", 10, [0], 0)
        tracker.update([])
        self.assertEqual(tracker.allocated_block_ids, [0])

    def test_update_unsupported_type(self):
        tracker = RequestTracker("req1", 10, [0], 0)
        with self.assertRaises(ValueError):
            tracker.update("invalid")  # type: ignore[arg-type]


class TestReqMeta(unittest.TestCase):
    def test_from_request_tracker_basic_save(self):
        tracker = RequestTracker(
            req_id="req1",
            token_len=16,
            allocated_block_ids=[0, 1],
            num_saved_tokens=0,
        )
        meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=8,
            load_spec=None,
            skip_save=False,
            block_hashes=[],
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.req_id, "req1")
        self.assertEqual(meta.token_len_chunk, 16)
        self.assertTrue(meta.can_save)

    def test_from_request_tracker_skip_save_and_no_load(self):
        tracker = RequestTracker("req1", 16, [0], 0)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, load_spec=None, skip_save=True
        )
        self.assertIsNone(meta)

    def test_from_request_tracker_with_load_spec(self):
        tracker = RequestTracker("req1", 16, [0, 1], 0)
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, load_spec=load_spec, skip_save=True
        )
        self.assertIsNotNone(meta)
        self.assertIsNotNone(meta.load_spec)

    def test_from_request_tracker_load_spec_cannot_load(self):
        tracker = RequestTracker("req1", 16, [0], 0)
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=False)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, load_spec=load_spec, skip_save=True
        )
        # can_load=False and skip_save=True -> None
        self.assertIsNone(meta)

    def test_from_request_tracker_partial_chunk_discard(self):
        # token_len=10, block_size=8 -> num_tokens_to_save = 8
        tracker = RequestTracker("req1", 10, [0, 1], 0)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, discard_partial_chunks=True
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 8)
        self.assertEqual(tracker.num_saved_tokens, 8)

    def test_from_request_tracker_no_discard_partial_chunks(self):
        tracker = RequestTracker("req1", 10, [0, 1], 0)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, discard_partial_chunks=False
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_len_chunk, 10)

    def test_from_request_tracker_already_saved_skip(self):
        # Already saved 16 tokens, token_len=16, block_size=8
        # chunk_boundary = cdiv(16+1, 8)*8 = 24
        # num_tokens_to_save = 16
        # 16 < 24 -> skip_save=True
        tracker = RequestTracker("req1", 16, [0, 1], num_saved_tokens=16)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, load_spec=None
        )
        self.assertIsNone(meta)

    def test_from_request_tracker_with_token_ids(self):
        tracker = RequestTracker(
            "req1", 16, [0, 1], 0, token_ids=list(range(16))
        )
        meta = ReqMeta.from_request_tracker(tracker, block_size=8)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.token_ids, list(range(16)))

    def test_from_request_tracker_is_last_chunk(self):
        tracker = RequestTracker("req1", 16, [0, 1], 0)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, is_last_chunk=True
        )
        self.assertTrue(meta.is_last_chunk)

    def test_from_request_tracker_original_block_size(self):
        tracker = RequestTracker("req1", 16, [0, 1], 0)
        meta = ReqMeta.from_request_tracker(
            tracker, block_size=8, original_block_size=4
        )
        self.assertEqual(meta.original_block_size, 4)


class TestAscendConnectorMetadata(unittest.TestCase):
    def test_creation(self):
        meta = AscendConnectorMetadata(
            unfinished_request_ids={"r1", "r2"},
            preempted_req_ids={"r3"},
        )
        self.assertEqual(meta.unfinished_request_ids, {"r1", "r2"})
        self.assertEqual(meta.preempted_req_ids, {"r3"})
        self.assertEqual(meta.requests, [])

    def test_add_request(self):
        meta = AscendConnectorMetadata(set(), set())
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
    def test_creation(self):
        meta = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[0, 4],
            ends=[4, 8],
            block_ids=[0, 1],
            layer_id=2,
        )
        self.assertEqual(meta.req_id, "r1")
        self.assertEqual(meta.layer_id, 2)
        self.assertTrue(meta.is_last_chunk)
        self.assertIsNone(meta.current_event)


if __name__ == "__main__":
    unittest.main()
