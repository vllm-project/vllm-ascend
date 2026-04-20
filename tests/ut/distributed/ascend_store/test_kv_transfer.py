import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)


class FakeKey:
    def __init__(self, value: str):
        self._value = value

    def to_string(self) -> str:
        return self._value


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        for i, _ in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            if start < mask_num:
                continue
            end = min((i + 1) * self.block_size, token_len)
            yield start, end, FakeKey(f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = start // self.block_size
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = start // self.block_size
        return [2000 + layer_id * 100 + block_id], [end - start]

    def decode_adaptor_prefill_pp(self, key, addr, size):
        return key, addr, size


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        ready = threading.Event()
        thread = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=ready,
            name="test",
        )
        return thread, store

    def test_get_and_clear_finished_requests(self):
        thread, _ = self._make_thread()
        thread.set_finished_request("r1")
        thread.set_finished_request("r2")
        finished = thread.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        # Should be empty now
        finished2 = thread.get_and_clear_finished_requests()
        self.assertEqual(finished2, set())

    def test_set_finished_request(self):
        thread, _ = self._make_thread()
        thread.set_finished_request("r1")
        thread.set_finished_request("r1")  # duplicate
        self.assertEqual(thread.finished_requests, {"r1"})

    def test_lookup_success(self):
        thread, _ = self._make_thread(exists_result=[1, 0, 1])
        result = thread.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_empty(self):
        thread, _ = self._make_thread(exists_result=[])
        result = thread.lookup([])
        self.assertEqual(result, [])

    def test_lookup_exception(self):
        thread, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn error"))
        result = thread.lookup(["k1", "k2"])
        self.assertEqual(result, [False, False])

    def test_update_kv_event(self):
        thread, _ = self._make_thread()
        event1 = MagicMock()
        event2 = MagicMock()
        thread.update_kv_event([event1])
        thread.update_kv_event([event2])
        events = thread.get_kv_events()
        self.assertEqual(events, [event1, event2])
        # Should be empty after get
        events2 = thread.get_kv_events()
        self.assertEqual(events2, [])

    def test_add_request(self):
        thread, _ = self._make_thread()
        req = MagicMock()
        thread.add_request(req)
        self.assertFalse(thread.request_queue.empty())

    def test_handle_request_base_is_noop(self):
        thread, _ = self._make_thread()
        # base _handle_request is a pass
        thread._handle_request(MagicMock())


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            enable_kv_event=enable_kv_event,
        )
        return thread, store

    def test_add_stored_request(self):
        thread, _ = self._make_thread()
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        self.assertEqual(thread.stored_requests["r1"], 2)

    def test_dec_stored_request(self):
        thread, _ = self._make_thread()
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        thread.dec_stored_request("r1")
        self.assertEqual(thread.stored_requests["r1"], 1)

    def test_dec_stored_request_not_exists(self):
        thread, _ = self._make_thread()
        thread.dec_stored_request("nonexistent")

    def test_delete_finished_stored_request(self):
        thread, _ = self._make_thread()
        thread.add_stored_request("r1")
        thread.delete_finished_stored_request("r1")
        self.assertNotIn("r1", thread.stored_requests)

    def test_delete_nonexistent_stored_request(self):
        thread, _ = self._make_thread()
        thread.delete_finished_stored_request("nonexistent")

    def test_handle_request_all_exist(self):
        thread, store = self._make_thread(exists_result=[1, 1])
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_stored(self):
        thread, store = self._make_thread()
        req = ReqMeta(
            req_id="r_unknown",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_puts_missing(self):
        thread, store = self._make_thread(exists_result=[1, 0])
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(store.put_calls[0][0], ["k1"])

    def test_handle_request_with_event(self):
        thread, store = self._make_thread(exists_result=[0])
        thread.add_stored_request("r1")
        mock_event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=mock_event,
        )
        thread._handle_request(req)
        mock_event.synchronize.assert_called_once()

    def test_handle_request_consumer_role(self):
        thread, store = self._make_thread(exists_result=[0], kv_role="kv_consumer")
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_handle_request_kv_event_enabled(self):
        thread, store = self._make_thread(exists_result=[0], enable_kv_event=True)
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        thread._handle_request(req)
        events = thread.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore(exists_result=[0, 0])
        db = FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
        )
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread._handle_request(req)
        # dcp_size > 1: no slicing, all keys put
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(len(store.put_calls[0][0]), 2)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def _make_thread(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        thread = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )
        return thread, store

    def test_handle_request(self):
        thread, store = self._make_thread()
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = thread.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
        )
        return thread, store

    def test_handle_request_puts_missing(self):
        thread, store = self._make_thread(exists_result=[1, 0])
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0), LayerPoolKey(meta, "h1", 0)],
            starts=[0, 16],
            ends=[16, 32],
            block_ids=[0, 1],
            layer_id=0,
            is_last_chunk=False,
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(store.put_calls[0][0], [f"m@pcp0@dcp0@head_or_tp_rank:0@h1@0"])

    def test_handle_request_all_exist(self):
        thread, store = self._make_thread(exists_result=[1, 1])
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0), LayerPoolKey(meta, "h1", 0)],
            starts=[0, 16],
            ends=[16, 32],
            block_ids=[0, 1],
            layer_id=0,
            is_last_chunk=False,
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_last_chunk_final_layer_sets_finished(self):
        thread, store = self._make_thread(exists_result=[0], num_layers=2)
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 1)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=1,  # final_layer_id = num_layers - 1 = 1
            is_last_chunk=True,
            current_event=None,
        )
        thread._handle_request(req)
        finished = thread.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys_last_chunk(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=2,  # tp_rank=0, put_step=2, stride will filter all
            ready_event=threading.Event(),
            num_layers=2,
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],  # only 1 key, sliced with step=2 from index 0
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=True,
            current_event=None,
        )
        thread._handle_request(req)

    def test_handle_request_with_event_sync(self):
        thread, store = self._make_thread(exists_result=[0])
        meta = KeyMetadata("m", 0, 0, 0, 0)
        mock_event = MagicMock()
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=mock_event,
        )
        thread._handle_request(req)
        mock_event.synchronize.assert_called_once()

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        thread, store = self._make_thread(exists_result=[1], num_layers=2)
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 1)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=1,
            is_last_chunk=True,
            current_event=None,
        )
        thread._handle_request(req)
        finished = thread.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_dcp_size_gt1_no_slicing(self):
        store = FakeStore(exists_result=[0, 0])
        db = FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=2,
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0), LayerPoolKey(meta, "h1", 0)],
            starts=[0, 16],
            ends=[16, 32],
            block_ids=[0, 1],
            layer_id=0,
            is_last_chunk=False,
            current_event=None,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(len(store.put_calls[0][0]), 2)


class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
        )
        thread._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


if __name__ == "__main__":
    unittest.main()
