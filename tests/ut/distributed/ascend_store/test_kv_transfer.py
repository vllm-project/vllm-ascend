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

import threading
import unittest
from unittest.mock import MagicMock

from vllm.distributed.kv_events import BlockStored

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


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        return self.exists_result[:len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class FakeKey:
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return self._val


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, h in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num:
                continue
            yield start, end, PoolKey(meta, f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = block_ids[start // self.block_size]
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start]

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=1, ready_event=threading.Event(),
            name="test",
        )
        return t, store

    def test_add_request(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

    def test_get_and_clear_finished_requests(self):
        t, _ = self._make_thread()
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        finished = t.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup_all_exist(self):
        t, _ = self._make_thread([1, 1, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, True, True])

    def test_lookup_partial(self):
        t, _ = self._make_thread([1, 0, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_exception(self):
        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        result = t.lookup(["k1"])
        self.assertEqual(result, [False])

    def test_update_and_get_kv_events(self):
        t, _ = self._make_thread()
        event1 = BlockStored(block_hashes=["h1"])
        event2 = BlockStored(block_hashes=["h2"])
        t.update_kv_event([event1, event2])
        events = t.get_kv_events()
        self.assertEqual(len(events), 2)
        # After get, events should be cleared
        self.assertEqual(len(t.get_kv_events()), 0)

    def test_handle_request_base_noop(self):
        t, _ = self._make_thread()
        # Base class _handle_request does nothing
        t._handle_request(MagicMock())


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=1, put_step=1,
            kv_role=kv_role, ready_event=threading.Event(),
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1", token_len_chunk=64, block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)

    def test_handle_request_all_exist_no_put(self):
        t, store = self._make_thread([1, 1])
        req = ReqMeta(
            req_id="r1", token_len_chunk=32, block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_in_stored(self):
        t, store = self._make_thread([0])
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_consumer_role(self):
        t, store = self._make_thread([0], kv_role="kv_consumer")
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_dec_nonexistent_request(self):
        t, _ = self._make_thread()
        t.dec_stored_request("nonexist")  # should not raise

    def test_delete_nonexistent_request(self):
        t, _ = self._make_thread()
        t.delete_finished_stored_request("nonexist")  # should not raise

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=event,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore([0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=2, put_step=1,
            kv_role="kv_producer", ready_event=threading.Event(),
        )
        req = ReqMeta(
            req_id="r1", token_len_chunk=32, block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=1, ready_event=threading.Event(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1", token_len_chunk=32, block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=1, put_step=1,
            ready_event=threading.Event(), num_layers=num_layers,
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1", keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1", keys=[], starts=[], ends=[],
            block_ids=[], layer_id=0, is_last_chunk=True,
        )
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0], ends=[16], block_ids=[0],
            layer_id=0, is_last_chunk=False, current_event=event,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store, token_database=db, block_size=16,
            tp_rank=0, dcp_size=1, ready_event=threading.Event(),
            get_event=get_event,
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0], ends=[16], block_ids=[0],
            layer_id=0,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


if __name__ == "__main__":
    unittest.main()
