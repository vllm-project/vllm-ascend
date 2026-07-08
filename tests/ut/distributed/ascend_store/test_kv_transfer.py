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

import threading
import unittest
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LoadSpec,
    PoolKey,
    ReqMeta,
)

# isort: on
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
        return self.exists_result[: len(keys)]

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
        return [2000 + layer_id * 100 + block_id], [end - start], block_id

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class MaskedFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self, block_size=16, masks=([True],)):
        super().__init__(block_size)
        self.masks = masks

    def store_mask(self, token_len, num_prompt_tokens=None):
        return self.masks

    def load_mask(self, block_hashes, token_len):
        return self.masks

    def mask_allows_chunk(self, masks, kv_cache_group_id, start):
        if masks is None:
            return True
        block_idx = start // self.block_size
        return block_idx < len(masks[kv_cache_group_id]) and masks[kv_cache_group_id][block_idx]


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
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
        event1 = BlockStored(
            block_hashes=["h1"],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["h2"],
            parent_block_hash="h1",
            token_ids=[4, 5, 6],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
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
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
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
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
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
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
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
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
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
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
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
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)

    def test_handle_request_applies_store_mask(self):
        store = FakeStore([0, 0])
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_applies_load_mask(self):
        store = FakeStore()
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.get_calls[0]
        self.assertEqual(len(keys), 1)


class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2, enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
            token_ids=list(range(num_keys * 16)),
            original_block_size=16,
            block_hashes=[f"h{i}".encode() for i in range(num_keys)],
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        _meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=event,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_layerwise_kv_event_published_on_final_layer(self):
        t, store = self._make_thread([0], num_layers=2, enable_kv_event=True)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])
        self.assertEqual(events[0].token_ids, list(range(16)))
        self.assertEqual(events[0].block_size, 16)

    def test_layerwise_kv_event_not_published_before_final_layer(self):
        t, store = self._make_thread([0], num_layers=2, enable_kv_event=True)
        req = self._make_layer_req(layer_id=0, is_last_chunk=False, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.get_kv_events(), [])

    def test_layerwise_kv_event_uses_missing_blocks_from_previous_layers(self):
        t, store = self._make_thread([0], num_layers=2, enable_kv_event=True)
        first_layer_req = self._make_layer_req(layer_id=0, is_last_chunk=True, num_keys=1)
        t.add_stored_request(first_layer_req.req_id)
        t.request_queue.put(first_layer_req)
        t._handle_request(first_layer_req)
        t.m_store.exists_result = [1]
        final_layer_req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(final_layer_req)
        t._handle_request(final_layer_req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])


class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
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
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


class _BlockingPutStore(FakeStore):
    """FakeStore whose put() blocks until release() is called.

    Used to make a put reliably exceed put_timeout_s so the request-level
    circuit breaker fires deterministically (no wall-clock sleeps in the test).
    """

    def __init__(self, exists_result=None):
        super().__init__(exists_result)
        self._release = threading.Event()

    def put(self, keys, addrs, sizes):
        # Block until released; 10s safety bound so a broken test can't hang.
        self._release.wait(timeout=10)
        super().put(keys, addrs, sizes)

    def release(self):
        self._release.set()


class TestKVCacheStoreSendingThreadPutTimeout(unittest.TestCase):
    """Request-level put-timeout circuit breaker (see kv_transfer.py)."""

    def _make_thread(self, store, put_timeout_s=0.05):
        db = FakeTokenDatabase()
        return KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=False,
            put_timeout_s=put_timeout_s,
        )

    def _make_req(self, req_id="r1"):
        return ReqMeta(
            req_id=req_id,
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )

    def test_put_timeout_circuit_breaks_and_still_accounts(self):
        # put blocks past the timeout -> request is circuit-broken, but
        # accounting must still run in finally (count -> 0, queue drained),
        # so blocks get released. A timeout must never become a leak.
        store = _BlockingPutStore([0, 0])
        t = self._make_thread(store, put_timeout_s=0.05)
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        try:
            t._handle_request(req)
            self.assertTrue(t._is_request_failed("r1"))
            self.assertEqual(t.stored_requests["r1"], 0)
            self.assertEqual(t.request_queue.unfinished_tasks, 0)
        finally:
            store.release()  # let the background executor thread finish & exit

    def test_failed_request_skips_remaining_puts(self):
        # Already-failed request: the breaker skips the put even with a fast
        # store, and the chunk is still accounted (so the request can finish).
        store = FakeStore([0, 0])
        t = self._make_thread(store, put_timeout_s=10.0)
        t._mark_request_failed("r1")
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)
        self.assertEqual(t.stored_requests["r1"], 0)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_normal_put_within_timeout_not_broken(self):
        # Healthy fast put within the timeout: not broken, put happens once,
        # accounting normal. (Regression: large timeout == old behavior.)
        store = FakeStore([0, 0])
        t = self._make_thread(store, put_timeout_s=10.0)
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertFalse(t._is_request_failed("r1"))
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(t.stored_requests["r1"], 0)

    def test_put_timeout_does_not_use_shared_executor(self):
        # A hung backend put must not consume the shared executor forever.
        store = FakeStore([0, 0])
        t = self._make_thread(store, put_timeout_s=10.0)
        if hasattr(t, "executor"):
            t.executor.submit = MagicMock(side_effect=AssertionError("shared executor should not be used"))
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        if hasattr(t, "executor"):
            t.executor.submit.assert_not_called()
        self.assertEqual(len(store.put_calls), 1)

    def test_none_put_timeout_disables_timeout(self):
        store = FakeStore([0, 0])
        t = self._make_thread(store, put_timeout_s=None)
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertFalse(t._is_request_failed("r1"))
        self.assertEqual(len(store.put_calls), 1)

    def test_non_positive_put_timeout_disables_timeout(self):
        store = FakeStore([0, 0])
        t = self._make_thread(store, put_timeout_s=0)
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertIsNone(t.put_timeout_s)
        self.assertFalse(t._is_request_failed("r1"))
        self.assertEqual(len(store.put_calls), 1)

    def test_put_exception_still_accounts(self):
        # A put that raises (not a timeout) is caught and still accounted.
        store = FakeStore([0, 0])
        store.put = MagicMock(side_effect=RuntimeError("boom"))
        t = self._make_thread(store, put_timeout_s=10.0)
        req = self._make_req("r1")
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.stored_requests["r1"], 0)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_delete_finished_clears_failed_flag(self):
        # Cleaning up a request must also clear its circuit-breaker flag so the
        # _failed_requests set can't grow unbounded.
        store = FakeStore([0, 0])
        t = self._make_thread(store)
        t._mark_request_failed("r1")
        self.assertTrue(t._is_request_failed("r1"))
        t.add_stored_request("r1")
        t.delete_finished_stored_request("r1")
        self.assertFalse(t._is_request_failed("r1"))


if __name__ == "__main__":
    unittest.main()
