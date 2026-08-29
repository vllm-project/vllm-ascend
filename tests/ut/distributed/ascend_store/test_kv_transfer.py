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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
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


class FakeTokenDatabase(ChunkedTokenDatabase):
    def __init__(self, block_size=16):
        super().__init__([KeyMetadata("m", 0, 0, 0, 0)], [block_size], None)
        self.set_group_buffers({0: [1000]}, {0: [block_size]}, {0: [1]}, group_num_layers={0: 1})


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
        block_idx = start // self.get_block_size(kv_cache_group_id)
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

    def test_queue_lifecycle(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

        t.set_finished_request("r1")
        t.set_finished_request("r2")
        self.assertEqual(t.get_and_clear_finished_requests(), {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup(self):
        for exists_result, expected in (
            ([1, 1, 1], [True, True, True]),
            ([1, 0, 1], [True, False, True]),
        ):
            with self.subTest(exists_result=exists_result):
                t, _ = self._make_thread(exists_result)
                self.assertEqual(t.lookup(["k1", "k2", "k3"]), expected)

        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        self.assertEqual(t.lookup(["k1"]), [False])

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

    def test_fatal_error_stops_before_next_queued_task(self):
        t, _ = self._make_thread()
        handled = []

        def fail(request):
            handled.append(request)
            raise RuntimeError("transfer failed")

        t._handle_request = fail
        t.add_request("first")
        t.add_request("second")

        t.start()
        t.join(timeout=1)

        self.assertFalse(t.is_alive())
        self.assertEqual(handled, ["first"])
        self.assertEqual(t.request_queue.qsize(), 1)
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            t.raise_if_failed()


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(
        self,
        exists_result=None,
        kv_role="kv_producer",
        enable_kv_event=False,
        block_size=16,
    ):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase(block_size=block_size)
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=block_size,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_save_decisions(self):
        cases = [
            ([1, 0, 1, 0], "kv_producer", True, 1, 2),
            ([1, 1], "kv_producer", True, 0, 0),
            ([0], "kv_producer", False, 0, 0),
            ([0], "kv_consumer", True, 1, 1),
        ]
        for exists, role, tracked, put_count, key_count in cases:
            with self.subTest(exists=exists, role=role, tracked=tracked):
                t, store = self._make_thread(exists, kv_role=role)
                req = ReqMeta(
                    req_id="r1",
                    token_len_chunk=16 * len(exists),
                    block_ids=list(range(len(exists))),
                    block_hashes=[f"h{i}" for i in range(len(exists))],
                    current_event=None,
                )
                if tracked:
                    t.add_stored_request("r1")
                t.request_queue.put(req)
                t._handle_request(req)
                self.assertEqual(len(store.put_calls), put_count)
                if put_count:
                    self.assertEqual(len(store.put_calls[0][0]), key_count)

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

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("nonexistent")
        t.delete_finished_stored_request("nonexistent")
        self.assertEqual(t.stored_requests, {"r1": 2})
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_handle_request_sync_and_dcp(self):
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

    def test_handle_request_skips_compressed_hit_in_raw_token_domain(self):
        t, store = self._make_thread([0, 0], block_size=64)
        t.token_database.group_cache_families["kv"][0] = "c4"
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=128,
            block_ids=[0, 1],
            block_hashes=[f"h{i}" for i in range(8)],
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=63,
                kvpool_store_skip_tokens=64,
                can_load=True,
            ),
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        keys, addrs, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)
        self.assertEqual(addrs, [[1001]])

    def test_save_exception_cleans_queue_lifecycle(self):
        t, store = self._make_thread([0])
        store.put = MagicMock(side_effect=RuntimeError("put failed"))
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)
        self.assertNotIn("r1", t.stored_requests)


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


class TestKVTransferTpMismatchDispatch(unittest.TestCase):
    """TP-mismatch worker dispatch wiring for Sending/Recving threads."""

    def _make_sending(self, worker=None, exists_result=None):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
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
            enable_kv_event=False,
            worker=worker,
        )
        return t, store

    def _make_recving(self, worker=None):
        store = FakeStore([0, 0, 0, 0])
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
            worker=worker,
        )
        return t, store

    def test_sending_dispatch_and_normal_path(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_sending(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._store_kv_tp_mismatch.assert_called_once_with(req)

        t, store = self._make_sending(worker=None, exists_result=[1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)  # normal path executed

    def test_recving_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_called_once()
        args = worker._load_kv_tp_mismatch.call_args.args
        # (block_hashes, block_ids, token_len, mask_num)
        self.assertEqual(args[2], 16)  # token_len
        self.assertEqual(args[3], 0)  # mask_num

    def test_recving_tp_mismatch_terminal_paths(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_not_called()
        self.assertEqual(t.get_and_clear_finished_requests(), {"r1"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

        worker = MagicMock()
        worker.tp_mismatch = True
        worker._load_kv_tp_mismatch.side_effect = RuntimeError("load failed")
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        with self.assertRaises(RuntimeError):
            t._handle_request(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
