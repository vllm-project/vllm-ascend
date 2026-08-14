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

import numpy as np

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LayerBatchReqMeta,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
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


class TestLayerBatchBuilder(unittest.TestCase):
    def test_uses_real_offsets_for_variable_cache_entries_per_layer(self):
        database = FakeTokenDatabase()
        database.set_group_buffers(
            {0: [1000, 2000, 3000]},
            {0: [10, 20, 30]},
            {0: [100, 200, 300]},
            group_num_layers={0: 2},
            group_layer_cache_entry_offsets={0: [0, 2, 3]},
        )
        builder = LayerBatchBuilder(
            database,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=60,
            num_layers=2,
        )

        layer_0 = builder._build_transfer_arrays(
            np.asarray([2]),
            np.asarray([500]),
            layer_id=0,
        )
        layer_1 = builder._build_transfer_arrays(
            np.asarray([2]),
            np.asarray([500]),
            layer_id=1,
        )

        np.testing.assert_array_equal(layer_0[0], [1200, 2400])
        np.testing.assert_array_equal(layer_0[1], [10, 20])
        np.testing.assert_array_equal(layer_0[2], [500, 510])
        np.testing.assert_array_equal(layer_1[0], [3600])
        np.testing.assert_array_equal(layer_1[1], [30])
        np.testing.assert_array_equal(layer_1[2], [530])


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


class TestGVALayerTransferFailures(unittest.TestCase):
    def _make_sending_thread(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 0
        store.batch_write_finish.return_value = [0]
        builder = MagicMock()
        builder.build_addrs.return_value = LayerBatchReqMeta(
            req_ids=["r1"],
            layer_id=0,
            is_last_chunks=[True],
            addr_array=np.asarray([10]),
            size_array=np.asarray([16]),
            gvas_array=np.asarray([100]),
        )
        save_finished = threading.Event()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=16,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[save_finished],
            sync_save_events=[MagicMock()],
            group_builders=[builder],
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            shared_block_data=SharedBlockData(
                block_ids_arr=np.asarray([0]),
                block_gvas_arr=np.asarray([100]),
                req_ids=["r1"],
                is_last_chunks=[True],
                save_keys=["k0"],
            ),
            write_finish_keys=["k0"],
        )
        thread.add_stored_request("r1")
        thread.request_queue.put([task])
        return thread, store, save_finished, task

    def test_write_finish_failure_does_not_complete_layer(self):
        thread, store, save_finished, task = self._make_sending_thread()
        store.batch_write_finish.return_value = [1]

        with self.assertRaisesRegex(RuntimeError, "batch_write_finish failed"):
            thread._handle_request([task])

        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(save_finished.is_set())

    def test_write_finish_uses_last_actual_save_task(self):
        thread, store, _, task = self._make_sending_thread()
        thread.final_layer_id = 1

        thread._handle_request([task])

        store.batch_write_finish.assert_called_once_with(["k0"], [0])


class TestGVALayerReceivingTaskOwnership(unittest.TestCase):
    def _make_thread(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 0
        load_finished = [threading.Event(), threading.Event()]
        save_finished = [threading.Event(), threading.Event()]
        sync_events = [MagicMock(), MagicMock()]
        builder = MagicMock()
        builder.build_addrs.return_value = LayerBatchReqMeta(
            req_ids=["r1"],
            layer_id=1,
            is_last_chunks=[False],
            addr_array=np.asarray([10]),
            size_array=np.asarray([16]),
            gvas_array=np.asarray([100]),
        )
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=16,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=load_finished,
            layer_save_finished_events=save_finished,
            sync_save_events=sync_events,
            num_layers=2,
            group_builders=[builder],
        )
        return thread, load_finished, save_finished, sync_events

    def test_handle_request_does_not_clear_worker_owned_tasks(self):
        thread, _, _, _ = self._make_thread()
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[],
            shared_block_data=SharedBlockData(
                block_ids_arr=np.asarray([0]),
                block_gvas_arr=np.asarray([100]),
                req_ids=["r1"],
                is_last_chunks=[False],
            ),
        )
        transfer_tasks = [task]
        load_task = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=transfer_tasks,
            layer_id=1,
        )
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(transfer_tasks, [task])

    def test_empty_reuse_gate_waits_for_non_saving_rank_compute(self):
        thread, load_finished, save_finished, sync_events = self._make_thread()
        save_finished[0].set()
        load_task = LayerLoadTask(
            wait_for_save_layer=0,
            transfer_tasks=[],
            layer_id=1,
        )
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        sync_events[0].synchronize.assert_called_once_with()
        self.assertFalse(save_finished[0].is_set())
        self.assertTrue(load_finished[1].is_set())


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

    def test_handle_request_skips_compressed_hit_in_raw_token_domain(self):
        t, store = self._make_thread([0, 0])
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


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=32,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[],
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
        t, store = self._make_thread([0], num_layers=2)
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
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=0, is_last_chunk=False, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.get_kv_events(), [])

    def test_layerwise_kv_event_uses_missing_blocks_from_previous_layers(self):
        t, store = self._make_thread([0], num_layers=2)
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


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
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

    def test_sending_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_sending(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._store_kv_tp_mismatch.assert_called_once_with(req)

    def test_sending_normal_path_when_worker_none(self):
        # worker=None -> tp_mismatch dispatch skipped, normal store path runs.
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

    def test_recving_tp_mismatch_missing_load_spec_finishes(self):
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

    def test_recving_tp_mismatch_task_done_on_exception(self):
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
