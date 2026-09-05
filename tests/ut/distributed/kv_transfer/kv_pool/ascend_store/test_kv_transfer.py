# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _circular_shift,
    record_failed_blocks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerPoolKey,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.requires_exists_before_put = True
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


class TestLayerBatchBuilderOffsets(unittest.TestCase):
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

    def test_get_missing_indices_skips_lookup_when_not_required(self):
        t, store = self._make_thread([1, 1])
        store.requires_exists_before_put = False
        store.exists = MagicMock(side_effect=AssertionError("exists should not be called"))

        result = t._get_missing_indices(["k1", "k2"])

        self.assertEqual(result, [0, 1])
        store.exists.assert_not_called()

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
        # Plain mock store: the layerwise threads are backend-agnostic.
        # `.store` is attached explicitly to pin batch_copy's return value.
        store = MagicMock()
        store.store = MagicMock(batch_copy=MagicMock(return_value=0))
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
    def _make_thread(self, external_slot_release_waiter=None, save_failure_checker=None):
        # Plain mock store: the layerwise threads are backend-agnostic.
        # `.store` is attached explicitly to pin batch_copy's return value.
        store = MagicMock()
        store.store = MagicMock(batch_copy=MagicMock(return_value=0))
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
            page_size_bytes=16,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=load_finished,
            layer_save_finished_events=save_finished,
            sync_save_events=sync_events,
            num_layers=2,
            group_builders=[builder],
            external_slot_release_waiter=external_slot_release_waiter,
            save_failure_checker=save_failure_checker,
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

    def test_source_save_failure_stops_receiver_wait(self):
        save_failure_checker = MagicMock(side_effect=RuntimeError("save thread failed"))
        thread, _, save_finished, _ = self._make_thread(save_failure_checker=save_failure_checker)
        save_finished[0] = MagicMock()
        save_finished[0].wait.return_value = False
        load_task = LayerLoadTask(
            wait_for_save_layer=0,
            transfer_tasks=[],
            layer_id=1,
        )
        thread.request_queue.put(load_task)

        with self.assertRaisesRegex(RuntimeError, "save thread failed"):
            thread._handle_request(load_task)

        save_failure_checker.assert_called_once_with()

    def test_h2d_waits_for_source_save_then_target_layer_reuse(self):
        call_order: list[tuple[str, int]] = []
        thread, _, save_finished, sync_events = self._make_thread(
            external_slot_release_waiter=lambda layer_id: call_order.append(("reuse", layer_id))
        )
        save_finished[0].set()
        sync_events[0].synchronize.side_effect = lambda: call_order.append(("save", 0))

        def record_h2d(*_args) -> int:
            call_order.append(("h2d", 1))
            return 0

        thread._batch_copy_with_limits = MagicMock(side_effect=record_h2d)
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
        load_task = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[task], layer_id=1)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(call_order, [("save", 0), ("reuse", 1), ("h2d", 1)])

    def test_empty_load_waits_for_target_layer_reuse_before_finish(self):
        load_finished_observed: list[bool] = []
        thread = None

        def wait_for_reuse(layer_id):
            assert thread is not None
            load_finished_observed.append(thread.layer_load_finished_events[layer_id].is_set())

        thread, load_finished, _, _ = self._make_thread(external_slot_release_waiter=wait_for_reuse)
        load_task = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[], layer_id=1)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(load_finished_observed, [False])
        self.assertTrue(load_finished[1].is_set())


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

    def test_handle_request_puts_all_keys_without_exists_check(self):
        t, store = self._make_thread([1, 1])
        store.requires_exists_before_put = False
        store.exists = MagicMock(side_effect=AssertionError("exists should not be called"))
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
        self.assertEqual(len(keys), 2)
        store.exists.assert_not_called()

    def test_handle_request_keeps_exists_check_for_kv_events(self):
        t, store = self._make_thread([1, 0], enable_kv_event=True)
        store.requires_exists_before_put = False
        store.exists = MagicMock(return_value=[1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(32)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)

        t._handle_request(req)

        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(t.get_kv_events()), 1)
        store.exists.assert_called_once()

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


class TestKVCacheStoreKeyLayerSendingThread(unittest.TestCase):
    def test_handle_request_puts_all_keys_without_exists_check(self):
        store = FakeStore([1, 1])
        store.requires_exists_before_put = False
        sync_event = MagicMock()
        thread = KVCacheStoreKeyLayerSendingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[threading.Event()],
            sync_save_events=[sync_event],
        )
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            is_last_chunk=False,
        )
        metadata = KeyMetadata("m", 0, 0, 0, 0)
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=request, start_block=0, end_block=2)],
            cached_process_tokens={
                0: [
                    (0, 16, [LayerPoolKey(metadata, "h0", 0)]),
                    (16, 32, [LayerPoolKey(metadata, "h1", 0)]),
                ]
            },
        )
        thread.request_queue.put([task])

        with patch.object(store, "exists", side_effect=AssertionError("exists should not be called")) as mock_exists:
            thread._handle_request([task])

        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)
        mock_exists.assert_not_called()
        sync_event.synchronize.assert_called_once()


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


class TestLayerBatchBuilder(unittest.TestCase):
    def _make_builder(self):
        db = FakeTokenDatabase()
        db.set_group_buffers(
            {0: [1000, 2000, 3000, 4000]},
            {0: [10, 20, 10, 20]},
            {0: [10, 20, 10, 20]},
            group_num_layers={0: 2},
        )
        return LayerBatchBuilder(
            db,
            page_size_bytes=100,
            num_layers=2,
        )

    @staticmethod
    def _make_request(**overrides):
        values = {
            "req_id": "r1",
            "block_ids_by_group": [[2, 3]],
            "block_ids_by_group_np": [np.asarray([2, 3])],
            "block_gvas_by_group_np": [np.asarray([10000, 20000])],
            "load_block_gvas_by_group_np": [np.asarray([30000, 40000])],
            "is_last_chunk": True,
        }
        values.update(overrides)
        return ReqMeta(**values)

    def test_builds_layer_addresses(self):
        request = self._make_request()
        task = LayerTransferTask(
            layer_id=1,
            layer_idx_in_group=1,
            block_ranges=[LayerBlockRange(request, 0, 2)],
        )

        result = self._make_builder().build(task)

        self.assertEqual(result.req_ids, ["r1"])
        np.testing.assert_array_equal(result.addr_array, [3020, 4040, 3030, 4060])
        np.testing.assert_array_equal(result.size_array, [10, 20, 10, 20])
        np.testing.assert_array_equal(result.gvas_array, [10030, 10040, 20030, 20040])

    def test_filters_and_deduplicates_blocks(self):
        request = self._make_request(
            block_ids_by_group=[[1, 1, 2]],
            block_ids_by_group_np=[np.asarray([1, 1, 2])],
            block_gvas_by_group_np=[np.asarray([100, 100, 0])],
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 3)],
        )

        result = self._make_builder().build_shared(task)

        np.testing.assert_array_equal(result.block_ids_arr, [1])
        np.testing.assert_array_equal(result.block_gvas_arr, [100])

    def test_load_offset_and_missing_metadata(self):
        builder = self._make_builder()
        request = self._make_request(
            load_block_gvas_by_group_np=[np.asarray([500])],
            load_gva_block_offset=1,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 1, 2)],
        )
        result = builder.build_shared(task, is_save=False)
        np.testing.assert_array_equal(result.block_ids_arr, [3])
        np.testing.assert_array_equal(result.block_gvas_arr, [500])

        request.load_block_gvas_by_group_np = None
        request.load_block_gvas_np = None
        with self.assertRaises(RuntimeError):
            builder.build_shared(task, is_save=False)


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


class _FakeStore:
    requires_exists_before_put = True

    def __init__(self, exists_result: list[int]):
        self.exists_result = exists_result
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []

    def set_device(self):
        return None

    def exists(self, keys: list[str]) -> list[int]:
        # Return exact number of states for requested keys.
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


class TestKVTransferMissingKeyPut(unittest.TestCase):
    def test_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0, 0)], [16], None)
        token_db.set_group_buffers({0: [1000]}, {0: [16]}, {0: [1]})
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread.add_stored_request("req-1")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(len(put_keys), 2)
        self.assertEqual(put_addrs, [[1001], [1003]])
        self.assertEqual(put_sizes, [[16], [16]])


class TestRecordFailedBlocks(unittest.TestCase):
    """Test cases for the record_failed_blocks function.

    The record_failed_blocks function takes a list of block IDs and their corresponding
    return codes from a KV transfer operation, and returns a set of block IDs that failed
    (i.e., those with non-zero return codes).
    """

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_all_blocks_succeed(self, mock_logger: MagicMock):
        """Test when all blocks are transferred successfully (all return codes are 0)."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [0, 0, 0, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        self.assertEqual(len(result), 0)
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_all_blocks_fail(self, mock_logger: MagicMock):
        """Test when all blocks fail to transfer (all return codes are non-zero)."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [1, 2, 3, 4, 5]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {1, 2, 3, 4, 5})
        self.assertEqual(len(result), 5)
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_partial_blocks_fail(self, mock_logger: MagicMock):
        """Test when some blocks fail and some succeed."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [0, 1, 0, 2, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 4})
        self.assertEqual(len(result), 2)
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_empty_lists(self, mock_logger: MagicMock):
        """Test with empty block_ids and ret_codes."""
        block_ids: list[int] = []
        ret_codes: list[int] = []

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_single_block_succeed(self, mock_logger: MagicMock):
        """Test with a single block that succeeds."""
        block_ids: list[int] = [42]
        ret_codes: list[int] = [0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_single_block_fail(self, mock_logger: MagicMock):
        """Test with a single block that fails."""
        block_ids: list[int] = [42]
        ret_codes: list[int] = [1]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {42})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_negative_return_codes(self, mock_logger: MagicMock):
        """Test with negative return codes (error conditions)."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, -1, -2]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 3})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_large_block_ids(self, mock_logger: MagicMock):
        """Test with large block ID values."""
        block_ids: list[int] = [1000000, 2000000, 3000000]
        ret_codes: list[int] = [0, 1, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2000000})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_mixed_error_codes(self, mock_logger: MagicMock):
        """Test with various non-zero error codes."""
        block_ids: list[int] = [10, 20, 30, 40, 50]
        ret_codes: list[int] = [0, -1, 100, 0, 999]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {20, 30, 50})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_logs_failed_blocks(self, mock_logger: MagicMock):
        """Test that failed blocks are logged."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, 1, 2]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 3})
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        log_msg = call_args[0]
        self.assertIn("Failed to load blocks", log_msg)
        # The last argument is the failed blocks set
        self.assertEqual(call_args[-1], {2, 3})

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_no_log_when_all_succeed(self, mock_logger: MagicMock):
        """Test that no error is logged when all blocks succeed."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_non_hybrid_single_block_semantics(self, mock_logger: MagicMock):
        """Test non-hybrid callers still map one return code to one block."""
        block_ids: list[int] = [10, 11, 12]
        ret_codes: list[int] = [0, 1, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {11})
        mock_logger.error.assert_called_once()


class TestRecordFailedBlocksEdgeCases(unittest.TestCase):
    """Additional edge case tests for record_failed_blocks."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_duplicate_block_ids_all_fail(self, mock_logger: MagicMock):
        """Test with duplicate block IDs that all fail."""
        # Note: This tests the behavior with duplicates
        # The set will deduplicate, but all should be marked as failed
        block_ids: list[int] = [1, 1, 2, 2]
        ret_codes: list[int] = [1, 1, 2, 2]

        result = record_failed_blocks(block_ids, ret_codes)

        # Set deduplicates, so we get unique failed block IDs
        self.assertEqual(result, {1, 2})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_zero_block_id_with_failure(self, mock_logger: MagicMock):
        """Test with block ID 0 failing."""
        block_ids: list[int] = [0, 1, 2]
        ret_codes: list[int] = [1, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {0})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_consecutive_failures(self, mock_logger: MagicMock):
        """Test with consecutive block failures."""
        block_ids: list[int] = [100, 101, 102, 103, 104]
        ret_codes: list[int] = [1, 1, 1, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {100, 101, 102})
        mock_logger.error.assert_called_once()


@pytest.fixture
def transfer_db():
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [4], None)
    db.set_group_buffers({0: [100, 200]}, {0: [8, 8]}, {0: [12, 12]}, group_num_layers={0: 2})
    return db


def test_finished_request_filtering_keeps_unrelated_completions(transfer_db):
    thread = KVTransferThread(MagicMock(), transfer_db, [4, 8], 0, ready_event=threading.Event())
    for request in ("a", "b", "c"):
        thread.set_finished_request(request)
    assert thread.get_and_clear_finished_requests({"a", "missing"}) == {"a"}
    thread.discard_finished_requests({"b", "absent"})
    assert thread.get_and_clear_finished_requests() == {"c"}
    assert thread._get_block_size(1) == 8
    assert thread._get_block_size(3) == 4
    thread.raise_if_failed()
    assert not thread._skip_null_blocks(ReqMeta("r", skip_null_blocks_by_group=[True]), 0, "state")


def test_legacy_database_call_signatures_delegate_to_real_address_calculation(transfer_db):
    class LegacyDatabase:
        group_block_len = transfer_db.group_block_len

        def prepare_value(self, start, end, block_ids):
            return transfer_db.prepare_value(start, end, block_ids)

        def decode_adaptor_prefill_pp(self, keys, addresses, sizes):
            return transfer_db.decode_adaptor_prefill_pp(keys, addresses, sizes)

    thread = KVTransferThread(MagicMock(), LegacyDatabase(), 4, 0)
    addresses, sizes, block = thread._prepare_value(0, 4, [2])
    assert addresses == [124, 224]
    assert sizes == [8, 8]
    assert block == 2
    assert thread._decode_adaptor_prefill_pp(["key"], [addresses], [sizes]) == (["key"], [addresses], [sizes])


@pytest.mark.parametrize("failure", [False, True])
def test_bulk_put_releases_bookkeeping_and_reports_completion_event(transfer_db, failure):
    store = MagicMock()
    store.requires_exists_before_put = False
    if failure:
        store.put.side_effect = RuntimeError("SDK put failed")
    thread = KVCacheStoreSendingThread(store, transfer_db, 4, 0)
    request = ReqMeta("r", token_len_chunk=4, block_hashes=[b"a"], block_ids=[2], event_id=7)
    thread.add_stored_request("r")
    thread.add_request(request)
    thread._handle_request(thread.request_queue.get_nowait())
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.get_completed_events() == {7: 1}
    assert thread.get_stored_request_count("r") is None
    assert thread.request_queue.unfinished_tasks == 0
    assert store.put.call_args.args[1:] == ([[124, 224]], [[8, 8]])


@pytest.mark.parametrize("scenario", ["unaligned", "pooled", "empty", "no_new_keys"])
def test_bulk_store_uses_real_coordinator_masks_and_skips_pooled_chunks(transfer_db, scenario):
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.int8)
    transfer_db.cache_coordinator = AscendStoreCoordinator([KVCacheGroupSpec(["a"], spec)], 4, 4, [4], ["c1"])
    store = MagicMock()
    store.requires_exists_before_put = False
    thread = KVCacheStoreSendingThread(store, transfer_db, 4, 0)
    request = ReqMeta("r", token_len_chunk=8, block_hashes=[b"a", b"b"], block_ids=[2, 3])
    if scenario == "unaligned":
        request.token_len_chunk = 7
    elif scenario == "pooled":
        request.load_spec = LoadSpec(0, 8, True)
    elif scenario == "empty":
        request.token_len_chunk = 0
    elif scenario == "no_new_keys":
        request.skip_null_blocks_by_group = [True]
        request.block_ids_by_group = [[0, 0]]
    thread._handle_stored_request(request)
    if scenario == "unaligned":
        assert len(store.put.call_args.args[0]) == 2
        assert store.put.call_args.args[1] == [[124, 224], [136, 236]]
    else:
        store.put.assert_not_called()


def test_thread_ignores_shutdown_sentinel_then_reports_sdk_failure(transfer_db, monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer as module

    store = MagicMock()
    store.put.side_effect = RuntimeError("SDK disconnected")
    thread = KVCacheStoreKeyLayerSendingThread(
        store,
        transfer_db,
        4,
        0,
        1,
        1,
        1,
        threading.Event(),
        2,
        [threading.Event(), threading.Event()],
        [MagicMock(), MagicMock()],
    )
    store.requires_exists_before_put = False
    request = ReqMeta("r", block_ids=[2], block_hashes=[b"a"], save_end_token=4)
    task = LayerTransferTask(0, [LayerBlockRange(request, 0, 1)])
    thread.add_request(None)
    thread.add_request([task])
    monkeypatch.setattr(module.ctypes, "CDLL", MagicMock(side_effect=OSError("not Linux")))
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert thread.ready_event.is_set()
    assert thread.request_queue.empty()
    assert thread._fatal_error.args == ("SDK disconnected",)
    store.put.assert_called_once()
    assert thread.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("tracked", [False, True])
def test_bulk_send_exception_cleanup_balances_queue_and_request(transfer_db, tracked):
    thread = KVCacheStoreSendingThread(MagicMock(), transfer_db, 4, 0)
    request = ReqMeta("r")
    if tracked:
        thread.add_stored_request("r")
    thread.request_queue.put(request)
    thread._handle_request_exception(request)
    assert thread.request_queue.unfinished_tasks == 0
    assert thread.get_stored_request_count("r") == (0 if tracked else None)
    assert thread.get_completed_events() is None
    thread.completed_events = {7: 1}
    assert thread.get_completed_events() == {7: 1}
    assert thread.get_completed_events() is None


def test_exception_cleanup_accepts_malformed_request_without_identifier(transfer_db):
    thread = KVCacheStoreSendingThread(MagicMock(), transfer_db, 4, 0)
    request = object()
    thread.add_request(request)
    thread._handle_request_exception(thread.request_queue.get_nowait())
    assert thread.request_queue.unfinished_tasks == 0
    assert thread.stored_requests == {}


def test_store_without_original_block_size_does_not_emit_incomplete_event(transfer_db):
    store = MagicMock()
    store.exists.return_value = [0]
    thread = KVCacheStoreSendingThread(store, transfer_db, 4, 0, enable_kv_event=True)
    request = ReqMeta("r", token_len_chunk=4, block_ids=[2], block_hashes=[b"a"])
    thread._handle_stored_request(request)
    assert store.put.call_args.args[1:] == ([[124, 224]], [[8, 8]])
    assert thread.get_kv_events() == []


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("result", [None, [1, 0], [0, 0], [0, 0, 1]])
def test_async_bulk_load_records_errors_and_completes_queue(transfer_db, hybrid, result):
    store = MagicMock(get=MagicMock(return_value=result))
    thread = KVCacheStoreRecvingThread(store, transfer_db, [4], 0)
    request = ReqMeta(
        "r",
        token_len_chunk=8,
        block_hashes=[b"a", b"b"],
        block_ids_by_group=[[2, 3], [4, 5]] if hybrid else [[2, 3]],
        load_spec=LoadSpec(0, 8, True),
        kv_cache_group_ids=[0],
    )
    request.load_spec.token_len = 8
    thread.request_queue.put(request)
    thread._handle_request(request)
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.request_queue.unfinished_tasks == 0
    expected = set() if hybrid or result in ([0, 0], [0, 0, 1]) else ({2, 3} if result is None else {2})
    assert thread._invalid_block_ids == expected
    assert store.get.call_args.args[1:] == ([[124, 224], [136, 236]], [[8, 8], [8, 8]])


def test_async_load_with_no_hashes_completes_without_sdk_call(transfer_db):
    store = MagicMock()
    thread = KVCacheStoreRecvingThread(store, transfer_db, 4, 0)
    request = ReqMeta("r", block_ids=[], block_hashes=[], load_spec=LoadSpec(0, 0, True))
    request.load_spec.token_len = 0
    thread.request_queue.put(request)
    thread._handle_request(request)
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.request_queue.unfinished_tasks == 0
    store.get.assert_not_called()


def test_shared_builder_collects_load_keys_once_and_handles_partial_only_range(transfer_db):
    builder = LayerBatchBuilder(transfer_db, 16, 2)
    request = ReqMeta(
        "r",
        block_ids_np=np.array([2, 3]),
        block_gvas_np=np.array([300, 400]),
        partial_save_gva_per_group=[500],
    )
    request.save_keys, request.load_keys = ["saved"], ["loaded"]
    task = LayerTransferTask(0, [LayerBlockRange(request, 0, 1), LayerBlockRange(request, 1, 1, 1)])
    shared = builder.build_shared(task)
    assert shared.load_keys == ["loaded"]
    assert shared.save_keys == ["saved", "saved"]
    np.testing.assert_array_equal(shared.block_ids_arr, [2, 3])
    np.testing.assert_array_equal(shared.block_gvas_arr, [300, 500])


def test_circular_shift_preserves_empty_and_unshifted_identity():
    values = [1, 2, 3]
    assert _circular_shift(values, 0) is values
    assert _circular_shift([], 1) == []
    assert _circular_shift(values, 1) == [2, 3, 1]
    assert values == [1, 2, 3]


@pytest.mark.parametrize("save", [False, True])
def test_batch_builder_partial_fallback_invalid_gvas_and_missing_arrays(transfer_db, save):
    builder = LayerBatchBuilder(transfer_db, 16, 2)
    request = ReqMeta(
        "r",
        block_ids=[1, 2, 3],
        last_block_gva=500,
        block_ids_np=np.array([1, 2, 3]),
        block_gvas_np=np.array([0, 300]),
        load_block_gvas_np=np.array([0, 300]),
    )
    task = LayerTransferTask(0, [LayerBlockRange(request, 0, 2, 2)])
    shared = builder.build_shared(task, is_save=save)
    np.testing.assert_array_equal(shared.block_ids_arr, [2, 3])
    np.testing.assert_array_equal(shared.block_gvas_arr, [300, 500])
    assert shared.req_ids == ["r"]
    request.block_ids_np = None
    with pytest.raises(RuntimeError, match="metadata is not initialized"):
        builder.build_shared(task, is_save=save)
    assert builder.build(LayerTransferTask(0, []), is_save=save) is None


@pytest.mark.parametrize("offset", [-1, 1])
def test_batch_builder_rejects_gva_range_outside_metadata(transfer_db, offset):
    builder = LayerBatchBuilder(transfer_db, 16, 2)
    request = ReqMeta("r", block_ids_np=np.array([1]), block_gvas_np=np.array([100]), gva_block_offset=offset)
    with pytest.raises(RuntimeError, match="does not cover requested block range"):
        builder.build_shared(LayerTransferTask(0, [LayerBlockRange(request, 0, 1)]))


def test_batch_builder_warns_on_nonpositive_gva_but_keeps_address_math(transfer_db):
    builder = LayerBatchBuilder(transfer_db, 16, 2)
    addresses, sizes, gvas = builder._build_transfer_arrays(np.array([2]), np.array([0]), 0)
    np.testing.assert_array_equal(addresses, [124])
    np.testing.assert_array_equal(sizes, [8])
    np.testing.assert_array_equal(gvas, [0])


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(0, ([100, 200], [1000, 2000], [10, 3])), (4, ([100, 200, 104, 108], [1000, 2000, 1004, 1008], [4, 3, 4, 2]))],
)
def test_packet_splitting_preserves_total_bytes_and_offsets(limit, expected):
    result = KVTransferThread._split_transfer_packets(
        np.array([100, 200]), np.array([1000, 2000]), np.array([10, 3]), limit
    )
    assert tuple(array.tolist() for array in result) == expected
    assert int(result[2].sum()) == 13


def test_batch_copy_stops_on_first_sdk_error_and_avoids_empty_calls(transfer_db):
    store = MagicMock()
    store.store.batch_copy.side_effect = [7, 0]
    thread = KVTransferThread(store, transfer_db, [4], 0)
    assert thread._get_block_size(3) == 4
    arrays = [np.array([], dtype=np.int64)] * 3
    assert thread._batch_copy_with_limits(*arrays, 1, 1, 4) == 0
    store.store.batch_copy.assert_not_called()
    assert (
        thread._batch_copy_with_limits(np.array([100, 200, 300]), np.array([10, 20, 30]), np.array([4, 4, 4]), 1, 1, 4)
        == 7
    )
    store.store.batch_copy.assert_called_once_with([100, 200], [10, 20], [4, 4], 1)


@pytest.fixture
def key_threads(transfer_db):
    store = MagicMock()
    store.exists.return_value = [0, 0]
    store.get.return_value = [0, 0]
    save_events = [threading.Event(), threading.Event()]
    load_events = [threading.Event(), threading.Event()]
    sender = KVCacheStoreKeyLayerSendingThread(
        store, transfer_db, 4, 0, 1, 1, 1, threading.Event(), 2, save_events, [MagicMock(), MagicMock()]
    )
    receiver = KVCacheStoreKeyLayerRecvingThread(
        store, transfer_db, 4, 1, 2, 1, threading.Event(), threading.Event(), load_events, save_events, 2
    )
    return sender, receiver, store


@pytest.mark.parametrize("cached", [False, True])
def test_key_layer_save_cached_and_uncached_paths_match_addresses(key_threads, cached):
    sender, _, store = key_threads
    request = ReqMeta("r", token_len_chunk=12, block_ids=[1, 2, 3], block_hashes=[b"a", b"b", b"c"], is_last_chunk=True)
    task = LayerTransferTask(1, [LayerBlockRange(request, 1, 2)])
    if cached:
        task.cached_process_tokens = sender.build_cached_process_tokens(task)
        assert len(task.cached_process_tokens[0]) == 1
    sender.add_stored_request("r")
    tasks = [task]
    sender.add_request(tasks)
    sender._handle_request(sender.request_queue.get_nowait())
    keys, addresses, sizes = store.put.call_args.args
    assert len(keys) == 1 and keys[0].endswith("@layer_id:1@62")
    assert addresses == [[224]]
    assert sizes == [[8]]
    assert sender.get_and_clear_finished_requests() == {"r"}
    assert sender.layer_save_finished_events[1].is_set()
    assert sender.request_queue.unfinished_tasks == 0
    assert tasks == []


@pytest.mark.parametrize("scenario", ["cached", "no_keys", "dcp", "intermediate", "not_last_chunk", "pending"])
def test_key_layer_save_respects_existing_keys_sharding_and_request_completion(key_threads, scenario):
    sender, _, store = key_threads
    request = ReqMeta("r", token_len_chunk=12, block_ids=[1, 2, 3], block_hashes=[b"a", b"b", b"c"], is_last_chunk=True)
    layer = 0 if scenario == "intermediate" else 1
    task = LayerTransferTask(layer, [LayerBlockRange(request, 1, 2)])
    if scenario == "cached":
        store.exists.return_value = [1]
    elif scenario == "no_keys":
        request.block_hashes = []
    elif scenario == "dcp":
        sender.dcp_size, sender.put_step, sender.tp_rank = 2, 2, 1
    elif scenario == "not_last_chunk":
        request.is_last_chunk = False
    elif scenario == "pending":
        sender.add_stored_request("r")
    sender.add_stored_request("r")
    sender.add_request([task])
    sender._handle_request(sender.request_queue.get_nowait())
    expected_done = {"r"} if scenario in {"cached", "no_keys", "dcp"} else set()
    assert sender.get_and_clear_finished_requests() == expected_done
    assert sender.request_queue.unfinished_tasks == 0
    if scenario in {"cached", "no_keys"}:
        store.put.assert_not_called()
    else:
        assert store.put.call_args.args[1] == [[124 if layer == 0 else 224]]


def test_cached_key_ranges_can_be_narrowed_for_a_later_layer(key_threads):
    sender, _, store = key_threads
    request = ReqMeta("r", token_len_chunk=12, block_ids=[1, 2, 3], block_hashes=[b"a", b"b", b"c"])
    task = LayerTransferTask(0, [LayerBlockRange(request, 0, 3)])
    task.cached_process_tokens = sender.build_cached_process_tokens(task)
    task.block_ranges = [LayerBlockRange(request, 1, 2)]
    sender.add_request([task])
    sender._handle_request(sender.request_queue.get_nowait())
    assert store.put.call_args.args[1:] == ([[124]], [[8]])
    assert sender.request_queue.unfinished_tasks == 0


def test_key_layer_load_rotates_requests_and_preserves_worker_task_list(key_threads):
    _, receiver, store = key_threads
    request = ReqMeta("r", block_ids=[1, 2], block_hashes=[b"a", "bb"], is_last_chunk=True)
    tasks = [LayerTransferTask(1, [LayerBlockRange(request, 0, 3)])]
    original_tasks = tasks.copy()
    load = LayerLoadTask(None, tasks, 1)
    receiver.add_request(load)
    receiver._handle_request(receiver.request_queue.get_nowait())
    keys, addresses, sizes = store.get.call_args.args
    assert [key.rsplit("@", 1)[-1] for key in keys] == ["bb", "61"]
    assert addresses == [[224], [212]]
    assert sizes == [[8], [8]]
    assert receiver.get_and_clear_finished_requests() == {"r"}
    assert receiver.layer_load_finished_events[1].is_set()
    assert receiver.get_event.is_set()
    assert receiver.request_queue.unfinished_tasks == 0
    assert tasks == original_tasks


@pytest.mark.parametrize("result", [None, [1], []])
def test_key_layer_load_failure_does_not_publish_success(key_threads, result):
    _, receiver, store = key_threads
    store.get.return_value = result
    request = ReqMeta("r", block_ids=[1], block_hashes=[b"a"], is_last_chunk=True)
    tasks = [LayerTransferTask(1, [LayerBlockRange(request, 0, 1)])]
    receiver.add_request(LayerLoadTask(None, tasks, 1))
    with pytest.raises(RuntimeError, match="load.*failed"):
        receiver._handle_request(receiver.request_queue.get_nowait())
    assert receiver.finished_requests == set()
    assert not receiver.layer_load_finished_events[1].is_set()
    assert not receiver.get_event.is_set()
    assert receiver.request_queue.unfinished_tasks == 0


def test_key_layer_receiver_waits_for_save_and_attention_then_handles_empty_task(key_threads):
    _, receiver, store = key_threads
    save = MagicMock(wait=MagicMock(side_effect=[False, True]))
    gate = MagicMock(wait=MagicMock(side_effect=[False, True]))
    receiver.layer_save_finished_events[0] = save
    data = LayerLoadTask(0, [], 0, gate)
    receiver.add_request(data)
    receiver._handle_request(receiver.request_queue.get_nowait())
    assert save.wait.call_count == gate.wait.call_count == 2
    save.clear.assert_called_once_with()
    store.get.assert_not_called()
    assert receiver.layer_load_finished_events[0].is_set()
    assert receiver.request_queue.unfinished_tasks == 0


def test_key_layer_task_count_validation(key_threads):
    sender, receiver, store = key_threads
    sender.add_request([])
    sender._handle_request(sender.request_queue.get_nowait())
    assert sender.request_queue.unfinished_tasks == 0
    assert sender.build_cached_process_tokens(LayerTransferTask(0, [])) is None
    tasks = [LayerTransferTask(0, []), LayerTransferTask(0, [])]
    sender.add_request(tasks)
    with pytest.raises(ValueError, match="at most one"):
        sender._handle_request(sender.request_queue.get_nowait())
    receiver.add_request(LayerLoadTask(None, tasks, 0))
    with pytest.raises(ValueError, match="at most one"):
        receiver._handle_request(receiver.request_queue.get_nowait())
    assert sender.request_queue.unfinished_tasks == receiver.request_queue.unfinished_tasks == 0
    assert store.mock_calls == []


@pytest.fixture
def gva_threads(transfer_db):
    store = MagicMock()
    store.store.batch_copy.return_value = 0
    store.batch_write_finish.return_value = [0]
    saves, loads = [threading.Event() for _ in range(4)], [threading.Event() for _ in range(4)]
    builder = LayerBatchBuilder(transfer_db, 16, 2)
    sender = KVCacheStoreLayerSendingThread(
        store, transfer_db, 4, 0, 1, 1, 16, threading.Event(), 4, saves, [MagicMock() for _ in range(4)]
    )
    receiver = KVCacheStoreLayerRecvingThread(
        store,
        transfer_db,
        4,
        0,
        1,
        1,
        16,
        threading.Event(),
        threading.Event(),
        loads,
        saves,
        [MagicMock() for _ in range(4)],
        4,
    )
    # There are four physical layers; this group's two cache-bearing layers
    # are mapped explicitly by layer_idx_in_group in each task.
    sender.layer_batch_builder = builder
    receiver.layer_batch_builder = LayerBatchBuilder(transfer_db, 16, 2)
    return sender, receiver, store


@pytest.mark.parametrize("groups", [False, True])
def test_gva_shared_builders_use_correct_save_and_load_arrays(gva_threads, groups):
    sender, receiver, _ = gva_threads
    if groups:
        sender.group_builders = [sender.layer_batch_builder]
        receiver.group_builders = [receiver.layer_batch_builder]
    request = ReqMeta(
        "r", block_ids_np=np.array([2]), block_gvas_np=np.array([300]), load_block_gvas_np=np.array([400])
    )
    task = LayerTransferTask(0, [LayerBlockRange(request, 0, 1)])
    saved = sender.build_shared_data(task)
    loaded = receiver.build_shared_data(task)
    np.testing.assert_array_equal(saved.block_gvas_arr, [300])
    np.testing.assert_array_equal(loaded.block_gvas_arr, [400])
    sender.add_stored_request("r")
    sender.delete_finished_stored_request("r")
    sender.delete_finished_stored_request("r")
    assert dict(sender.stored_requests) == {}


@pytest.mark.parametrize("has_task", [False, True])
def test_gva_empty_save_marks_layer_complete_without_sdk_call(gva_threads, has_task):
    sender, _, store = gva_threads
    tasks = [LayerTransferTask(0, [])] if has_task else []
    sender.add_request(tasks)
    sender._handle_request(sender.request_queue.get_nowait())
    assert sender.layer_save_finished_events[0].is_set() == has_task
    assert sender.request_queue.unfinished_tasks == 0
    store.store.batch_copy.assert_not_called()


def test_gva_cached_save_requires_no_key_publication_and_keeps_pending_request(gva_threads):
    sender, _, store = gva_threads
    task = LayerTransferTask(0, [], SharedBlockData(np.array([2]), np.array([300]), ["r"], [False]))
    sender.add_stored_request("r")
    sender.add_stored_request("r")
    sender.add_request([task])
    sender._handle_request(sender.request_queue.get_nowait())
    store.store.batch_copy.assert_called_once_with([300], [124], [8], 0)
    store.batch_write_finish.assert_not_called()
    assert sender.write_results == {}
    assert sender.finished_requests == set()
    assert sender.stored_requests == {"r": 1}
    assert sender.request_queue.unfinished_tasks == 0


def test_gva_empty_load_can_wait_without_an_external_failure_callback(gva_threads):
    _, receiver, store = gva_threads
    event = MagicMock(wait=MagicMock(side_effect=[False, True]))
    receiver.layer_save_finished_events[0] = event
    receiver.add_request(LayerLoadTask(0, [LayerTransferTask(1, [])], 1))
    receiver._handle_request(receiver.request_queue.get_nowait())
    assert event.wait.call_count == 2
    receiver.sync_save_events[0].synchronize.assert_called_once_with()
    assert receiver.request_queue.unfinished_tasks == 0
    assert receiver.layer_load_finished_events[1].is_set()
    store.store.batch_copy.assert_not_called()


def test_key_final_layer_does_not_complete_an_intermediate_request_chunk(key_threads):
    _, receiver, store = key_threads
    request = ReqMeta("r", block_ids=[2], block_hashes=[b"a"], is_last_chunk=False)
    task = LayerTransferTask(1, [LayerBlockRange(request, 0, 1)])
    store.get.return_value = [0]
    receiver.add_request(LayerLoadTask(None, [task], 1))
    receiver._handle_request(receiver.request_queue.get_nowait())
    assert receiver.get_and_clear_finished_requests() == set()
    assert receiver.layer_load_finished_events[1].is_set()
    assert receiver.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("direction", ["save", "load"])
@pytest.mark.parametrize("raises", [False, True])
def test_gva_copy_failure_preserves_completion_events(gva_threads, direction, raises):
    sender, receiver, store = gva_threads
    thread = sender if direction == "save" else receiver
    store.store.batch_copy.return_value = 7
    if raises:
        store.store.batch_copy.side_effect = RuntimeError("SDK disconnected")
    shared = SharedBlockData(np.array([2]), np.array([300]), ["r"], [True])
    task = LayerTransferTask(0, [], shared_block_data=shared, layer_idx_in_group=0)
    data = [task] if direction == "save" else LayerLoadTask(None, [task], 0)
    thread.add_request(data)
    with pytest.raises(RuntimeError, match="SDK disconnected" if raises else "batch_copy failed with return code 7"):
        thread._handle_request(thread.request_queue.get_nowait())
    assert thread.finished_requests == set()
    assert not thread.layer_save_finished_events[0].is_set()
    if direction == "load":
        assert not thread.layer_load_finished_events[0].is_set()
    store.batch_write_finish.assert_not_called()
    store.batch_remove_lease.assert_not_called()
    assert thread.request_queue.unfinished_tasks == 0


def test_gva_multi_task_save_combines_addresses_and_publishes_one_key(gva_threads):
    sender, _, store = gva_threads
    shared = SharedBlockData(np.array([1]), np.array([300]), ["r"], [True], save_keys=["a"])
    tasks = [LayerTransferTask(0, [], shared, write_finish_keys=["a"]), LayerTransferTask(0, [], shared)]
    sender.add_stored_request("r")
    sender.add_stored_request("r")
    sender.add_request(tasks)
    sender._handle_request(sender.request_queue.get_nowait())
    store.store.batch_copy.assert_called_once_with([300, 300], [112, 112], [8, 8], 0)
    store.batch_write_finish.assert_called_once_with(["a"], [0])
    assert sender.write_results == {}
    assert sender.get_and_clear_finished_requests() == {"r"}
    assert sender.layer_save_finished_events[0].is_set()
    assert sender.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("empty_kind", ["no_tasks", "no_ranges"])
def test_gva_empty_load_releases_external_slot_after_save_barrier(gva_threads, empty_kind):
    _, receiver, store = gva_threads
    receiver.external_slot_release_waiter = MagicMock()
    save_event = MagicMock(wait=MagicMock(side_effect=[False, True]))
    receiver.layer_save_finished_events[0] = save_event
    receiver.save_failure_checker = MagicMock()
    tasks = [] if empty_kind == "no_tasks" else [LayerTransferTask(1, [])]
    receiver.add_request(LayerLoadTask(0, tasks, 1))
    receiver._handle_request(receiver.request_queue.get_nowait())
    assert receiver.save_failure_checker.call_count == 2
    receiver.sync_save_events[0].synchronize.assert_called_once_with()
    save_event.clear.assert_called_once_with()
    receiver.external_slot_release_waiter.assert_called_once_with(1)
    assert receiver.layer_load_finished_events[1].is_set()
    assert receiver.request_queue.unfinished_tasks == 0
    store.store.batch_copy.assert_not_called()


def test_gva_final_layer_releases_deduplicated_leases_and_only_last_chunk_requests(gva_threads):
    _, receiver, store = gva_threads
    shared = SharedBlockData(np.array([1]), np.array([300]), ["done", "partial"], [True, False], load_keys=["a", "a"])
    tasks = [
        LayerTransferTask(3, [], shared, layer_idx_in_group=1),
        LayerTransferTask(3, [], shared, layer_idx_in_group=1),
    ]
    gate = MagicMock(wait=MagicMock(side_effect=[False, True]))
    receiver.external_slot_release_waiter = MagicMock()
    receiver.add_request(LayerLoadTask(None, tasks, 3, gate))
    receiver._handle_request(receiver.request_queue.get_nowait())
    store.store.batch_copy.assert_called_once_with([308, 308], [212, 212], [8, 8], 1)
    store.batch_remove_lease.assert_called_once_with(["a"])
    assert receiver.get_and_clear_finished_requests() == {"done"}
    assert len(tasks) == 2
    assert gate.wait.call_count == 2
    assert receiver.get_event.is_set()
    receiver.external_slot_release_waiter.assert_called_once_with(3)


def test_stagger_uses_controlled_clock_and_rank_slot(gva_threads, monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer as transfer_module

    _, receiver, _ = gva_threads
    receiver.tp_rank, receiver.tp_size, receiver.h2d_stagger_us = 1, 4, 10
    assert receiver._get_h2d_stagger_delay_us(2) == 30
    assert receiver._get_h2d_stagger_delay_us(3) == 0
    clock = MagicMock(side_effect=[0.0, 0.000001, 0.000030])
    monkeypatch.setattr(transfer_module.time, "perf_counter", clock)
    receiver._stagger_h2d_submit(2)
    assert clock.call_count == 3
