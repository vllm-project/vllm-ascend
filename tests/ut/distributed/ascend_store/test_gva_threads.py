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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    GVALayerwiseCapable,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.gva_threads import (
    GVALayerwiseThreadContext,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    LayerBatchBuilder,
    create_gva_recving_thread,
    create_gva_sending_thread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)
# isort: on


class _DualSpecStore(Backend, GVALayerwiseCapable):
    """Concrete double-inheritance stand-in so MagicMock spec passes the
    isinstance asserts in _GVALayerTransferThreadBase."""


class FakeTokenDatabase(ChunkedTokenDatabase):
    def __init__(self, block_size=16):
        super().__init__([KeyMetadata("m", 0, 0, 0, 0)], [block_size], None)
        self.set_group_buffers({0: [1000]}, {0: [block_size]}, {0: [1]}, group_num_layers={0: 1})


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


class TestGVALayerTransferFailures(unittest.TestCase):
    def _make_sending_thread(self):
        store = MagicMock(spec=_DualSpecStore)
        store.batch_copy.return_value = 0
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
        store = MagicMock(spec=_DualSpecStore)
        store.batch_copy.return_value = 0
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


class _LoadPathProbeDatabase(FakeTokenDatabase):
    """Two-layer single-group layout with explicit entry offsets."""

    def __init__(self):
        super().__init__()
        self.set_group_buffers(
            {0: [1000, 2000, 3000]},
            {0: [10, 20, 30]},
            {0: [100, 200, 300]},
            group_num_layers={0: 2},
            group_layer_cache_entry_offsets={0: [0, 2, 3]},
        )


class TestLoadPathEndToEndProbe(unittest.TestCase):
    """UT 5: the only direct probe against the layerwise load-path silent
    failure. Runs the real chain protocol -> layout -> transfer and asserts
    the GVAs reaching batch_copy are nonzero. If any stage degrades (lazy
    empty key-info return, skipped lease, all-zero GVAs, wrong ReqMeta
    field), the captured GVA list contains zeros and this test goes red.
    """

    def test_load_path_end_to_end_nonzero_gva(self):
        from tests.ut.distributed.ascend_store.test_gva_protocol import (
            _FakeKeyInfo,
            _make_req,
            _make_session,
        )

        # Stage 1 (protocol): GVASession.prepare_load_gvas writes the
        # allocated GVAs from the store into the request metadata.
        session = _make_session()
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True)
        req = _make_req(target_token_len=32, num_hashes=2, load_spec=load_spec)
        session._store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1, gva=100),
            _FakeKeyInfo(1, gva=101),
        ]
        session.prepare_load_gvas([req])
        self.assertEqual(req.load_block_gvas_np.tolist(), [100, 101])

        # Stage 2 (layout + transfer): the factory-built receiving thread
        # moves those GVAs through LayerBatchBuilder into batch_copy.
        store = MagicMock(spec=_DualSpecStore)
        store.batch_copy.return_value = 0
        database = _LoadPathProbeDatabase()
        ctx = GVALayerwiseThreadContext(
            m_store=store,
            token_database=database,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            num_layers=2,
            layer_save_finished_events=[threading.Event() for _ in range(2)],
            sync_save_events=[MagicMock() for _ in range(2)],
            max_transfer_blocks=0,
            max_transfer_bytes=0,
            num_kv_cache_groups=1,
            group_num_layers={0: 2},
            group_block_len={0: [10, 20, 30]},
        )
        load_finished = [threading.Event() for _ in range(2)]
        thread = create_gva_recving_thread(
            ctx,
            threading.Event(),
            threading.Event(),
            load_finished,
            h2d_stagger_us=0,
        )

        task = LayerTransferTask(
            layer_id=0,
            layer_idx_in_group=0,
            block_ranges=[LayerBlockRange(req, 0, 2)],
        )
        load_task = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[task], layer_id=0)
        thread.request_queue.put(load_task)
        thread._handle_request(load_task)

        store.batch_copy.assert_called_once()
        gvas, addrs, sizes, direction = store.batch_copy.call_args.args
        self.assertEqual(direction, 1)
        self.assertGreater(len(gvas), 0)
        self.assertTrue(all(gva > 0 for gva in gvas), f"silent load-path failure: gvas={gvas}")
        self.assertTrue(all(addr > 0 for addr in addrs))
        self.assertTrue(all(size > 0 for size in sizes))
        self.assertTrue(load_finished[0].is_set())


class TestFactoryParameterMapping(unittest.TestCase):
    """The factories must forward every thread-context field to the thread
    constructors one-to-one; a dropped or swapped parameter fails here."""

    def _make_ctx(self, store, database):
        return GVALayerwiseThreadContext(
            m_store=store,
            token_database=database,
            block_size=16,
            tp_rank=2,
            tp_size=4,
            dcp_size=1,
            page_size_bytes=64,
            num_layers=2,
            layer_save_finished_events=[threading.Event() for _ in range(2)],
            sync_save_events=[MagicMock() for _ in range(2)],
            max_transfer_blocks=8,
            max_transfer_bytes=4096,
            num_kv_cache_groups=1,
            group_num_layers={0: 2},
            group_block_len={0: [30, 34]},
        )

    def test_sending_factory_maps_all_parameters(self):
        store = MagicMock(spec=_DualSpecStore)
        database = FakeTokenDatabase()
        ctx = self._make_ctx(store, database)
        ready_event = threading.Event()

        thread = create_gva_sending_thread(ctx, ready_event)

        self.assertIs(thread.m_store, store)
        self.assertIs(thread.token_database, database)
        self.assertEqual(thread.block_size, 16)
        self.assertEqual(thread.tp_rank, 2)
        self.assertEqual(thread.tp_size, 4)
        self.assertEqual(thread.dcp_size, 1)
        self.assertIs(thread.ready_event, ready_event)
        self.assertEqual(thread.final_layer_id, 1)
        self.assertIs(thread.layer_save_finished_events, ctx.layer_save_finished_events)
        self.assertIs(thread.sync_save_events, ctx.sync_save_events)
        self.assertEqual(thread.max_transfer_blocks, 8)
        self.assertEqual(thread.max_transfer_bytes, 4096)
        self.assertEqual(len(thread.group_builders), 1)
        self.assertEqual(thread.group_builders[0].group_id, 0)
        self.assertIs(thread.layer_batch_builder, thread.group_builders[0])

    def test_recving_factory_maps_all_parameters(self):
        store = MagicMock(spec=_DualSpecStore)
        database = FakeTokenDatabase()
        ctx = self._make_ctx(store, database)
        ready_event = threading.Event()
        get_event = threading.Event()
        load_finished = [threading.Event() for _ in range(2)]
        external_waiter = MagicMock()
        failure_checker = MagicMock()

        thread = create_gva_recving_thread(
            ctx,
            ready_event,
            get_event,
            load_finished,
            h2d_stagger_us=250,
            external_slot_release_waiter=external_waiter,
            save_failure_checker=failure_checker,
        )

        self.assertIs(thread.m_store, store)
        self.assertIs(thread.token_database, database)
        self.assertEqual(thread.block_size, 16)
        self.assertEqual(thread.tp_rank, 2)
        self.assertEqual(thread.tp_size, 4)
        self.assertEqual(thread.dcp_size, 1)
        self.assertIs(thread.ready_event, ready_event)
        self.assertIs(thread.get_event, get_event)
        self.assertIs(thread.layer_load_finished_events, load_finished)
        self.assertIs(thread.layer_save_finished_events, ctx.layer_save_finished_events)
        self.assertIs(thread.sync_save_events, ctx.sync_save_events)
        self.assertEqual(thread.final_layer_id, 1)
        self.assertEqual(thread.h2d_stagger_us, 250)
        self.assertEqual(thread.max_transfer_blocks, 8)
        self.assertEqual(thread.max_transfer_bytes, 4096)
        self.assertIs(thread.external_slot_release_waiter, external_waiter)
        self.assertIs(thread.save_failure_checker, failure_checker)
        self.assertEqual(len(thread.group_builders), 1)
        self.assertIs(thread.layer_batch_builder, thread.group_builders[0])

    def test_send_and_recv_builders_are_independent(self):
        # LayerBatchBuilder owns reusable numpy buffers; the send and recv
        # threads must never share builder instances.
        store = MagicMock(spec=_DualSpecStore)
        ctx = self._make_ctx(store, FakeTokenDatabase())

        send = create_gva_sending_thread(ctx, threading.Event())
        recv = create_gva_recving_thread(
            ctx,
            threading.Event(),
            threading.Event(),
            [threading.Event() for _ in range(2)],
        )

        self.assertIsNot(send.group_builders[0], recv.group_builders[0])
        self.assertIsNot(send.layer_batch_builder, recv.layer_batch_builder)
