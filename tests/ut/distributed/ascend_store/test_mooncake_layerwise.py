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

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import (
    LayerwiseSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import kv_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    LayerBatchBuilder,
    _build_range_debug_payload,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LayerBlockRange,
    LayerRangeReqMeta,
    LayerTransferTask,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import (
    LayerwiseWriter,
    LayerwiseSenderExecutor,
    LayerWriteRange,
)

# isort: on


def make_token_database() -> ChunkedTokenDatabase:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [1000, 2000, 3000]},
        {0: [10, 20, 30]},
        {0: [100, 200, 300]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 2, 3]},
    )
    return database


class TestKVPoolRangeDebug(unittest.TestCase):
    def test_audit_events_follow_ascend_config(self):
        cases = (
            (kv_transfer, kv_transfer._emit_range_debug_event, ("save", 3, [[10]], [[0]], [10]), "range"),
            (kv_transfer, kv_transfer._emit_commit_debug_event, (3, 1, [0]), "commit"),
            (mooncake_backend, mooncake_backend._emit_whole_key_debug_event, ("get", 1), "whole_key"),
        )
        for module, emit, args, event in cases:
            for enabled in (False, True):
                with (
                    self.subTest(event=event, enabled=enabled),
                    patch.object(
                        module, "get_ascend_config", return_value=SimpleNamespace(kv_pool_range_debug=enabled)
                    ),
                    patch.object(module.logger, "info") as log,
                ):
                    emit(*args)
                    if enabled:
                        log.assert_called_once()
                        self.assertEqual(json.loads(log.call_args.args[2])["event"], event)
                        log.side_effect = RuntimeError("logging failed")
                        emit(*args)  # Audit failures must not abort transfers.
                    else:
                        log.assert_not_called()


class TestMooncakeLayerBatchBuilder(unittest.TestCase):
    def test_range_debug_payload_reports_per_key_bytes_and_offsets(self):
        payload = _build_range_debug_payload(
            "save",
            3,
            [[10, 20], [7]],
            [[30, 40], [50]],
            [30, -1],
        )

        self.assertEqual(payload["event"], "range")
        self.assertEqual(payload["layer_id"], 3)
        self.assertEqual(payload["requested_bytes"], [30, 7])
        self.assertEqual(payload["object_offsets"], [[30, 40], [50]])
        self.assertEqual(payload["results"], [30, -1])

    def test_key_major_ranges_use_full_object_offsets(self):
        request = ReqMeta("r1", block_ids=[2], block_hashes=[])
        request.save_block_keys = ["key"]
        task = LayerTransferTask(
            layer_id=1,
            layer_idx_in_group=1,
            block_ranges=[LayerBlockRange(request, 0, 1)],
            use_key_major_ranges=True,
        )
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)

        result = builder.build(task)

        self.assertIsInstance(result, LayerRangeReqMeta)
        assert isinstance(result, LayerRangeReqMeta)
        self.assertEqual(result.keys, ["key"])
        self.assertEqual(result.block_ids, [2])
        self.assertEqual(result.all_buffers, [[3600]])
        self.assertEqual(result.all_sizes, [[30]])
        self.assertEqual(result.all_offsets, [[30]])

    def test_range_limits_split_rows_and_large_segments(self):
        batches = KVTransferThread._range_transfer_batches(
            ["k0", "k1"],
            [[100], [200]],
            [[25], [5]],
            [[1000], [2000]],
            max_transfer_blocks=1,
            max_transfer_bytes=10,
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (["k0"], [[100, 110, 120]], [[10, 10, 5]], [[1000, 1010, 1020]]))
        self.assertEqual(batches[1], (["k1"], [[200]], [[5]], [[2000]]))


class TestMooncakeLayerSaveSession(unittest.TestCase):
    def test_common_writer_runs_on_the_existing_sender_and_does_not_close_it(self):
        sender, store, _, _ = self._make_sender([])
        store.layerwise_data_plane = "gva"
        store.batch_alloc.return_value = [10000]
        store.batch_write_finish.return_value = [0]
        threads = []

        def copy(*args):
            threads.append(threading.get_ident())
            return 0

        store.batch_copy_gva.side_effect = copy
        sender.start()
        self.assertTrue(sender.ready_event.wait(5))
        queue = LayerwiseSenderExecutor(sender)
        writer = LayerwiseWriter(store, lease_ttl_ms=10000, executor=queue)
        try:
            handle = writer.prepare(
                {"object": 8},
                {
                    "l0": (LayerWriteRange("object", 1000, 0, 4),),
                    "l1": (LayerWriteRange("object", 2000, 4, 4),),
                },
            )
            writer.submit_layer(handle, "l0", MagicMock())
            writer.submit_layer(handle, "l1", MagicMock())
            writer.finish(handle).result(timeout=5)
            writer.close()
            self.assertEqual(threads, [sender.ident, sender.ident])
            self.assertTrue(sender.is_alive())
            self.assertEqual(queue.submit(lambda: 42).result(timeout=5), 42)
        finally:
            writer.close()
            sender.stop()
            sender.join(5)
        self.assertFalse(sender.is_alive())

    def test_stopped_sender_resolves_queued_writer_futures_before_cleanup(self):
        sender, store, _, _ = self._make_sender([])
        store.layerwise_data_plane = "gva"
        entered, release = threading.Event(), threading.Event()

        def allocate(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release allocation")
            return [10000]

        store.batch_alloc.side_effect = allocate
        store.batch_write_finish.return_value = [0]
        sender.start()
        self.assertTrue(sender.ready_event.wait(5))
        writer = LayerwiseWriter(store, lease_ttl_ms=10000, executor=LayerwiseSenderExecutor(sender))
        try:
            handle = writer.prepare({"object": 4}, {"l0": (LayerWriteRange("object", 1000, 0, 4),)})
            self.assertTrue(entered.wait(5))
            writer.submit_layer(handle, "l0", MagicMock())
            sender.stop()
            release.set()
            sender.join(5)
            self.assertFalse(sender.is_alive())
            self.assertTrue(handle.copy_futures[-1].cancelled())
            writer.close()
            store.batch_copy_gva.assert_not_called()
            store.batch_write_finish.assert_called_once_with(["object"], [-1])
            self.assertFalse(writer._handles)
        finally:
            release.set()
            sender.stop()
            sender.join(5)
            writer.close()

    @staticmethod
    def _make_sender(keys):
        store = MagicMock()
        # Range APIs may return the positive number of bytes moved on success.
        store.batch_copy_put.side_effect = lambda keys, *_: [30] * len(keys)
        store.batch_commit.side_effect = lambda keys: [0] * len(keys)
        store.batch_revoke.side_effect = lambda keys: [0] * len(keys)
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [(key, 0) for key in keys])
        save_finished = [threading.Event(), threading.Event()]
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=make_token_database(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=save_finished,
            sync_save_events=[MagicMock(), MagicMock()],
            group_builders=[builder],
            put_started_keys=set(keys),
            session_tracker=tracker,
        )
        request = ReqMeta("r1", block_ids=list(range(2, 2 + len(keys))), block_hashes=[], is_last_chunk=True)
        request.save_block_keys = keys

        tasks = []
        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=layer_id,
                block_ranges=[LayerBlockRange(request, 0, len(keys))],
                shared_block_data=builder.build_shared(
                    LayerTransferTask(
                        layer_id=layer_id,
                        block_ranges=[LayerBlockRange(request, 0, len(keys))],
                        use_key_major_ranges=True,
                    )
                ),
                use_key_major_ranges=True,
            )
            tasks.append(task)
        thread.prepare_write_step([[task] for task in tasks])
        return thread, store, tracker, tasks

    def test_final_layer_commits_after_all_ranges(self):
        thread, store, tracker, tasks = self._make_sender(["key"])
        for task in tasks:
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            thread._handle_request([task])

        self.assertEqual(store.batch_copy_put.call_count, 2)
        store.batch_commit.assert_called_once_with(["key"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("key", 0)])

    def test_partial_copy_failure_only_commits_complete_successful_keys(self):
        thread, store, tracker, tasks = self._make_sender(["good", "bad"])
        store.batch_copy_put.side_effect = [[30, -1], [30]]
        for task in tasks:
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            thread._handle_request([task])
        self.assertEqual(store.batch_copy_put.call_args_list[1].args[0], ["good"])
        store.batch_revoke.assert_called_once_with(["bad"])
        store.batch_commit.assert_called_once_with(["good"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("good", 0)])
        self.assertFalse(thread._put_started_keys)

    def test_skipped_required_layer_revokes_started_objects_without_commit(self):
        thread, store, tracker, tasks = self._make_sender(["key"])
        last = tasks[-1]
        thread.add_stored_request("r1")
        thread.request_queue.put([last])
        with self.assertRaisesRegex(ValueError, "Required object layers"):
            thread._handle_request([last])
        store.batch_commit.assert_not_called()
        store.batch_revoke.assert_called_once_with(["key"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])
        self.assertFalse(thread._put_started_keys)
        self.assertTrue(thread.layer_save_finished_events[-1].is_set())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_partial_revoke_retains_only_failed_sessions_for_shutdown_retry(self):
        thread, store, tracker, _ = self._make_sender(["good", "bad"])
        store.batch_revoke.side_effect = lambda keys: [0 if key == "good" else -1 for key in keys]
        with self.assertRaisesRegex(RuntimeError, "revoke failed"):
            thread._revoke_range_keys(["good", "bad"])
        self.assertEqual(thread._put_started_keys, {"bad"})
        self.assertIsNotNone(thread._fatal_error)
        with self.assertRaisesRegex(RuntimeError, "Cannot queue cleanup"):
            thread.add_revoke_request(["bad"])
        store.batch_revoke.side_effect = lambda keys: [0] * len(keys)
        thread.cleanup_pending_puts()
        self.assertEqual(store.batch_revoke.call_args.args[0], ["bad"])
        self.assertFalse(thread._put_started_keys)
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_unknown_revoke_results_preserve_all_ownership(self):
        for response in (None, [], [True], RuntimeError("reply lost")):
            with self.subTest(response=response):
                thread, store, _, _ = self._make_sender(["key"])
                store.batch_revoke.side_effect = response if isinstance(response, Exception) else None
                store.batch_revoke.return_value = response
                with self.assertRaises((RuntimeError, ValueError)):
                    thread._revoke_range_keys(["key"])
                self.assertEqual(thread._put_started_keys, {"key"})

    def test_commit_success_is_not_revoked_when_sibling_cleanup_fails(self):
        thread, store, tracker, tasks = self._make_sender(["good", "bad"])
        store.batch_commit.side_effect = lambda keys: [0 if key == "good" else -1 for key in keys]
        store.batch_revoke.side_effect = lambda keys: [-1] * len(keys)
        for index, task in enumerate(tasks):
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            if index == 0:
                thread._handle_request([task])
            else:
                with self.assertRaisesRegex(RuntimeError, "revoke failed"):
                    thread._handle_request([task])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("good", 0)])
        self.assertEqual(thread._put_started_keys, {"bad"})
        self.assertTrue(all(call.args[0] == ["bad"] for call in store.batch_revoke.call_args_list))
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_shutdown_keeps_backend_open_until_put_cleanup_succeeds(self):
        thread, store, _, _ = self._make_sender(["key"])
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker._closed = False
        worker.kv_send_thread, worker.kv_recv_thread = thread, None
        worker._step_load_keys = set()
        worker.m_store = store
        store.batch_revoke.side_effect = [[-1], [0]]
        with self.assertRaisesRegex(RuntimeError, "revoke failed"):
            worker.shutdown()
        store.close.assert_not_called()
        self.assertFalse(worker._closed)
        self.assertEqual(thread._put_started_keys, {"key"})
        worker.shutdown()
        store.close.assert_called_once()
        self.assertTrue(worker._closed)

    def test_shutdown_keeps_read_lease_ownership_until_release_succeeds(self):
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker._closed = False
        worker.kv_send_thread = worker.kv_recv_thread = None
        worker._step_load_keys = {"key"}
        worker.m_store = MagicMock()
        worker.m_store.batch_remove_lease.side_effect = [-1, 0]
        with self.assertRaisesRegex(RuntimeError, "lease cleanup failed"):
            worker.shutdown()
        worker.m_store.close.assert_not_called()
        self.assertEqual(worker._step_load_keys, {"key"})
        worker.shutdown()
        worker.m_store.close.assert_called_once()
        self.assertFalse(worker._step_load_keys)


class TestMooncakeWorkerSessionPreparation(unittest.TestCase):
    @staticmethod
    def _make_worker() -> KVPoolWorker:
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.dcp_size = 1
        worker.tp_rank = 0
        worker.pp_rank = 0
        worker.put_step = 1
        worker.dcp_size = 1
        worker.group_uses_align_state = [False]
        worker.block_size = 16
        worker.grouped_block_size = [16]
        worker.num_layers = 1
        worker.hash_block_size = 16
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.metadata = [KeyMetadata("model", 0, 0, 0)]
        worker.backend_name = "mooncake"
        worker.layerwise_backend = mooncake_backend.MooncakeBackend
        worker.layerwise_keys = worker.layerwise_backend.bind_layerwise_keys(
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)),
            kv_cache_config=None,
            model_name=worker.model_name,
            use_hybrid=False,
            grouped_block_size=worker.grouped_block_size,
        )
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.independent_layers = []
        worker.page_size_bytes = 60
        worker.group_block_len = {0: [10, 20, 30]}
        worker.layerwise_max_transfer_blocks = 0
        worker.use_eagle = False
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._layerwise_session_tracker = LayerwiseSessionTracker()
        worker.m_store = MagicMock()
        return worker

    def test_put_start_uses_full_current_layout_size_and_skips_hits(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[1, 2],
            block_hashes=[b"h0", b"h1"],
            can_save=True,
            load_spec=LoadSpec(0, 16, can_load=True),
        )

        worker._prepare_layerwise_put_session(request)

        worker.m_store.batch_put_start.assert_called_once_with(["model@6831@0"], [60])
        self.assertEqual(request.save_key_block_offset, 1)
        self.assertEqual(request.save_block_keys, ["model@6831@0"])

    def test_full_remote_hit_loads_last_block_even_when_vllm_keeps_one_token(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            block_ids=[1, 2, 3, 4],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            load_spec=LoadSpec(0, 63, can_load=True, kvpool_store_skip_tokens=64),
        )

        slots = worker._prepare_layerwise_get_session(request)

        self.assertEqual(len(slots), 4)
        self.assertEqual(request.load_block_keys[-1], "model@6833@0")
        self.assertIsNone(request.load_last_block_key)

    def test_next_chunk_reloads_committed_prefix_without_new_load_spec(self):
        worker = self._make_worker()
        worker._layerwise_session_tracker.register_put_keys(
            "r1",
            [("model@6830@0", 0)],
        )
        worker._layerwise_session_tracker.commit_put_keys(["model@6830@0"])
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0", b"h1"],
            load_spec=None,
            is_last_chunk=False,
        )

        slots = worker._prepare_layerwise_get_session(request)
        worker.layer_load_tasks = [[]]
        worker._process_load_for_layer_batch([request], 0)

        self.assertEqual(slots, [("model@6830@0", 10, 0)])
        self.assertEqual(request.load_block_keys, ["model@6830@0"])
        self.assertEqual(len(worker.layer_load_tasks[0]), 1)
        block_range = worker.layer_load_tasks[0][0].block_ranges[0]
        self.assertEqual((block_range.start_block, block_range.end_block), (0, 1))

    def test_hashless_boundary_key_uses_the_matching_block_slot(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(0, 32, can_load=True),
        )

        slots = worker._prepare_layerwise_get_session(request)

        self.assertEqual(
            request.load_block_keys,
            ["model@6830@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(slots[-1], ("model@r1_lastblock@0", 11, 1))


class TestLayerwiseSessionTracker(unittest.TestCase):
    def test_commit_promotes_shared_put_key_to_every_request_owner(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("shared", 0)])
        tracker.register_put_keys("r2", [("shared", 1)])

        tracker.commit_put_keys(["shared"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("shared", 0)])
        self.assertEqual(tracker.prepare_load_entries("r2", []), [("shared", 1)])

    def test_complete_key_replaces_partial_key_for_the_same_block(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("partial", 1)])
        tracker.commit_put_keys(["partial"])
        tracker.register_put_keys("r1", [("complete", 1)])
        tracker.commit_put_keys(["complete"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("complete", 1)])

    def test_shared_get_ends_only_after_the_last_owner_releases_it(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.prepare_load_entries("r2", [("shared", 0)])
        tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

        self.assertEqual(tracker.release_terminal({"r1"}), [])
        self.assertEqual(tracker.release_terminal({"r2"}), ["shared"])
        self.assertEqual(tracker.release_terminal({"r2"}), [])

    def test_failed_renewal_retains_desired_keys_for_retry(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.register_put_keys("r1", [("pending", 1)])
        tracker.record_get_result("shared", {"r1"}, succeeded=True)

        tracker.record_get_result("shared", {"r1"}, succeeded=False)
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.release_for_retry({"r1"}), [])
        self.assertEqual(
            tracker.prepare_load_entries("r1", []),
            [("shared", 0), ("pending", 1)],
        )

    def test_failed_get_attempt_preserves_unrelated_shared_owner(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("old-owner", [("shared", 0)])
        tracker.prepare_load_entries(
            "new-owner",
            [("shared", 0), ("new-key", 1)],
        )
        tracker.record_get_result(
            "shared",
            {"old-owner", "new-owner"},
            succeeded=True,
        )

        keys_to_end = tracker.release_failed_get_attempts(
            {
                "shared": {"new-owner"},
                "new-key": {"new-owner"},
            }
        )

        self.assertEqual(keys_to_end, ["new-key"])
        self.assertEqual(tracker.release_terminal({"old-owner"}), ["shared"])
        self.assertEqual(
            tracker.prepare_load_entries("new-owner", []),
            [("shared", 0), ("new-key", 1)],
        )

    def test_terminal_request_loses_pending_put_ownership(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("pending", 0)])

        tracker.release_terminal({"r1"})
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_chunk_commit_retry_and_terminal_cleanup(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("k0", 0)])
        tracker.commit_put_keys(["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_for_retry({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_terminal({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])


if __name__ == "__main__":
    unittest.main()
