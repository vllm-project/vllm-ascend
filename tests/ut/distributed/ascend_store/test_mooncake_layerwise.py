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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _build_range_debug_payload,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerRangeReqMeta,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
    make_layerwise_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

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


class TestMooncakeLayerBatchBuilder(unittest.TestCase):
    def test_key_major_ranges_use_group_local_layer_index(self):
        database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [16], None)
        database.set_group_buffers(
            {1: [4000]},
            {1: [40]},
            {1: [100]},
            group_num_layers={1: 1},
            group_layer_cache_entry_offsets={1: [0, 1]},
        )
        request = ReqMeta("r1", block_ids=[2], block_hashes=[])
        request.save_block_keys = ["key"]
        task = LayerTransferTask(
            layer_id=17,
            layer_idx_in_group=0,
            block_ranges=[LayerBlockRange(request, 0, 1)],
            use_key_major_ranges=True,
        )

        result = LayerBatchBuilder(database, page_size_bytes=40, num_layers=1, group_id=1).build(task)

        self.assertIsInstance(result, LayerRangeReqMeta)
        assert isinstance(result, LayerRangeReqMeta)
        self.assertEqual(result.layer_id, 17)
        self.assertEqual(result.all_buffers, [[4200]])

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
    def test_final_layer_commits_after_all_ranges(self):
        store = MagicMock()
        # Range APIs may return the positive number of bytes moved on success.
        store.batch_copy_put.return_value = [30]
        store.batch_commit.return_value = [0]
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("key", 0)])
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
            put_started_keys={"key"},
            session_tracker=tracker,
        )
        request = ReqMeta("r1", block_ids=[2], block_hashes=[], is_last_chunk=True)
        request.save_block_keys = ["key"]

        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=layer_id,
                block_ranges=[LayerBlockRange(request, 0, 1)],
                shared_block_data=builder.build_shared(
                    LayerTransferTask(
                        layer_id=layer_id,
                        block_ranges=[LayerBlockRange(request, 0, 1)],
                        use_key_major_ranges=True,
                    )
                ),
                use_key_major_ranges=True,
            )
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            thread._handle_request([task])

        self.assertEqual(store.batch_copy_put.call_count, 2)
        store.batch_commit.assert_called_once_with(["key"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("key", 0)])


class TestMooncakeWorkerSessionPreparation(unittest.TestCase):
    @staticmethod
    def _make_worker() -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.tp_rank = 0
        worker.put_step = 1
        worker.block_size = 16
        worker.grouped_block_size = [16]
        worker.hash_block_size = 16
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.backend_name = "mooncake"
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.independent_layers = []
        worker.page_size_bytes = 60
        worker.group_block_len = {0: [10, 20, 30]}
        worker.layerwise_max_transfer_blocks = 0
        worker.use_eagle = False
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._mooncake_session_tracker = MooncakeSessionTracker()
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

        worker._prepare_mooncake_put_session(request)

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

        slots = worker._prepare_mooncake_get_session(request)

        self.assertEqual(len(slots), 4)
        self.assertEqual(request.load_block_keys[-1], "model@6833@0")
        self.assertIsNone(request.load_last_block_key)

    def test_next_chunk_reloads_committed_prefix_without_new_load_spec(self):
        worker = self._make_worker()
        worker._mooncake_session_tracker.register_put_keys(
            "r1",
            [("model@6830@0", 0)],
        )
        worker._mooncake_session_tracker.commit_put_keys(["model@6830@0"])
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0", b"h1"],
            load_spec=None,
            is_last_chunk=False,
        )

        slots = worker._prepare_mooncake_get_session(request)
        worker.layer_load_tasks = [[]]
        worker._process_load_for_layer_batch([request], 0)

        self.assertEqual(slots, [("model@6830@0", 10, (0, 0))])
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

        slots = worker._prepare_mooncake_get_session(request)

        self.assertEqual(
            request.load_block_keys,
            ["model@6830@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(slots[-1], ("model@r1_lastblock@0", 11, (0, 1)))


class TestMooncakeSessionTracker(unittest.TestCase):
    def test_commit_promotes_shared_put_key_to_every_request_owner(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("shared", 0)])
        tracker.register_put_keys("r2", [("shared", 1)])

        tracker.commit_put_keys(["shared"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("shared", 0)])
        self.assertEqual(tracker.prepare_load_entries("r2", []), [("shared", 1)])

    def test_complete_key_replaces_partial_key_for_the_same_block(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("partial", 1)])
        tracker.commit_put_keys(["partial"])
        tracker.register_put_keys("r1", [("complete", 1)])
        tracker.commit_put_keys(["complete"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("complete", 1)])

    def test_shared_get_ends_only_after_the_last_owner_releases_it(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.prepare_load_entries("r2", [("shared", 0)])
        tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

        self.assertEqual(tracker.release_terminal({"r1"}), [])
        self.assertEqual(tracker.release_terminal({"r2"}), ["shared"])
        self.assertEqual(tracker.release_terminal({"r2"}), [])

    def test_failed_renewal_retains_desired_keys_for_retry(self):
        tracker = MooncakeSessionTracker()
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
        tracker = MooncakeSessionTracker()
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
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("pending", 0)])

        tracker.release_terminal({"r1"})
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_chunk_commit_retry_and_terminal_cleanup(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("k0", 0)])
        tracker.commit_put_keys(["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_for_retry({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_terminal({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])


class TestMooncakeHybridLayerwise(unittest.TestCase):
    """Multi-group (hybrid) Mooncake layerwise behavior."""

    @staticmethod
    def _make_hybrid_worker() -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.tp_rank = 0
        worker.put_step = 1
        worker.block_size = 16
        worker.grouped_block_size = [16, 32]
        worker.hash_block_size = 16
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.backend_name = "mooncake"
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.independent_layers = []
        worker.page_size_bytes = 60
        worker.group_block_len = {0: [10, 20, 30], 1: [40, 50]}
        worker.layerwise_max_transfer_blocks = 0
        worker.use_eagle = False
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._mooncake_session_tracker = MooncakeSessionTracker()
        worker.m_store = MagicMock()
        return worker

    def test_block_key_embeds_group_id_only_for_multi_group(self):
        self.assertEqual(make_layerwise_block_key("m", "h", 0), "m@h@0")
        self.assertEqual(make_layerwise_block_key("m", "h", 0, group_id=0, num_groups=2), "m@g0@h@0")
        self.assertEqual(make_layerwise_block_key("m", "h", 1, group_id=1, num_groups=2), "m@g1@h@1")

    def test_put_session_generates_per_group_keys_and_object_sizes(self):
        worker = self._make_hybrid_worker()
        worker.m_store.batch_put_start.side_effect = lambda keys, sizes: [0] * len(keys)
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[[1, 2], [3]],
            block_hashes=[b"h0", b"h1"],
            can_save=True,
        )

        worker._prepare_mooncake_put_session(request)

        g0_hashes = get_block_hashes(request.block_hashes, 16, 16)
        g1_hashes = get_block_hashes(request.block_hashes, 32, 16)
        expected_g0 = [make_layerwise_block_key("model", block_hash_to_str(h), 0, 0, 2) for h in g0_hashes]
        expected_g1 = [make_layerwise_block_key("model", block_hash_to_str(h), 0, 1, 2) for h in g1_hashes]
        self.assertEqual(request.save_block_keys_by_group[0], expected_g0)
        self.assertEqual(request.save_block_keys_by_group[1], expected_g1)
        # One put_start call per group, each sized with that group's page bytes.
        self.assertEqual(worker.m_store.batch_put_start.call_count, 2)
        object_sizes = {call.args[1][0] for call in worker.m_store.batch_put_start.call_args_list}
        self.assertEqual(object_sizes, {60, 90})
        # Flat group-0 mirror stays populated for legacy consumers.
        self.assertEqual(request.save_block_keys, expected_g0)

    def test_get_session_builds_per_group_keys_and_coords(self):
        worker = self._make_hybrid_worker()
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[[1, 2], [3]],
            block_hashes=[b"h0", b"h1"],
            load_spec=LoadSpec(0, 32, can_load=True),
        )

        slots = worker._prepare_mooncake_get_session(request)

        g0_hashes = get_block_hashes(request.block_hashes, 16, 16)
        g1_hashes = get_block_hashes(request.block_hashes, 32, 16)
        expected_g0 = [make_layerwise_block_key("model", block_hash_to_str(h), 0, 0, 2) for h in g0_hashes]
        expected_g1 = [make_layerwise_block_key("model", block_hash_to_str(h), 0, 1, 2) for h in g1_hashes]
        self.assertEqual(request.load_block_keys_by_group[0], expected_g0)
        self.assertEqual(request.load_block_keys_by_group[1], expected_g1)
        self.assertEqual(slots[0], (expected_g0[0], 1, (0, 0)))
        self.assertIn((expected_g1[0], 3, (1, 0)), slots)

    def test_session_tracker_keeps_group_coords_separate(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("g0-key", (0, 5)), ("g1-key", (1, 5))])

        tracker.commit_put_keys(["g0-key", "g1-key"])

        entries = dict(tracker.prepare_load_entries("r1", []))
        self.assertEqual(entries["g0-key"], (0, 5))
        self.assertEqual(entries["g1-key"], (1, 5))

    def test_multi_group_load_failure_aborts_without_invalid_block_report(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._invalid_block_ids = set()
        thread._invalid_block_ids_lock = threading.Lock()
        thread._load_abort_event = threading.Event()
        thread.num_kv_cache_groups = 2

        thread._record_invalid_block_ids({7, 8})

        self.assertEqual(thread._invalid_block_ids, set())
        self.assertTrue(thread._load_abort_event.is_set())

    def test_single_group_load_failure_reports_invalid_blocks(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._invalid_block_ids = set()
        thread._invalid_block_ids_lock = threading.Lock()
        thread._load_abort_event = threading.Event()
        thread.num_kv_cache_groups = 1

        thread._record_invalid_block_ids({7})

        self.assertEqual(thread._invalid_block_ids, {7})
        self.assertFalse(thread._load_abort_event.is_set())

    @staticmethod
    def _make_group_builder(group_id: int, block_len: int, base_addr: int) -> tuple[LayerBatchBuilder, object]:
        database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [16], None)
        database.set_group_buffers(
            {group_id: [base_addr]},
            {group_id: [block_len]},
            {group_id: [block_len]},
            group_num_layers={group_id: 1},
            group_layer_cache_entry_offsets={group_id: [0, 1]},
        )
        return LayerBatchBuilder(database, page_size_bytes=block_len, num_layers=1, group_id=group_id), database

    def test_multi_group_tasks_at_one_layer_share_commit(self):
        store = MagicMock()
        store.batch_copy_put.side_effect = lambda keys, buffers, sizes, offsets: [30] * len(keys)
        store.batch_commit.side_effect = lambda keys: [0] * len(keys)
        tracker = MooncakeSessionTracker()
        builder_g0, database_g0 = self._make_group_builder(0, 10, 1000)
        builder_g1, _ = self._make_group_builder(1, 40, 4000)
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=database_g0,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=50,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=[threading.Event(), threading.Event()],
            sync_save_events=[MagicMock(), MagicMock()],
            group_builders=[builder_g0, builder_g1],
            put_started_keys=set(),
            session_tracker=tracker,
        )

        def make_task(group_id: int, key: str, block_id: int, layer_id: int) -> LayerTransferTask:
            request = ReqMeta(f"r{group_id}", block_ids=[[block_id]], block_hashes=[])
            request.save_block_keys_by_group = [[key] for _ in range(2)]
            request.save_key_block_offset_by_group = [0, 0]
            request.save_last_block_key_by_group = [None, None]
            return LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=0,
                group_id=group_id,
                block_ranges=[LayerBlockRange(request, 0, 1)],
                use_key_major_ranges=True,
            )

        # Two groups contribute tasks for the same (final) physical layer.
        tasks = [make_task(0, "g0-key", 2, 1), make_task(1, "g1-key", 5, 1)]
        for task in tasks:
            builder = builder_g0 if task.group_id == 0 else builder_g1
            task.shared_block_data = builder.build_shared(task, is_save=True)
            thread.add_stored_request(task.block_ranges[0].request.req_id)

        thread.request_queue.put(tasks)
        thread._handle_request(tasks)

        self.assertEqual(store.batch_copy_put.call_count, 2)
        copied_keys = {call.args[0][0] for call in store.batch_copy_put.call_args_list}
        self.assertEqual(copied_keys, {"g0-key", "g1-key"})
        store.batch_commit.assert_called_once()
        committed_keys = set(store.batch_commit.call_args.args[0])
        self.assertEqual(committed_keys, {"g0-key", "g1-key"})

    def test_late_joining_group_registers_keys_and_commits(self):
        """Group 1's first task appears at a later layer than group 0's."""
        store = MagicMock()
        store.batch_copy_put.side_effect = lambda keys, buffers, sizes, offsets: [30] * len(keys)
        store.batch_commit.side_effect = lambda keys: [0] * len(keys)
        tracker = MooncakeSessionTracker()
        builder_g0, database_g0 = self._make_group_builder(0, 10, 1000)
        builder_g1, _ = self._make_group_builder(1, 40, 4000)
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=database_g0,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=50,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=[threading.Event(), threading.Event()],
            sync_save_events=[MagicMock(), MagicMock()],
            group_builders=[builder_g0, builder_g1],
            put_started_keys=set(),
            session_tracker=tracker,
        )

        def make_task(group_id: int, key: str, block_id: int, layer_id: int) -> LayerTransferTask:
            request = ReqMeta(f"r{group_id}", block_ids=[[block_id]], block_hashes=[])
            request.save_block_keys_by_group = [[key] for _ in range(2)]
            request.save_key_block_offset_by_group = [0, 0]
            request.save_last_block_key_by_group = [None, None]
            return LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=0,
                group_id=group_id,
                block_ranges=[LayerBlockRange(request, 0, 1)],
                use_key_major_ranges=True,
            )

        # Layer 0: only group 0 contributes.
        layer0_tasks = [make_task(0, "g0-key", 2, 0)]
        # Layer 1 (final): group 0 plus the late-joining group 1.
        layer1_tasks = [make_task(0, "g0-key", 2, 1), make_task(1, "g1-key", 5, 1)]
        for tasks in (layer0_tasks, layer1_tasks):
            for task in tasks:
                builder = builder_g0 if task.group_id == 0 else builder_g1
                task.shared_block_data = builder.build_shared(task, is_save=True)
                thread.add_stored_request(task.block_ranges[0].request.req_id)
            thread.request_queue.put(tasks)
            thread._handle_request(tasks)

        copied_keys = [call.args[0][0] for call in store.batch_copy_put.call_args_list]
        self.assertEqual(copied_keys, ["g0-key", "g0-key", "g1-key"])
        committed_keys = set(store.batch_commit.call_args.args[0])
        self.assertEqual(committed_keys, {"g0-key", "g1-key"})


if __name__ == "__main__":
    unittest.main()
