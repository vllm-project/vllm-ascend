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
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on


def make_token_database() -> ChunkedTokenDatabase:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [1000, 2000, 3000]},
        {0: [10, 20, 30]},
        {0: [100, 200, 300]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 2, 3]},
    )
    return database


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

        slots = worker._prepare_mooncake_get_session(request)

        self.assertEqual(
            request.load_block_keys,
            ["model@6830@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(slots[-1], ("model@r1_lastblock@0", 11, 1))


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


if __name__ == "__main__":
    unittest.main()
