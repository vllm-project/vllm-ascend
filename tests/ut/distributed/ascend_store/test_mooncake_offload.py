# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-level scratch reuse, chunk snapshots, batching and failure fences."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

from tests.ut.distributed.ascend_store.test_mooncake_hybrid import MemoryRangeStore, cpu_cache
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_offload import (
    _load_entries,
    snapshot_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import _LayerRevokeTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheTopology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)


class TestMooncakeOffload(unittest.TestCase):
    def make_ring(self, buffers=1, hybrid=False, block_size=16, num_blocks=9, native_mtp=False):
        names = [f"model.layers.{layer}.kv" for layer in range(4)]
        spec = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype="uint8")
        if native_mtp:
            names.append("model.layers.4.kv")
        config = SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=[KVCacheGroupSpec(names, spec)])
        if native_mtp:
            config.layerwise_draft_layer_names = (names[-1],)
            config.layerwise_cache_topology = LayerwiseCacheTopology(
                main_packs=tuple((name,) for name in names[:-1]),
                group_ids=tuple((name, 0) for name in names),
                draft_layer_names=(names[-1],),
            )
        if hybrid:
            recurrent = MambaSpec(
                block_size=block_size, shapes=((block_size * 2,),), dtypes=("uint8",), mamba_cache_mode="align"
            )
            config.kv_cache_groups = [KVCacheGroupSpec(names[::2], spec), KVCacheGroupSpec(names[1::2], recurrent)]
            config.layerwise_cache_topology = LayerwiseCacheTopology(
                main_packs=((names[0], names[1]), (names[2], names[3])),
                group_ids=tuple((name, layer % 2) for layer, name in enumerate(names)),
            )
        worker = make_worker(
            self,
            num_layers=4,
            num_hidden_layers=4,
            kv_cache_config=config,
            use_layerwise=True,
            extra_config={
                "backend": "mooncake",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": buffers,
                "layerwise_independent_layers": [],
                "layerwise_prefetch_layers": 2,
            },
        )
        worker.vllm_config.speculative_config = None
        worker.m_store = MemoryRangeStore()
        arrays = [np.zeros((num_blocks, block_size * 2), dtype=np.uint8) for _ in range(buffers)]
        if native_mtp:
            arrays.append(np.zeros((num_blocks, block_size * 2), dtype=np.uint8))
            worker.vllm_config.speculative_config = SimpleNamespace(method="deepseek_mtp", num_speculative_tokens=3)
        if hybrid:
            worker.cache_coordinator = object()
            worker.token_database.store_mask = MagicMock(return_value=(None, None))
            worker.token_database.load_mask = MagicMock(return_value=(None, None))
        with (
            patch.object(KVTransferThread, "start", lambda lane: lane.ready_event.set()),
            patch.object(worker, "_align_kv_ptrs"),
        ):
            worker.register_kv_caches(
                {
                    name: (cpu_cache(arrays[-1] if native_mtp and layer == 4 else arrays[layer % buffers]),)
                    for layer, name in enumerate(names)
                }
            )
        jobs, errors = [], []

        def dispatch(lane, task):
            KVTransferThread.add_request(lane, task)

            def run():
                try:
                    with lane.test_lock:
                        lane._handle_request(lane.request_queue.get())
                except Exception as exc:
                    errors.append(exc)
                    lane._fatal_error = exc

            job = threading.Thread(target=run, daemon=True)
            jobs.append(job)
            job.start()

        for lane in (worker.kv_send_thread, worker.kv_recv_thread):
            lane.test_lock = threading.Lock()
            lane.add_request = lambda task, lane=lane: dispatch(lane, task)
        worker.kv_send_thread.add_revoke_request = lambda keys: dispatch(
            worker.kv_send_thread, _LayerRevokeTask(tuple(keys))
        )

        def finish():
            for job in jobs:
                job.join(timeout=3)
                self.assertFalse(job.is_alive(), "Scratch reuse deadlocked a transfer lane")
            worker._release_layerwise_requests_terminal(set(worker._current_layerwise_request_ids))

        self.addCleanup(finish)
        return worker, arrays, errors

    @staticmethod
    def request(end, *, previous=0, snapshot=None, incarnation="incarnation-1", last=False, block_size=16, blocks=4):
        request = ReqMeta(
            "request",
            token_len_chunk=end // block_size * block_size,
            save_start_token=previous // block_size * block_size,
            save_end_token=end // block_size * block_size,
            target_token_len=end,
            block_ids=list(range(1, blocks + 1)),
            block_hashes=[f"hash-{index}".encode() for index in range(end // block_size)],
            can_save=True,
            is_last_chunk=last,
            load_spec=LoadSpec(
                previous,
                previous,
                can_load=True,
                layerwise_previous_step=True,
                layerwise_snapshot_end_token=snapshot,
                layerwise_snapshot_start_block=previous // block_size if snapshot else None,
            )
            if previous
            else None,
        )
        request.layerwise_request_id = incarnation
        return request

    def run_step(self, worker, arrays, request, previous=0):
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(request)
        worker.prepare_layerwise_step(metadata)
        for layer in range(worker.num_layers):
            worker.wait_for_layer_load()
            rows = arrays[layer % len(arrays)][1:5].reshape(-1)
            if previous:
                np.testing.assert_array_equal(rows[: previous * 2], layer + 11)
            with attention_transfer_window():
                rows[previous * 2 : request.target_token_len * 2] = layer + 11
            worker.save_kv_layer(metadata)
        self.assertFalse(worker.m_store.open_reads)

    def test_real_shared_buffer_roundtrip_across_partial_and_aligned_chunks(self):
        for buffers in (1, 2):
            with self.subTest(buffers=buffers):
                worker, arrays, errors = self.make_ring(buffers)
                self.run_step(worker, arrays, self.request(20))
                self.run_step(worker, arrays, self.request(32, previous=20, snapshot=20), 20)
                self.run_step(worker, arrays, self.request(48, previous=32, last=True), 32)
                self.assertFalse(errors)

    def test_metadata_selection_is_built_once_per_group(self):
        worker, arrays, _ = self.make_ring()
        with (
            patch.object(worker, "_process_save_for_layer_batch", wraps=worker._process_save_for_layer_batch) as save,
            patch.object(worker, "_process_load_for_layer_batch", wraps=worker._process_load_for_layer_batch) as load,
        ):
            self.run_step(worker, arrays, self.request(32))
        self.assertEqual(save.call_count, 1)
        self.assertEqual(load.call_count, 1)

    def test_hybrid_main_pack_reuses_attention_and_recurrent_groups(self):
        worker, arrays, errors = self.make_ring(hybrid=True)
        self.assertEqual(worker.prefetch_layer_map, {2: 0, 3: 1})
        for end in (16, 32, 48):
            previous = end - 16
            request = self.request(end, previous=previous, last=end == 48)
            request.block_ids_by_group = [[1, 2, 3, 4], [5, 6, 7, 8]]
            request.block_ids_by_group_np = [np.asarray(ids) for ids in request.block_ids_by_group]
            request.num_prompt_tokens = 48
            meta = AscendConnectorMetadata(set())
            meta.add_request(request)
            worker.prepare_layerwise_step(meta)
            for layer in range(4):
                worker.wait_for_layer_load()
                group = layer % 2
                ids = request.block_ids_by_group[group]
                if previous:
                    check_ids = ids[: previous // 16] if group == 0 else [ids[previous // 16 - 1]]
                    np.testing.assert_array_equal(arrays[0][check_ids], layer + 11)
                with attention_transfer_window():
                    arrays[0][ids[end // 16 - 1]] = layer + 11
                worker.save_kv_layer(meta)
            self.assertFalse(worker.m_store.open_reads)
        self.assertFalse(errors)

    def test_batch_put_start_is_not_per_layer_or_request(self):
        worker, _, _ = self.make_ring()
        first, second = self.request(20), self.request(20, incarnation="incarnation-2")
        second.req_id = "second"
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(first)
        metadata.add_request(second)
        with patch.object(worker.m_store, "batch_put_start", wraps=worker.m_store.batch_put_start) as start:
            worker.prepare_layerwise_step(metadata)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(len(start.call_args.args[0]), 3)  # common prefix + two private tails

    def test_missing_required_prefix_aborts_before_computation(self):
        worker, _, _ = self.make_ring()
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(self.request(32, previous=16))
        with self.assertRaisesRegex(RuntimeError, "refusing incomplete"):
            worker.prepare_layerwise_step(metadata)
        self.assertFalse(worker.m_store.open_reads)

    def test_failed_put_start_does_not_enable_reuse(self):
        worker, _, _ = self.make_ring()
        worker.m_store.batch_put_start = lambda keys, sizes: [-1] * len(keys)
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(self.request(16))
        with self.assertRaisesRegex(RuntimeError, "required PUT session"):
            worker.prepare_layerwise_step(metadata)

    def test_failed_range_put_stops_before_next_owner(self):
        worker, arrays, errors = self.make_ring()
        worker.m_store.batch_copy_put = lambda keys, *args: [-1] * len(keys)
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            self.run_step(worker, arrays, self.request(16))
        self.assertTrue(errors)
        self.assertFalse(worker.m_store.complete)

    def test_request_incarnation_prevents_snapshot_aliasing(self):
        worker, _, _ = self.make_ring()
        self.assertNotEqual(
            snapshot_key(worker, self.request(20), 0, 1, 20),
            snapshot_key(worker, self.request(20, incarnation="new"), 0, 1, 20),
        )

    def test_resident_recurrent_state_is_not_restored_from_previous_step(self):
        worker = SimpleNamespace(group_uses_align_state=[True], _group_reuses_buffer=lambda group: False)
        self.assertEqual(_load_entries(worker, self.request(32, previous=16), 0, [], None), [])

    def test_empty_step_has_no_sessions_or_tasks(self):
        worker, _, _ = self.make_ring()
        worker.prepare_layerwise_step(AscendConnectorMetadata(set()))
        self.assertFalse(worker.m_store.objects)
        self.assertTrue(all(not tasks for tasks in worker.layer_save_tasks + worker.layer_load_tasks))

    def test_failed_commit_is_reported_at_step_boundary(self):
        worker, arrays, errors = self.make_ring()
        worker.m_store.batch_commit = lambda keys: [-1] * len(keys)
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            self.run_step(worker, arrays, self.request(16))
        self.assertTrue(errors)
        self.assertFalse(worker.m_store.complete)

    def test_resident_final_layer_ends_load_state_at_last_actual_transfer(self):
        worker, arrays, _ = self.make_ring()
        self.run_step(worker, arrays, self.request(16))
        worker.independent_layers = [3]
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(self.request(32, previous=16))
        worker.prepare_layerwise_step(metadata)
        self.assertFalse(worker.layer_load_tasks[3])
        self.assertTrue(worker.layer_load_tasks[2][0].final_group_layer)

    def test_kda_exact_partial_endpoint_roundtrip_at_reported_token_lengths(self):
        size = 1152
        count = 16
        worker, arrays, errors = self.make_ring(hybrid=True, block_size=size, num_blocks=2 * count + 1)
        previous = 0
        # Reproduce the reported aligned hit followed by the unaligned prompt
        # endpoint, then restore that endpoint before a subsequent step.
        endpoints = (13824, 16395, *range(16396, 16443), 17280, 17281)
        for end in endpoints:
            request = self.request(
                end,
                previous=previous,
                snapshot=previous if previous % size else None,
                block_size=size,
                blocks=count,
                last=end >= 16395,
            )
            request.num_prompt_tokens = 16395
            request.block_ids_by_group = [list(range(1, count + 1)), list(range(count + 1, 2 * count + 1))]
            request.block_ids_by_group_np = [np.asarray(ids) for ids in request.block_ids_by_group]
            metadata = AscendConnectorMetadata(set())
            metadata.add_request(request)
            worker.prepare_layerwise_step(metadata)
            for layer in range(4):
                worker.wait_for_layer_load()
                group = layer % 2
                ids = request.block_ids_by_group[group]
                if previous:
                    selected = ids[: (previous + size - 1) // size] if group == 0 else [ids[(previous - 1) // size]]
                    np.testing.assert_array_equal(arrays[0][selected], layer + previous % 100 if group else layer + 11)
                with attention_transfer_window():
                    selected = ids[: (end + size - 1) // size] if group == 0 else [ids[(end - 1) // size]]
                    arrays[0][selected] = layer + end % 100 if group else layer + 11
                worker.save_kv_layer(metadata)
            self.assertFalse(worker.m_store.open_reads)
            previous = end
        self.assertFalse(errors)

    def test_unaligned_recurrent_restore_never_uses_a_later_speculative_endpoint(self):
        worker, _, _ = self.make_ring(hybrid=True)
        request = self.request(32, previous=20, snapshot=24)
        request.block_ids_by_group = [[1, 2, 3, 4], [5, 6, 7, 8]]
        with self.assertRaisesRegex(ValueError, "exact saved snapshot endpoint"):
            _load_entries(worker, request, 1, [b"hash-0"], None)

    def test_native_mtp_prefix_is_transferred_and_committed_after_its_hook(self):
        worker, arrays, errors = self.make_ring(native_mtp=True)
        self.assertEqual(worker.num_layers, 5)
        self.assertIn(4, worker.independent_layers)
        self.assertFalse(worker._dspark_draft_layer_names)
        for end, previous, external in ((16, 0, False), (20, 16, True), (32, 20, False)):
            request = self.request(end, previous=previous, snapshot=20 if previous == 20 else None)
            if external:
                request.load_spec.layerwise_previous_step = False
                request.load_spec.vllm_cached_tokens = 0
                arrays[-1].fill(0)
            metadata = AscendConnectorMetadata(set())
            metadata.add_request(request)
            worker.prepare_layerwise_step(metadata)
            if previous and not external:
                self.assertFalse(worker.layer_load_tasks[4])
            for layer in range(5):
                worker.wait_for_layer_load()
                rows = arrays[-1 if layer == 4 else 0][1:5].reshape(-1)
                if previous:
                    np.testing.assert_array_equal(rows[: previous * 2], layer + 11)
                with attention_transfer_window():
                    rows[previous * 2 : end * 2] = layer + 11
                worker.save_kv_layer(metadata)
                if not previous and layer == 3:
                    worker.kv_send_thread.request_queue.join()
                    self.assertFalse(worker.m_store.complete, "A target-only object omitted its native MTP payload")
            self.assertTrue(worker.m_store.complete)
        self.assertFalse(errors)

    def test_recurrent_snapshot_identity_survives_later_hash_publication(self):
        worker, _, _ = self.make_ring(hybrid=True)
        request = self.request(32, previous=16, snapshot=16)
        request.block_ids_by_group = [[1, 2, 3, 4], [5, 6, 7, 8]]
        # The preceding aligned page had no content hash at save time.
        entries = _load_entries(worker, request, 1, [b"newly-published-hash"], None)
        self.assertEqual(entries, [(snapshot_key(worker, request, 1, 0, 16), 0)])

    def test_scheduler_retains_exact_snapshot_identity_for_single_token_decode(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import (
            LayerwiseKVPoolScheduler,
        )

        tracker = RequestTracker(
            "request",
            token_len=16395,
            num_saved_tokens=13824,
            allocated_block_ids=list(range(1, 17)),
            num_prompt_tokens=16395,
        )
        tracker.layerwise_request_id = "incarnation"
        hashes = [f"hash-{i}".encode() for i in range(14)]
        ReqMeta.from_request_tracker(tracker, 1152, block_hashes=hashes, save_partial_block=True)
        self.assertEqual(tracker.last_layerwise_snapshot_end_token, 16395)
        scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
        scheduler.layerwise_offload = True
        load = scheduler._running_load_spec(tracker, 16395)
        tracker.token_len = 16396
        next_step = ReqMeta.from_request_tracker(
            tracker, 1152, block_hashes=hashes, load_spec=load, save_partial_block=True
        )
        self.assertTrue(next_step.load_spec.layerwise_previous_step)
        self.assertEqual(next_step.load_spec.layerwise_snapshot_end_token, 16395)
        self.assertEqual(next_step.target_token_len, 16396)
        self.assertEqual(tracker.last_layerwise_snapshot_end_token, 16396)
