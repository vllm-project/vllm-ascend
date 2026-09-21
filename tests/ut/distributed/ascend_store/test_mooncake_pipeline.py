# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise actual worker sessions/range I/O with distinct bytes on every stage."""

import threading
import unittest
from copy import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store.test_mooncake_hybrid import MemoryRangeStore, cpu_cache
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise as protocol
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread, _LayerRevokeTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestMooncakePipeline(unittest.TestCase):
    def make_stage(self, stage, store, *, hybrid=True, draft=False, tp_rank=0, mla=False, empty_tail=False):
        # Uneven PP partition; the last stage optionally owns an extra draft
        # layer that is not part of get_num_layers().
        layers = list(range(3)) if stage == 0 else list(range(3, 7 + int(draft)))
        names = [f"model.layers.{layer}.kv" for layer in layers]
        spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype="uint8")
        groups = [KVCacheGroupSpec(names, spec)]
        if hybrid:
            groups = [
                KVCacheGroupSpec([name for layer, name in zip(layers, names) if layer in (0, 2, 3, 6, 7)], spec),
                KVCacheGroupSpec(
                    [f"model.layers.{layer}.c4" for layer in layers if layer in (1, 2, 4, 5, 6, 7)]
                    + (["model.layers.6.c4_indexer"] if stage else []),
                    FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=1, dtype="uint8"),
                ),
                KVCacheGroupSpec(
                    [f"model.layers.{layer}.state" for layer in layers if layer in (0, 2, 3, 6, 7)],
                    SlidingWindowSpec(block_size=16, sliding_window=16, num_kv_heads=1, head_size=1, dtype="uint8"),
                ),
            ]
        if empty_tail:
            # The final layer owns only unreachable SWA state in this step.
            groups[0].layer_names = ["model.layers.3.kv", "model.layers.5.kv"]
            groups[1].layer_names = ["model.layers.4.c4", "model.layers.5.c4"]
            groups[2].layer_names = ["model.layers.6.state"]
        config = SimpleNamespace(num_blocks=10, kv_cache_groups=groups)
        with patch.object(KVPoolWorker, "_build_cache_coordinator", return_value=None):
            worker = make_worker(
                self,
                num_layers=(3, 4)[stage],
                num_hidden_layers=7,
                kv_cache_config=config,
                use_layerwise=True,
                pp_rank=stage,
                pp_partition=(3, 4),
                tp_size=2,
                tp_rank=tp_rank,
                num_kv_heads=1 if mla else 2,
                use_mla=mla,
            )
        worker.m_store = store
        arrays = {}
        # Noncontiguous pages, different widths and values for every cache,
        # layer, TP head and block. A wrong copy cannot pass by symmetry.
        for group_id, group in enumerate(groups):
            for index, name in enumerate(group.layer_names):
                width = 8 + group_id * 4 + index
                backing = np.arange(10 * (width + 7), dtype=np.uint8).reshape(10, width + 7)
                backing ^= np.uint8(stage * 67 + tp_rank * 23 + group_id * 11 + index)
                arrays[name] = backing[:, :width]
        with (
            patch.object(KVTransferThread, "start", lambda thread: thread.ready_event.set()),
            patch.object(worker, "_align_kv_ptrs"),
        ):
            worker.register_kv_caches({name: (cpu_cache(array),) for name, array in reversed(list(arrays.items()))})
        worker.cache_coordinator = object() if hybrid else None
        if hybrid:
            masks = ([True] * 4, [True] * 2, [False, False, False, not empty_tail])
            worker.token_database.store_mask = MagicMock(return_value=masks)
            worker.token_database.load_mask = MagicMock(return_value=masks)
        jobs, errors = [], []

        # Each lane uses the real production queue handler, including event
        # waits, group commits and session cleanup.
        def dispatch(thread, task):
            KVTransferThread.add_request(thread, task)

            def run():
                try:
                    with thread.test_lock:
                        thread._handle_request(thread.request_queue.get())
                except Exception as exc:
                    errors.append(exc)
                    thread._fatal_error = exc

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
                job.join(timeout=2)
                self.assertFalse(job.is_alive(), "transfer did not finish")
            self.assertFalse(errors)

        self.addCleanup(finish)
        return worker, arrays

    @staticmethod
    def run_step(worker, request):
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        for _ in range(worker.num_layers):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
            if request.can_save:
                worker.save_kv_layer(meta)

    def make_scheduler(self, worker, store):
        # Construct the real scheduler from stage 0's projected config, as vLLM
        # does; do not hand-copy the worker's layout ID into the scheduler.
        with (
            patch.object(KVPoolScheduler, "_build_cache_coordinator", return_value=None),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib"),
        ):
            scheduler = KVPoolScheduler(worker.vllm_config, use_layerwise=True, kv_cache_config=worker.kv_cache_config)
        store.batch_is_readable = lambda keys: [key in store.complete for key in keys]
        scheduler.store_scheduler = store
        return scheduler

    def test_uneven_stages_roundtrip_all_heads_and_groups(self):
        for hybrid, draft in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(hybrid=hybrid, draft=draft):
                store = MemoryRangeStore()
                workers = [
                    self.make_stage(stage, store, hybrid=hybrid, draft=draft, tp_rank=head)
                    for stage in range(2)
                    for head in range(2)
                ]
                scheduler = self.make_scheduler(workers[0][0], store)
                expected_keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
                self.assertEqual(len(set(expected_keys)), 4)
                original = []
                for index, (worker, arrays) in enumerate(workers):
                    self.assertEqual(worker.block_key_hybrid_layout, scheduler.block_key_hybrid_layout)
                    self.assertEqual(worker.mooncake_layerwise_namespace, scheduler.mooncake_layerwise_namespace)
                    self.assertEqual(worker.num_layers, (3 if worker.pp_rank == 0 else 4 + int(draft)))
                    self.assertEqual(worker.layerwise_key_layers, worker.num_layers)
                    for builder in worker.kv_send_thread.group_builders:
                        self.assertEqual(builder.layer_byte_offset, 0)
                        self.assertEqual(builder.page_size_bytes, sum(worker.group_block_len[builder.group_id]))
                    original.append({name: array.copy() for name, array in arrays.items()})
                    ids = [[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]] if hybrid else [[1, 2, 3, 4]]
                    request = ReqMeta(
                        "save",
                        token_len_chunk=64,
                        block_ids_by_group=ids,
                        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                        can_save=True,
                        is_last_chunk=True,
                    )
                    self.run_step(worker, request)
                    self.assertEqual(scheduler._query_layerwise_block_hits([expected_keys]), [index == 3])
                # Destroy every destination and remap block IDs; stale local
                # bytes and local-prefix-cache hits cannot hide a bad restore.
                for (worker, arrays), saved in zip(workers, original):
                    for array in arrays.values():
                        array.fill(0)
                    worker.kv_role = "kv_consumer"
                    ids = [[4, 5, 6, 7], [5, 6], [4, 5, 6, 7]] if hybrid else [[4, 5, 6, 7]]
                    request = ReqMeta(
                        "load",
                        token_len_chunk=64,
                        block_ids_by_group=ids,
                        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                        load_spec=LoadSpec(0, 64, can_load=True),
                        is_last_chunk=True,
                    )
                    self.run_step(worker, request)
                    for name, array in arrays.items():
                        src, dst = (
                            ([4], [7])
                            if "state" in name
                            else (([1, 2], [5, 6]) if "c4" in name else ([1, 2, 3, 4], [4, 5, 6, 7]))
                        )
                        np.testing.assert_array_equal(array[dst], saved[name][src])
                        untouched = [slot for slot in range(10) if slot not in dst]
                        self.assertFalse(array[untouched].any(), "null/unreachable blocks were overwritten")
                self.assertFalse(store.open_reads)
                self.doCleanups()

    def test_mla_replica_writes_only_once_per_stage(self):
        store = MemoryRangeStore()
        for stage in range(2):
            for head in range(2):
                worker, _ = self.make_stage(stage, store, mla=True, tp_rank=head)
                self.assertEqual(worker._is_layerwise_save_owner(), head == 0)
                self.assertEqual(worker.head_or_tp_rank, 0)
                req = ReqMeta(
                    "r",
                    token_len_chunk=64,
                    block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
                    block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                    can_save=True,
                    is_last_chunk=True,
                )
                self.run_step(worker, req)
        self.assertEqual(len(store.complete), 14)

    def test_nonzero_stage_uses_local_group_indices(self):
        worker, _ = self.make_stage(1, MemoryRangeStore())
        self.assertEqual(worker.pp_layer_offset, 3)
        self.assertEqual(worker._groups_for_layerwise_transfer(0), [(0, 0), (2, 0)])
        self.assertEqual(worker._groups_for_layerwise_transfer(1), [(1, 0)])
        self.assertEqual(worker._groups_for_layerwise_transfer(2), [(1, 1)])
        self.assertEqual(worker._groups_for_layerwise_transfer(3), [(0, 1), (1, 2), (2, 1)])

    def test_direct_builder_uses_group_layer_index(self):
        worker, arrays = self.make_stage(1, MemoryRangeStore())
        request = ReqMeta(
            "r",
            token_len_chunk=64,
            block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            can_save=True,
            is_last_chunk=True,
        )
        worker.process_layer_data([request])
        task = worker.layer_save_tasks[3][0]
        self.assertEqual((task.group_id, task.layer_id, task.layer_idx_in_group), (0, 3, 1))
        result = worker.kv_send_thread.group_builders[0].build(task)
        cache = arrays["model.layers.6.kv"]
        self.assertEqual(result.all_buffers[0], [cache.ctypes.data + cache.strides[0]])
        self.assertEqual(result.all_offsets[0], [arrays["model.layers.3.kv"].shape[1]])

    def test_failed_stage_group_remains_a_miss(self):
        store = MemoryRangeStore()
        stage0, _ = self.make_stage(0, store)
        stage1, _ = self.make_stage(1, store)
        scheduler = self.make_scheduler(stage0, store)
        # Focus on one saving head while retaining the real PP key namespace.
        scheduler.put_step = 2
        real_put = store.batch_copy_put

        def fail_one_group(keys, *args):
            if all("@pp_rank:1@group:1@" in key for key in keys):
                return [-1] * len(keys)
            return real_put(keys, *args)

        store.batch_copy_put = fail_one_group
        for worker in (stage0, stage1):
            self.run_step(
                worker,
                ReqMeta(
                    "r",
                    token_len_chunk=64,
                    block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
                    block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                    can_save=True,
                    is_last_chunk=True,
                ),
            )
        complete = scheduler._make_layerwise_hit_check_keys(0, b"h1".hex())
        incomplete = scheduler._make_layerwise_hit_check_keys(1, b"h1".hex())
        self.assertEqual(scheduler._query_layerwise_block_hits([complete, incomplete]), [True, False])
        self.assertFalse(any("@pp_rank:1@group:1@" in key for key in store.objects))

    def test_layout_identity_is_rank_independent_but_config_sensitive(self):
        worker, _ = self.make_stage(0, MemoryRangeStore())
        config = worker.vllm_config
        original = protocol.layerwise_topology_namespace(config)
        for rank in range(4):
            config.parallel_config.rank = rank
            self.assertEqual(original, protocol.layerwise_topology_namespace(config))
        config.cache_config.cache_dtype = "fp8"
        self.assertNotEqual(original, protocol.layerwise_topology_namespace(config))
        config.cache_config.cache_dtype = "auto"
        config.model_config.compute_hash.return_value = "different-model-or-attention-config"
        self.assertNotEqual(original, protocol.layerwise_topology_namespace(config))
        config.model_config.compute_hash.return_value = "test-model-config"
        config.model_config.get_layers_start_end_indices.side_effect = lambda p: ((0, 4), (4, 7))[p.rank // 2]
        self.assertNotEqual(original, protocol.layerwise_topology_namespace(config))
        groups = worker.kv_cache_config.kv_cache_groups
        wrapped = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    g.layer_names, UniformTypeKVCacheSpecs.from_specs({name: g.kv_cache_spec for name in g.layer_names})
                )
                for g in groups
            ]
        )
        self.assertEqual(
            protocol.hybrid_layout_id(worker.kv_cache_config, 2, namespace=original),
            protocol.hybrid_layout_id(wrapped, 2, namespace=original),
        )

    def test_empty_groups_and_missing_local_mapping_fail_closed(self):
        worker, _ = self.make_stage(1, MemoryRangeStore())
        worker.physical_layer_to_group_layers.pop(0)
        with self.assertRaisesRegex(RuntimeError, "no KV cache group"):
            worker._groups_for_layerwise_transfer(0)
        group = copy(worker.kv_cache_config.kv_cache_groups[0])
        group.layer_names = []
        with self.assertRaisesRegex(ValueError, "empty groups"):
            protocol.validate_pp_groups(SimpleNamespace(kv_cache_groups=[group]), worker.vllm_config.parallel_config)
        for dimension in ("decode_context_parallel_size", "prefill_context_parallel_size"):
            config = SimpleNamespace(pipeline_parallel_size=2, **{dimension: 2})
            with self.assertRaisesRegex(ValueError, "context parallelism"):
                protocol.validate_topology(config)

    def test_empty_last_layer_waits_for_earlier_put(self):
        store = MemoryRangeStore()
        worker, _ = self.make_stage(1, store, empty_tail=True)
        copy_started, release_copy, final_entered, returned = (threading.Event() for _ in range(4))
        real_put = store.batch_copy_put
        real_finish = worker._wait_for_final_layer_save
        failures = []

        def delayed_put(*args):
            copy_started.set()
            if not release_copy.wait(timeout=3):
                raise RuntimeError("test did not release PUT")
            return real_put(*args)

        def finish(*args):
            final_entered.set()
            return real_finish(*args)

        def run():
            try:
                self.run_step(
                    worker,
                    ReqMeta(
                        "r",
                        token_len_chunk=64,
                        block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
                        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                        can_save=True,
                        is_last_chunk=True,
                    ),
                )
            except Exception as exc:
                failures.append(exc)
            finally:
                returned.set()

        store.batch_copy_put = delayed_put
        worker._wait_for_final_layer_save = finish
        compute = threading.Thread(target=run, daemon=True)
        compute.start()
        try:
            self.assertTrue(copy_started.wait(timeout=1))
            self.assertTrue(final_entered.wait(timeout=1), "per-layer fencing destroyed compute/transfer overlap")
            self.assertFalse(returned.wait(timeout=0.1), "source block lifetime ended before the earlier PUT")
        finally:
            release_copy.set()
            compute.join(timeout=2)
        self.assertFalse(compute.is_alive())
        self.assertFalse(failures)
        self.assertEqual(len(store.complete), 6)

    def test_pp_one_keys_are_unchanged(self):
        self.assertEqual(protocol.make_block_key("model", "hash", 1), "model@hash@1")
        self.assertEqual(
            protocol.hybrid_block_key("model", "layout", 0, 16, "hash", 1),
            "model@mooncake_hybrid_v1:layout@group:0@block:16@hash@1",
        )


if __name__ == "__main__":
    unittest.main()
