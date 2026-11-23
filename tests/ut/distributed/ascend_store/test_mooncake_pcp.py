# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP session ownership, publication and byte-exact multi-group restores."""

import unittest
from unittest.mock import patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from tests.ut.distributed.ascend_store import test_mooncake_hybrid as hybrid_tests
from tests.ut.distributed.ascend_store import test_mooncake_pipeline as pipeline_tests
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import create_kv_pool_scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import BaseKVPoolScheduler


class TestMooncakePCP(unittest.TestCase):
    def test_single_group_pcp_pp_and_mtp_roundtrip(self):
        store = hybrid_tests.MemoryRangeStore()
        workers = []
        for stage in range(2):
            for pcp_rank in (1, 0):
                worker, arrays = pipeline_tests.TestMooncakePipeline.make_stage(
                    self, stage, store, hybrid=False, draft=True, mla=True, pcp_size=2, pcp_rank=pcp_rank
                )
                workers.append((worker, arrays, {name: array.copy() for name, array in arrays.items()}))
        scheduler = pipeline_tests.TestMooncakePipeline.make_scheduler(self, workers[0][0], store)
        keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
        self.assertEqual(len(set(keys)), 2)
        for index, (worker, _, _) in enumerate(workers):
            before = set(store.objects)
            request = ReqMeta(
                "save",
                token_len_chunk=32,
                block_ids=[1, 2],
                block_hashes=[b"h0", b"h1"],
                can_save=True,
                is_last_chunk=True,
            )
            pipeline_tests.TestMooncakePipeline.run_step(worker, request)
            worker._drain_deferred_last_save()
            if worker.pcp_rank != 0:
                self.assertEqual(set(store.objects), before)
            self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [index == 3])
        self.assertEqual(len(store.complete), 4)
        for worker, arrays, saved in workers:
            for array in arrays.values():
                array.fill(0)
            worker.kv_role = "kv_consumer"
            request = ReqMeta(
                "load",
                token_len_chunk=32,
                block_ids=[5, 6],
                block_hashes=[b"h0", b"h1"],
                load_spec=LoadSpec(0, 32, can_load=True),
                is_last_chunk=True,
            )
            pipeline_tests.TestMooncakePipeline.run_step(worker, request)
            for name, array in arrays.items():
                np.testing.assert_array_equal(array[[5, 6]], saved[name][[1, 2]])
        self.assertFalse(store.open_reads)

    def finish_jobs(self, jobs, errors):
        for job in jobs:
            job.join(timeout=3)
            self.assertFalse(job.is_alive(), "PCP transfer did not finish")
        self.assertFalse(errors)

    def test_hybrid_roundtrip_requires_all_pcp_or_tp_head_shards(self):
        for tp_size, pcp_size in ((1, 4), (2, 1)):
            with self.subTest(tp_size=tp_size, pcp_size=pcp_size):
                store = hybrid_tests.MemoryRangeStore()
                workers = []
                for pcp_rank in reversed(range(pcp_size)):
                    for tp_rank in range(tp_size):
                        worker, _, arrays, jobs, errors = hybrid_tests.TestMooncakeHybrid.make_hybrid(
                            self,
                            extra_entry=True,
                            private_tail=True,
                            pcp_size=pcp_size,
                            pcp_rank=pcp_rank,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                            use_mla=True,
                        )
                        self.addCleanup(self.finish_jobs, jobs, errors)
                        worker.m_store = worker.kv_send_thread.m_store = worker.kv_recv_thread.m_store = store
                        for name, array in arrays.items():
                            if "state" in name:
                                array += pcp_rank * 31 + tp_rank * 7
                        saved = {name: array.copy() for name, array in arrays.items()}
                        workers.append((worker, arrays, saved))
                first = workers[0][0]
                with (
                    patch.object(BaseKVPoolScheduler, "_build_cache_coordinator", return_value=None),
                    patch(
                        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend."
                        "MooncakeBackend.create_scheduler_client"
                    ),
                ):
                    scheduler = create_kv_pool_scheduler(
                        first.vllm_config, use_layerwise=True, kv_cache_config=first.kv_cache_config
                    )
                scheduler.store_scheduler = store
                self.assertEqual(first.grouped_block_size, first.original_block_size)
                self.assertEqual(scheduler.grouped_block_size, first.grouped_block_size)
                self.assertEqual(scheduler.hash_block_size, first.hash_block_size)
                keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
                state_keys = scheduler._make_layerwise_hit_check_keys(2, b"h3".hex())
                self.assertEqual(len(keys), 1)
                self.assertEqual(len(state_keys), pcp_size * tp_size)
                for index, (worker, _, _) in enumerate(workers):
                    self.assertEqual(worker.metadata[2].head_or_tp_rank, worker.tp_rank)
                    state_key = worker.layerwise_keys.make_base_key(2, b"h3".hex(), worker.tp_rank, worker.pp_rank)
                    if pcp_size > 1:
                        state_key += f"@pcp_size:{pcp_size}@pcp_rank:{worker.pcp_rank}"
                    self.assertEqual(worker._make_layerwise_full_key(2, b"h3".hex()), state_key)
                    self.assertIn(state_key, state_keys)
                    for group in worker.cacheable_group_ids:
                        self.assertIn(
                            worker._make_layerwise_full_key(group, b"h0".hex()),
                            scheduler._make_layerwise_hit_check_keys(group, b"h0".hex()),
                        )
                    hybrid_tests.TestMooncakeHybrid.save_prefix(self, worker)
                    # Missing the last PCP/TP state keeps the snapshot a miss.
                    self.assertEqual(scheduler._query_layerwise_block_hits([state_keys]), [index == len(workers) - 1])
                self.assertEqual(len(store.complete), 4 + 2 + pcp_size * tp_size)
                store.complete.remove(keys[-1])
                self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [False])
                store.complete.add(keys[-1])
                self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [True])

                # Clear all local bytes and use different block IDs, proving
                # the GET restores this rank rather than a stale local hit.
                for worker, arrays, saved in workers:
                    for array in arrays.values():
                        array.fill(0)
                    worker.kv_role = "kv_consumer"
                    request = ReqMeta(
                        "load",
                        token_len_chunk=64,
                        block_ids_by_group=[[3, 4, 5, 6], [5, 6], [3, 4, 5, 6], [7]],
                        block_hashes=[b"h0", b"h1", b"h2", b"h3"],
                        load_spec=LoadSpec(0, 64, can_load=True),
                        is_last_chunk=True,
                    )
                    pipeline_tests.TestMooncakePipeline.run_step(worker, request)
                    for name, array in arrays.items():
                        if "tail_cache" in name:
                            self.assertFalse(array.any())
                            continue
                        src, dst = (
                            ([4], [6])
                            if "state" in name
                            else (([1, 2], [5, 6]) if "c4" in name else ([1, 2, 3, 4], [3, 4, 5, 6]))
                        )
                        np.testing.assert_array_equal(array[dst], saved[name][src])
                        self.assertFalse(array[[slot for slot in range(8) if slot not in dst]].any())
                self.assertFalse(store.open_reads)
                self.doCleanups()
