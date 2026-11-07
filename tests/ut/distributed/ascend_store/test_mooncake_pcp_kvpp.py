# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP x KVPP: owner-scoped layerwise keys, completeness and byte-exact restores."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from tests.ut.distributed.ascend_store import test_mooncake_hybrid as hybrid_tests
from tests.ut.distributed.ascend_store import test_mooncake_pipeline as pipeline_tests
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import LayerwiseKeyBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta

TP_SIZE = 2
PCP_SIZE = 2
KVPP_SIZE = TP_SIZE * PCP_SIZE
LAYERS = list(range(KVPP_SIZE))
LAYER_NAMES = [f"model.layers.{layer}.kv" for layer in LAYERS]


class TestMooncakePCPKVPP(unittest.TestCase):
    """Every KVPP owner owns one target layer; PCP folds into ``kvpp_rank``."""

    def configure_kvpp_worker(self, worker, arrays, owner):
        config = worker.vllm_config
        config.additional_config = {"enable_kvpp": True}
        # The KVPP key space needs a draft-free projected cache layout so the
        # layer owner mapping matches the registered caches.
        config.speculative_config = None
        config.compilation_config = SimpleNamespace(static_forward_context={})
        worker.use_kvpp = True
        worker.kvpp_rank = owner
        worker.kvpp_size = KVPP_SIZE
        worker.head_or_tp_rank = owner
        worker.put_step = 1
        for metadata in worker.metadata:
            metadata.head_or_tp_rank = owner
        worker.layerwise_keys = worker.layerwise_backend.bind_layerwise_keys(
            vllm_config=config,
            kv_cache_config=worker.kv_cache_config,
            model_name=worker.model_name,
            use_hybrid=False,
            grouped_block_size=worker.grouped_block_size,
        )

        # KVPP registration receives every cache entry and keeps only this
        # owner's persistent target-layer shard. Pin layer-index extraction so
        # the owner partition does not depend on the test dependency mocks.
        def layer_index(layer_name):
            return int(layer_name.split(".")[2])

        with patch("vllm_ascend.core.kv_cache_placement.extract_layer_index", side_effect=layer_index):
            worker.register_kv_caches({name: (hybrid_tests.cpu_cache(arrays[name]),) for name in LAYER_NAMES})
        builders = worker._build_group_layer_builders()
        for thread in (worker.kv_send_thread, worker.kv_recv_thread):
            thread.group_builders = builders
            thread.layer_batch_builder = builders[0]
        self.assertEqual(worker.group_num_layers, {0: 1})
        self.assertEqual(worker.kvpp_rank, owner)

    def make_workers(self, store):
        workers = []
        for pcp_rank in range(PCP_SIZE):
            for tp_rank in range(TP_SIZE):
                owner = pcp_rank * TP_SIZE + tp_rank
                worker, arrays = pipeline_tests.TestMooncakePipeline.make_stage(
                    self,
                    0,
                    store,
                    hybrid=False,
                    draft=False,
                    mla=True,
                    tp_rank=tp_rank,
                    pcp_size=PCP_SIZE,
                    pcp_rank=pcp_rank,
                    layers=LAYERS,
                    num_layers=len(LAYERS),
                    # DP x PP x PCP x TP with PP=1: keep the owner plane flat.
                    pp_partition=None,
                )
                self.configure_kvpp_worker(worker, arrays, owner)
                workers.append((worker, arrays))
        return workers

    def test_scheduler_writer_count_folds_pcp_under_kvpp(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import (
            LayerwiseKVPoolScheduler,
        )

        def base_key(group, block_hash, rank, stage):
            return f"model@{block_hash}@{rank}@{stage}"

        scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
        scheduler.use_kvpp = True
        scheduler.kvpp_size = 8
        scheduler.tp_size = 2
        scheduler.put_step = 2
        scheduler.num_speculative_blocks_by_group = {}
        scheduler.dspark_prefix_keys = None
        scheduler.dspark_draft_only_group_ids = frozenset()
        scheduler.layerwise_keys = LayerwiseKeyBuilder(base_key, pp_size=2, pcp_size=1, kvpp=True)
        keys = LayerwiseKVPoolScheduler._make_layerwise_hit_check_keys(scheduler, 0, "hash")
        self.assertEqual(len(keys), 2 * 8)
        for key in keys:
            self.assertNotIn("@pcp_size", key)

        # Non-KVPP MLA keeps one saving head per PP stage.
        scheduler.use_kvpp = False
        scheduler.kvpp_size = 1
        scheduler.layerwise_keys = LayerwiseKeyBuilder(base_key, pp_size=2)
        keys = LayerwiseKVPoolScheduler._make_layerwise_hit_check_keys(scheduler, 0, "hash")
        self.assertEqual(len(keys), 2)

    def test_worker_key_folds_pcp_only_under_kvpp(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
            LayerwiseKVPoolWorker,
        )

        def base_key(group, block_hash, rank, stage):
            return f"model@{block_hash}@{rank}@{stage}"

        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.pp_rank = 1
        worker.pcp_rank = 3
        worker.use_kvpp = True
        worker.metadata = [SimpleNamespace(head_or_tp_rank=5)]
        worker.layerwise_keys = LayerwiseKeyBuilder(base_key, pp_size=2, pcp_size=1, kvpp=True)
        key = LayerwiseKVPoolWorker._make_layerwise_full_key(worker, 0, "hash")
        self.assertEqual(key, "model@hash@5@1")
        # The local PCP rank is only encoded when KVPP does not fold it.
        worker.use_kvpp = False
        worker.layerwise_keys = LayerwiseKeyBuilder(base_key, pp_size=2, pcp_size=3)
        key = LayerwiseKVPoolWorker._make_layerwise_full_key(worker, 0, "hash")
        self.assertEqual(key, "model@hash@5@1@pcp_size:3@pcp_rank:3")

    def test_worker_key_index_and_writer_count_fold_kvpp(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, tp_rank=1, tp_size=2, pcp_rank=1, pcp_size=2, use_mla=True, use_kvpp=True)
        self.assertEqual(worker.kvpp_rank, 3)
        self.assertEqual(worker.kvpp_size, 4)
        self.assertEqual(worker.my_key_index, 3)
        self.assertEqual(worker.num_ranks_per_layer, 4)

    def test_owner_coverage_fails_closed_for_zero_layer_owner(self):
        from vllm_ascend.core.kv_cache_placement import validate_kvpp_owner_coverage

        validate_kvpp_owner_coverage({"model.layers.0.kv": 0, "model.layers.1.kv": 1}, 2)
        with self.assertRaisesRegex(ValueError, "every owner"):
            validate_kvpp_owner_coverage({"model.layers.0.kv": 0}, 2)

    def test_pcp_kvpp_owner_roundtrip_and_incomplete_owner_miss(self):
        store = hybrid_tests.MemoryRangeStore()
        workers = self.make_workers(store)
        scheduler = pipeline_tests.TestMooncakePipeline.make_scheduler(self, workers[0][0], store)
        self.assertTrue(scheduler.use_kvpp)
        self.assertEqual(scheduler.kvpp_size, KVPP_SIZE)
        keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
        self.assertEqual(len(set(keys)), KVPP_SIZE)
        for index, (worker, _) in enumerate(workers):
            # Hit-check keys and each owner's writer key are byte-identical.
            self.assertEqual(worker._make_layerwise_full_key(0, b"h0".hex()), keys[index])

        saved = []
        for index, (worker, arrays) in enumerate(workers):
            saved.append({name: array.copy() for name, array in arrays.items()})
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
            # An incomplete owner snapshot must stay a miss: no partial restore.
            self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [index == len(workers) - 1])

        for owner, (worker, arrays) in enumerate(workers):
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
                if name == LAYER_NAMES[owner]:
                    np.testing.assert_array_equal(array[[5, 6]], saved[owner][name][[1, 2]])
                else:
                    self.assertFalse(array.any(), f"{name} must not be loaded by a non-owning rank")
        self.assertFalse(store.open_reads)

    def test_pcp_kvpp_missing_owner_keeps_previous_prefix_miss(self):
        store = hybrid_tests.MemoryRangeStore()
        workers = self.make_workers(store)
        scheduler = pipeline_tests.TestMooncakePipeline.make_scheduler(self, workers[0][0], store)
        keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
        for index, (worker, _) in enumerate(workers[:-1]):
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
            self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [False])
        self.assertEqual(len(set(keys)), KVPP_SIZE)
