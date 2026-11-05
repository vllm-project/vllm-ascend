# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DCP session ownership, publication and byte-exact shard restores."""

import unittest

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from tests.ut.distributed.ascend_store import test_mooncake_hybrid as hybrid_tests
from tests.ut.distributed.ascend_store import test_mooncake_pipeline as pipeline_tests
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta


class TestMooncakeDCP(unittest.TestCase):
    def test_single_group_dcp_roundtrip_and_incomplete_rank_miss(self):
        store = hybrid_tests.MemoryRangeStore()
        workers = []
        # Keep the real two-stage PP layout used by the shared fixture; DCP
        # ranks are the TP ranks within each stage.
        for stage in range(2):
            for dcp_rank in range(2):
                worker, arrays = pipeline_tests.TestMooncakePipeline.make_stage(
                    self,
                    stage,
                    store,
                    hybrid=False,
                    draft=False,
                    mla=True,
                    tp_rank=dcp_rank,
                    dcp_size=2,
                    dcp_rank=dcp_rank,
                )
                # Each DCP rank owns a different context shard of the same
                # logical block, so the payloads must not be interchangeable.
                for array in arrays.values():
                    array ^= np.uint8(41 * dcp_rank)
                workers.append((worker, arrays, {name: array.copy() for name, array in arrays.items()}))

        scheduler = pipeline_tests.TestMooncakePipeline.make_scheduler(self, workers[0][0], store)
        keys = scheduler._make_layerwise_hit_check_keys(0, b"h0".hex())
        self.assertEqual(len(set(keys)), 4)
        for worker, _, _ in workers:
            self.assertIn(worker._make_layerwise_full_key(0, b"h0".hex()), keys)

        save_request = ReqMeta(
            "save",
            token_len_chunk=64,
            block_ids=[1, 2],
            block_hashes=[b"h0", b"h1"],
            can_save=True,
            is_last_chunk=True,
        )
        for index, (worker, _, _) in enumerate(workers):
            pipeline_tests.TestMooncakePipeline.run_step(worker, save_request)
            worker._drain_deferred_last_save()
            # Every PP stage and DCP shard is required for a complete block.
            self.assertEqual(scheduler._query_layerwise_block_hits([keys]), [index == len(workers) - 1])

        # Clear every destination and use different block IDs, proving each
        # rank restores its own remote shard rather than another rank's data.
        for worker, arrays, saved in workers:
            for array in arrays.values():
                array.fill(0)
            worker.kv_role = "kv_consumer"
            load_request = ReqMeta(
                "load",
                token_len_chunk=64,
                block_ids=[3, 4],
                block_hashes=[b"h0", b"h1"],
                load_spec=LoadSpec(0, 64, can_load=True),
                is_last_chunk=True,
            )
            pipeline_tests.TestMooncakePipeline.run_step(worker, load_request)
            for name, array in arrays.items():
                np.testing.assert_array_equal(array[[3, 4]], saved[name][[1, 2]])
        self.assertFalse(store.open_reads)


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
