# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU range-session integration tests for PP/DCP, without an NPU runtime."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    get_mooncake_layerwise_namespace,
    make_layerwise_block_key,
    validate_mooncake_layerwise_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


class MemoryRangeStore:
    """Exercise real range offsets against CPU buffers, including visibility."""

    def __init__(self):
        self.objects = {}
        self.complete = set()
        self.open_reads = set()

    def register_buffer(self, ptrs, lengths):
        assert all(length > 0 for length in lengths)

    def validate_layerwise_support(self):
        pass

    def batch_put_start(self, keys, sizes):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects, f"Duplicate shard writer: {key}"
            self.objects[key] = bytearray(size)
        return [0] * len(keys)

    def batch_copy_put(self, keys, buffers, sizes, offsets):
        for key, addrs, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
            for addr, length, start in zip(addrs, lengths, starts, strict=True):
                assert 0 <= start < start + length <= len(self.objects[key])
                self.objects[key][start : start + length] = ctypes.string_at(addr, length)
        return [sum(row) for row in sizes]

    def batch_commit(self, keys):
        self.complete.update(keys)
        return [0] * len(keys)

    def batch_revoke(self, keys):
        for key in keys:
            self.objects.pop(key, None)
            self.complete.discard(key)
        return [0] * len(keys)

    def batch_is_exist(self, keys):
        return [int(key in self.complete) for key in keys]

    def batch_get_start(self, keys):
        self.open_reads.update(key for key in keys if key in self.complete)
        return [0 if key in self.complete else -1 for key in keys]

    def batch_copy_get(self, keys, buffers, sizes, offsets):
        for key, addrs, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
            assert key in self.open_reads
            for addr, length, start in zip(addrs, lengths, starts, strict=True):
                assert 0 <= start < start + length <= len(self.objects[key])
                ctypes.memmove(addr, bytes(self.objects[key][start : start + length]), length)
        return [sum(row) for row in sizes]

    def batch_get_end(self, keys):
        self.open_reads.difference_update(keys)
        return 0


def cpu_cache(array):
    cache = MagicMock()
    cache.shape = array.shape
    cache.element_size.return_value = array.itemsize
    cache.stride.return_value = array.strides[0] // array.itemsize
    cache.data_ptr.return_value = array.ctypes.data
    cache.untyped_storage.return_value.data_ptr.return_value = array.ctypes.data
    cache.__getitem__.return_value.numel.return_value = array[0].size
    return cache


def handle_inline(thread, task):
    # Keep queue accounting and real transfer handlers, but schedule them
    # deterministically so the tests never depend on NPU/Python threads.
    KVTransferThread.add_request(thread, task)
    thread._handle_request(thread.request_queue.get_nowait())


class TestMooncakeLayerwiseParallel(unittest.TestCase):
    def make_scheduler(self, tp, pp, dcp, heads):
        config = make_config(extra_config={"backend": "mooncake", "layerwise_max_transfer_blocks": 1})
        config.parallel_config.tensor_parallel_size = tp
        config.parallel_config.pipeline_parallel_size = pp
        config.parallel_config.decode_context_parallel_size = dcp
        config.parallel_config.cp_kv_cache_interleave_size = 1
        config.model_config.get_total_num_kv_heads.return_value = heads
        config.model_config.use_mla = heads == 1
        config.model_config.get_layers_start_end_indices.side_effect = lambda pc: (
            (pc.rank // tp) * 2,
            (pc.rank // tp + 1) * 2,
        )
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib"):
            return KVPoolScheduler(config, use_layerwise=True)

    def test_pp_dcp_round_trip_all_shards_and_stage_local_layers(self):
        for tp, pp, dcp, heads in [(2, 2, 1, 1), (2, 1, 2, 1), (4, 2, 2, 1), (8, 2, 2, 2)]:
            with self.subTest(tp=tp, pp=pp, dcp=dcp, heads=heads):
                store = MemoryRangeStore()
                scheduler = self.make_scheduler(tp, pp, dcp, heads)
                scheduler.store_scheduler = store
                workers = []
                query = SimpleNamespace(request_id="query", block_hashes=[b"h0", b"h1"])
                for pp_rank in range(pp):
                    for tp_rank in range(tp):
                        cache_config = SimpleNamespace(
                            num_blocks=4,
                            kv_cache_groups=[
                                SimpleNamespace(
                                    layer_names=[f"model.layers.{pp_rank * 2 + i}.self_attn" for i in range(2)],
                                    kv_cache_spec=SimpleNamespace(block_size=16),
                                )
                            ],
                        )
                        worker = make_worker(
                            self,
                            tp_rank=tp_rank,
                            tp_size=tp,
                            pp_size=pp,
                            pp_rank=pp_rank,
                            dcp_size=dcp,
                            num_kv_heads=heads,
                            use_mla=heads == 1,
                            num_hidden_layers=2 * pp,
                            use_layerwise=True,
                            kv_cache_config=cache_config,
                        )
                        worker.m_store = store
                        # Every PP stage and DCP shard carries different bytes;
                        # only truly replicated KV-head/DCP ranks share bytes.
                        value = 1 + pp_rank * 32 + worker.head_or_tp_rank * 8 + worker.dcp_rank * 2
                        arrays = [np.full((4, 16), value + layer, dtype=np.uint8) for layer in range(2)]
                        with patch.object(KVTransferThread, "start", lambda thread: thread.ready_event.set()):
                            worker.register_kv_caches(
                                {
                                    f"model.layers.{pp_rank * 2 + layer}.self_attn": (cpu_cache(arrays[layer]),)
                                    for layer in reversed(range(2))
                                }
                            )
                        self.assertEqual(worker.num_layers, 2)
                        self.assertEqual(worker.kv_send_thread.final_layer_id, 1)
                        self.assertEqual(worker._mooncake_object_size_bytes(), 32)
                        self.assertEqual(worker.mooncake_layerwise_namespace, scheduler.mooncake_layerwise_namespace)
                        worker.kv_send_thread.add_request = lambda task, thread=worker.kv_send_thread: handle_inline(
                            thread, task
                        )
                        worker.kv_recv_thread.add_request = lambda task, thread=worker.kv_recv_thread: handle_inline(
                            thread, task
                        )
                        request = ReqMeta(
                            "save",
                            token_len_chunk=worker.block_size * 2,
                            save_start_token=0,
                            save_end_token=worker.block_size * 2,
                            target_token_len=worker.block_size * 2,
                            block_ids=[0, 1],
                            block_hashes=[b"h0", b"h1"],
                            can_save=True,
                            is_last_chunk=True,
                        )
                        metadata = AscendConnectorMetadata(set())
                        metadata.add_request(request)
                        worker.start_load_kv(metadata)
                        before = store.complete.copy()
                        worker.wait_for_layer_load()
                        worker.save_kv_layer(metadata)
                        self.assertEqual(store.complete, before, "A stage must not commit before its last local layer")
                        worker.wait_for_layer_load()
                        worker.save_kv_layer(metadata)
                        self.assertEqual(worker.current_layer, 2)
                        workers.append((worker, arrays, [array.copy() for array in arrays]))

                self.assertEqual(len(store.complete), pp * dcp * heads * 2)
                self.assertEqual(scheduler._get_mooncake_layerwise_hit_tokens(query, 32 * dcp, 0), 32 * dcp)
                for worker, arrays, expected in workers:
                    # A load-only consumer must progress without save callbacks.
                    worker.kv_role = "kv_consumer"
                    for array in arrays:
                        array.fill(0)
                    metadata = AscendConnectorMetadata(set())
                    metadata.add_request(
                        ReqMeta(
                            "load",
                            token_len_chunk=worker.block_size * 2,
                            block_ids=[0, 1],
                            block_hashes=[b"h0", b"h1"],
                            is_last_chunk=True,
                            load_spec=LoadSpec(0, worker.block_size * 2, can_load=True),
                        )
                    )
                    worker.start_load_kv(metadata)
                    worker.wait_for_layer_load()
                    worker.wait_for_layer_load()
                    self.assertEqual(worker.current_layer, 2)
                    self.assertEqual(worker.get_block_ids_with_load_errors(), set())
                    for actual, saved in zip(arrays, expected, strict=True):
                        np.testing.assert_array_equal(actual[:2], saved[:2])
                        self.assertFalse(actual[2:].any(), "Transfer must stay within requested local blocks")
                self.assertEqual(store.open_reads, set())

                # Missing just one stage/shard stops the contiguous hit prefix.
                for missing in {worker._make_mooncake_layerwise_key(b"h1".hex()) for worker, _, _ in workers}:
                    store.complete.remove(missing)
                    self.assertEqual(scheduler._get_mooncake_layerwise_hit_tokens(query, 32 * dcp, 0), 16 * dcp)
                    store.complete.add(missing)

    def test_partial_session_keys_are_isolated_by_stage_and_dcp_shard(self):
        keys = set()
        for pp_rank in range(2):
            for tp_rank in range(2):
                worker = make_worker(
                    self,
                    pp_rank=pp_rank,
                    pp_size=2,
                    tp_rank=tp_rank,
                    tp_size=2,
                    dcp_size=2,
                    use_layerwise=True,
                    use_mla=True,
                )
                worker.group_block_len = {0: [16, 16]}
                worker.m_store.batch_put_start.return_value = [0]
                request = ReqMeta(
                    "shared-request-id",
                    token_len_chunk=16,
                    save_end_token=16,
                    block_ids=[0],
                    block_hashes=[],
                    can_save=True,
                    partial_block_index=0,
                )
                worker._prepare_mooncake_put_session(request)
                self.assertIsNotNone(request.save_last_block_key)
                self.assertNotIn(request.save_last_block_key, keys)
                keys.add(request.save_last_block_key)
                request.load_spec = LoadSpec(0, 16, can_load=True)
                slots = worker._prepare_mooncake_get_session(request)
                self.assertEqual(slots, [(request.save_last_block_key, 0, 0)])
        self.assertEqual(len(keys), 4)

    def test_namespace_separates_partitions_and_interleave(self):
        parallel = SimpleNamespace(
            rank=7,
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        )
        model = MagicMock()
        model.get_layers_start_end_indices.side_effect = lambda pc: [(0, 3), (3, 5)][pc.rank // 4]
        first = get_mooncake_layerwise_namespace(model, parallel)
        self.assertIn("pp:0-3,3-5", first)
        self.assertEqual(parallel.rank, 7)
        model.get_layers_start_end_indices.side_effect = lambda pc: [(0, 2), (2, 5)][pc.rank // 4]
        second = get_mooncake_layerwise_namespace(model, parallel)
        self.assertNotEqual(first, second)
        parallel.cp_kv_cache_interleave_size = 16
        self.assertNotEqual(second, get_mooncake_layerwise_namespace(model, parallel))
        self.assertEqual(make_layerwise_block_key("model", "hash", 0), "model@hash@0")
        with self.assertRaisesRegex(ValueError, "topology namespace"):
            make_layerwise_block_key("model", "hash", 0, pp_rank=1)
        model.get_layers_start_end_indices.side_effect = lambda pc: [(0, 5), (5, 5)][pc.rank // 4]
        with self.assertRaisesRegex(ValueError, "at least one model layer"):
            get_mooncake_layerwise_namespace(model, parallel)

    def test_pcp_remains_rejected(self):
        parallel = SimpleNamespace(
            prefill_context_parallel_size=2, pipeline_parallel_size=2, decode_context_parallel_size=2
        )
        with self.assertRaisesRegex(ValueError, "PCP"):
            validate_mooncake_layerwise_topology(parallel, "mooncake", True)
        validate_mooncake_layerwise_topology(parallel, "mooncake", False)
        parallel.prefill_context_parallel_size = 1
        validate_mooncake_layerwise_topology(parallel, "mooncake", True)

    def test_dcp_must_fit_within_replicated_heads(self):
        with self.assertRaisesRegex(ValueError, "replicated KV-head group"):
            self.make_scheduler(tp=4, pp=1, dcp=2, heads=4)
        with self.assertRaisesRegex(ValueError, "replicated KV-head group"):
            make_worker(self, tp_size=4, dcp_size=2, num_kv_heads=4, use_layerwise=True)
