# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec

from tests.ut.distributed.ascend_store import test_pool_scheduler as scheduler_tests
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
    reset_attention_compute_start_gate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread, _LayerRevokeTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_hybrid import hybrid_block_key, hybrid_layout_id
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import MooncakeSessionTracker
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


class TestMooncakeHybrid(unittest.TestCase):
    def make_hybrid(self, extra_entry=False):
        groups = [
            KVCacheGroupSpec(
                ["model.layers.0.kv", "model.layers.2.kv"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype="uint8"),
            ),
            KVCacheGroupSpec(
                ["model.layers.1.c4", "model.layers.2.c4", "model.layers.3.c4"],
                FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=1, dtype="uint8"),
            ),
            KVCacheGroupSpec(
                ["model.layers.0.state", "model.layers.3.state"],
                SlidingWindowSpec(block_size=16, sliding_window=16, num_kv_heads=1, head_size=1, dtype="uint8"),
            ),
        ]
        if extra_entry:
            groups[1].layer_names.append("model.layers.2.c4_indexer")
        config = SimpleNamespace(num_blocks=8, kv_cache_groups=groups)
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.KVPoolWorker._build_cache_coordinator",
            return_value=None,
        ):
            worker = make_worker(self, num_layers=4, num_hidden_layers=4, use_layerwise=True, kv_cache_config=config)
        store = MemoryRangeStore()
        worker.m_store = store
        arrays = {}
        for group, spec in enumerate(groups):
            for layer, name in enumerate(spec.layer_names):
                arrays[name] = np.full((8, 8 + group * 4), 1 + group * 16 + layer, dtype=np.uint8)
        with (
            patch.object(KVTransferThread, "start", lambda thread: thread.ready_event.set()),
            patch.object(worker, "_align_kv_ptrs"),
        ):
            worker.register_kv_caches({name: (cpu_cache(array),) for name, array in reversed(list(arrays.items()))})
        # Real worker queues/handlers in bounded Python threads; no NPU kernels.
        # Each dispatched job is tracked so a test assertion cannot leave it hung.
        jobs = []
        errors = []

        lane_locks = {thread: threading.Lock() for thread in (worker.kv_send_thread, worker.kv_recv_thread)}

        def dispatch(thread, task):
            KVTransferThread.add_request(thread, task)

            def run():
                try:
                    with lane_locks[thread]:
                        thread._handle_request(thread.request_queue.get())
                except Exception as exc:
                    errors.append(exc)
                    thread._fatal_error = exc

            job = threading.Thread(target=run, daemon=True)
            jobs.append(job)
            job.start()

        worker.kv_send_thread.add_request = lambda task: dispatch(worker.kv_send_thread, task)
        worker.kv_send_thread.add_revoke_request = lambda keys: dispatch(
            worker.kv_send_thread, _LayerRevokeTask(tuple(keys))
        )
        worker.kv_recv_thread.add_request = lambda task: dispatch(worker.kv_recv_thread, task)
        worker.cache_coordinator = object()
        masks = ([True] * 4, [True] * 2, [False, False, False, True])
        worker.token_database.store_mask = MagicMock(return_value=masks)
        worker.token_database.load_mask = MagicMock(return_value=masks)
        return worker, store, arrays, jobs, errors

    def test_multigroup_roundtrip_sparse_masks_and_group_commit_boundaries(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        saved = {name: array.copy() for name, array in arrays.items()}
        request = ReqMeta(
            "save",
            token_len_chunk=64,
            block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            can_save=True,
            is_last_chunk=True,
        )
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        self.assertEqual(len(store.objects), 7)
        self.assertEqual(sorted(len(value) for value in store.objects.values()), [16] * 4 + [32] + [36] * 2)
        for layer in range(4):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
            if layer < 2:
                self.assertFalse(store.complete)
            elif layer == 2:
                self.assertEqual(len(store.complete), 4, "Group 0 completes before the last physical layer")
            worker.save_kv_layer(meta)
        self.assertEqual(len(store.complete), 7)
        self.assertFalse(worker._put_started_keys)
        for array in arrays.values():
            array.fill(0)
        worker.kv_role = "kv_consumer"
        meta = AscendConnectorMetadata(set())
        meta.add_request(
            ReqMeta(
                "load",
                token_len_chunk=64,
                block_ids_by_group=request.block_ids_by_group,
                block_hashes=request.block_hashes,
                load_spec=LoadSpec(0, 64, can_load=True),
                is_last_chunk=True,
            )
        )
        worker.start_load_kv(meta)
        for _ in range(4):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
        for name, actual in arrays.items():
            slots = [4] if "state" in name else [1, 2] if "c4" in name else [1, 2, 3, 4]
            np.testing.assert_array_equal(actual[slots], saved[name][slots])
            self.assertFalse(actual[0].any(), "Null block must stay untouched")
            if "state" in name:
                self.assertFalse(actual[1:4].any(), "Unreachable state must not be loaded")
        for job in jobs:
            job.join(timeout=2)
            self.assertFalse(job.is_alive())
        self.assertFalse(errors)
        self.assertFalse(store.open_reads)

    def test_tracker_keeps_equal_block_indices_in_different_groups(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("request", [("swa", 0)], group_id=0)
        tracker.register_put_keys("request", [("compressed", 0)], group_id=1)
        tracker.commit_put_keys(["swa", "compressed"])
        self.assertEqual(tracker.prepare_load_entries("request", [], group_id=0), [("swa", 0)])
        self.assertEqual(tracker.prepare_load_entries("request", [], group_id=1), [("compressed", 0)])
        for key in ("swa", "compressed"):
            tracker.record_get_result(key, ["request"], succeeded=True)
        self.assertEqual(set(tracker.release_terminal({"request"})), {"swa", "compressed"})

    def test_layout_fingerprint_covers_group_membership_and_dtype(self):
        config = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(["model.layers.0.kv"], FullAttentionSpec(block_size=16, dtype="uint8")),
            ]
        )
        original = hybrid_layout_id(config)
        self.assertEqual(original, hybrid_layout_id(config))
        config.kv_cache_groups[0].layer_names.append("model.layers.1.kv")
        self.assertNotEqual(original, hybrid_layout_id(config))
        config.kv_cache_groups[0].layer_names.pop()
        config.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(block_size=16, dtype="float16")
        self.assertNotEqual(original, hybrid_layout_id(config))

    def test_attention_window_drains_before_communication_and_on_exception(self):
        for fail in (False, True):
            events = []
            gate = reset_attention_compute_start_gate()
            gate.on_start = lambda events=events: events.append("put/get submit")
            gate.on_finish = lambda events=events: events.append("put/get complete")
            try:
                with attention_transfer_window():
                    events.append("attention submitted")
                    if fail:
                        raise RuntimeError("kernel launch failed")
            except RuntimeError:
                pass
            events.append("communication")
            self.assertEqual(events, ["put/get submit", "attention submitted", "put/get complete", "communication"])

    def test_coordinator_queries_all_heads_using_mooncake_existence(self):
        scheduler = object.__new__(KVPoolScheduler)
        scheduler.mooncake_hybrid = True
        scheduler.mooncake_hybrid_layout = "layout"
        scheduler.model_name = "model"
        scheduler.tp_size = 2
        scheduler.put_step = 1
        scheduler.grouped_block_size = [16, 32]
        scheduler.kv_cache_group_families = ["c1", "c4"]
        scheduler.layerwise_max_transfer_blocks = 1
        scheduler.store_scheduler = MagicMock()
        scheduler.cache_coordinator = MagicMock()

        def lookup(hashes, length, query, **kwargs):
            self.assertEqual(query(0, hashes, [False, True]), [b"h1"])
            self.assertEqual(query(1, [b"h1"], None), [], "One missing head must make its group miss")
            return 0

        scheduler.cache_coordinator.find_reachable_hit_tokens.side_effect = lookup
        scheduler.store_scheduler.batch_is_exist.side_effect = [[1, 1], [1, 0]]
        req = SimpleNamespace(request_id="r", block_hashes=[b"h0", b"h1"])
        self.assertEqual(scheduler._lookup_layerwise_with_coordinator(req, 32), 0)
        scheduler.store_scheduler.batch_get_key_info.assert_not_called()
        keys = scheduler.store_scheduler.batch_is_exist.call_args_list[0].args[0]
        self.assertEqual(keys[0], hybrid_block_key("model", "layout", 0, "c1", 16, "6831", 0))

    def save_prefix(self, worker, last_chunk=True):
        request = ReqMeta(
            "request",
            token_len_chunk=64,
            block_ids_by_group=[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4]],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            can_save=True,
            is_last_chunk=last_chunk,
        )
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        for _ in range(4):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
            worker.save_kv_layer(meta)
        return request

    def test_unequal_cache_entries_in_one_physical_layer(self):
        worker, store, arrays, jobs, errors = self.make_hybrid(extra_entry=True)
        request = self.save_prefix(worker)
        self.assertEqual(sorted(len(value) for value in store.objects.values()), [16] * 4 + [32] + [48] * 2)
        saved = {name: array.copy() for name, array in arrays.items()}
        for array in arrays.values():
            array.fill(0)
        worker.kv_role = "kv_consumer"
        request.load_spec = LoadSpec(0, 64, can_load=True)
        request.can_save = False
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        for _ in range(4):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
        for name, array in arrays.items():
            slots = [4] if "state" in name else [1, 2] if "c4" in name else [1, 2, 3, 4]
            np.testing.assert_array_equal(array[slots], saved[name][slots])
        self.assertFalse(errors)
        self.assertFalse(store.open_reads)

    def test_continuation_restores_group_keys_into_new_local_blocks(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        request = self.save_prefix(worker, last_chunk=False)
        saved = {name: array.copy() for name, array in arrays.items()}
        for array in arrays.values():
            array.fill(0)
        # A continuation has no new scheduler load_spec. The logical block
        # indices stay stable, but preemption/reallocation changes local IDs.
        request.save_start_token = request.save_end_token = 64
        request.block_ids_by_group = [[2, 3, 4, 5], [3, 4], [2, 3, 4, 5]]
        request.can_save = False
        request.is_last_chunk = True
        self.assertIsNone(request.load_spec)
        worker.kv_role = "kv_consumer"
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        for _ in range(4):
            worker.wait_for_layer_load()
            with attention_transfer_window():
                pass
        for name, array in arrays.items():
            src, dst = (
                ([4], [5]) if "state" in name else ([1, 2], [3, 4]) if "c4" in name else ([1, 2, 3, 4], [2, 3, 4, 5])
            )
            np.testing.assert_array_equal(array[dst], saved[name][src])
        self.assertTrue(all(indices is None for indices in worker.kv_recv_thread._group_load_indices.values()))
        self.assertFalse(errors)
        self.assertFalse(store.open_reads)

    def test_failed_put_range_revokes_only_affected_group_keys(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        real_put = store.batch_copy_put

        def put(keys, buffers, sizes, offsets):
            if all("@group:1@" in key for key in keys):
                return [-1] * len(keys)
            return real_put(keys, buffers, sizes, offsets)

        store.batch_copy_put = put
        self.save_prefix(worker)
        self.assertEqual(len(store.complete), 5)
        self.assertTrue(all("@group:1@" not in key for key in store.objects))
        self.assertFalse(worker._put_started_keys)
        self.assertFalse(errors)

    def test_failed_get_does_not_allow_incomplete_attention(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        request = self.save_prefix(worker)
        request.load_spec = LoadSpec(0, 64, can_load=True)
        request.can_save = False
        worker.kv_role = "kv_consumer"
        store.batch_copy_get = lambda keys, *args: [-1] * len(keys)
        meta = AscendConnectorMetadata(set())
        meta.add_request(request)
        worker.start_load_kv(meta)
        with self.assertRaisesRegex(RuntimeError, "refusing incomplete"):
            worker.wait_for_layer_load()
        for job in jobs:
            job.join(timeout=2)
            self.assertFalse(job.is_alive())
        self.assertTrue(worker.get_block_ids_with_load_errors())
        self.assertFalse(store.open_reads)

    def test_put_start_failure_rolls_back_earlier_groups(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        real_start = store.batch_put_start

        def start(keys, sizes):
            real_start(keys, sizes)
            if any("@group:1@" in key for key in keys):
                raise RuntimeError("allocator disconnected")
            return [0] * len(keys)

        store.batch_put_start = start
        with self.assertRaisesRegex(RuntimeError, "allocator disconnected"):
            self.save_prefix(worker)
        for job in jobs:
            job.join(timeout=2)
            self.assertFalse(job.is_alive())
        self.assertFalse(store.objects)
        self.assertFalse(worker._put_started_keys)
        self.assertFalse(errors)

    def test_fatal_recv_with_queued_work_does_not_hang_drain(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        worker.kv_recv_thread.request_queue.put(object())
        worker.kv_recv_thread._fatal_error = RuntimeError("receiver died")
        with self.assertRaisesRegex(RuntimeError, "failed during asynchronous transfer"):
            worker._drain_attention_transfers()
        worker.kv_recv_thread.request_queue.get_nowait()
        worker.kv_recv_thread.request_queue.task_done()
        # Also cover failure before a new attention gate has been configured.
        with self.assertRaises(RuntimeError):
            worker.wait_for_layer_load()

    def test_shared_coordinator_requires_reachable_state_in_all_groups(self):
        group_config = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(["model.layers.0.kv"], FullAttentionSpec(block_size=16, dtype="uint8")),
                KVCacheGroupSpec(
                    ["model.layers.1.state"], SlidingWindowSpec(block_size=16, sliding_window=16, dtype="uint8")
                ),
            ]
        )
        config = scheduler_tests.make_config(block_size=32)
        config.cache_config.prefix_match_unit = 16
        config.scheduler_config.disable_hybrid_kv_cache_manager = False
        with (
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as importer,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
                side_effect=lambda spec: (
                    scheduler_tests._SparseSWAHitManager
                    if getattr(spec, "sliding_window", None) is not None
                    else scheduler_tests._PrefixHitManager
                ),
            ),
        ):
            importer.import_module.return_value = MagicMock()
            scheduler = KVPoolScheduler(config, kv_cache_config=group_config, use_layerwise=True)
            self.assertTrue(scheduler.mooncake_hybrid)
            request = SimpleNamespace(request_id="r", block_hashes=[bytes([i]) * 32 for i in range(4)])
            for tail_blocks, expected in (({1, 3}, 64), ({1}, 32), (set(), 0)):

                def exists(keys, tail_blocks=tail_blocks):
                    return [int("@group:0@" in key or int(key.split("@")[-2][:2], 16) in tail_blocks) for key in keys]

                scheduler.store_scheduler.batch_is_exist.side_effect = exists
                self.assertEqual(scheduler._get_layerwise_hit_tokens(request, 64, 0), expected)
            scheduler.store_scheduler.batch_get_key_info.assert_not_called()

    def test_prefetch_waits_for_device_attention_boundary(self):
        device_ready = threading.Event()
        synchronizing = threading.Event()
        transfer_started = threading.Event()
        device_event = MagicMock()

        def synchronize():
            synchronizing.set()
            if not device_ready.wait(timeout=2):
                raise TimeoutError("test device fence")

        device_event.synchronize.side_effect = synchronize
        gate = reset_attention_compute_start_gate()
        job = threading.Thread(target=lambda: transfer_started.set() if gate.wait(timeout=2) else None, daemon=True)
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence.torch.npu.Event",
            return_value=device_event,
        ):
            job.start()
            gate.record()
            self.assertTrue(synchronizing.wait(timeout=1))
            self.assertFalse(transfer_started.is_set())
            device_ready.set()
            job.join(timeout=2)
        self.assertTrue(transfer_started.is_set())
        self.assertFalse(job.is_alive())

    def test_mask_failure_does_not_fall_back_to_copying_all_state(self):
        worker, store, arrays, jobs, errors = self.make_hybrid()
        worker.token_database.store_mask.side_effect = AssertionError("unreachable state")
        with self.assertRaisesRegex(AssertionError, "unreachable state"):
            self.save_prefix(worker)
        self.assertFalse(store.objects)
