# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise shared-source scheduling through the real layerwise GVA data plane."""

import ctypes
import threading
import unittest
from collections import Counter
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from tests.ut.distributed.ascend_store import test_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import create_kv_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta


class MemoryGVAStore:
    """Copy real CPU tensor descriptors to simulated GVA objects."""

    requires_exists_before_put = True

    def __init__(self):
        self.store = self
        self.pool = bytearray(1 << 20)
        self.next_gva = 4096
        self.objects = {}
        self.published = set()
        self.leases = set()
        self.copies = []
        self.finish_calls = []
        self.block_address = None
        self.block_direction = 1
        self.blocked = threading.Event()
        self.resume = threading.Event()
        self.fail_copy = False

    def ensure_initialized(self):
        pass

    def set_device(self):
        pass

    def register_buffer(self, *_args):
        pass

    def close(self):
        pass

    def exists(self, keys):
        return [int(key in self.published) for key in keys]

    def batch_is_exist(self, keys):
        return self.exists(keys)

    def batch_is_readable(self, keys):
        return [key in self.published for key in keys]

    def batch_alloc(self, keys, sizes, _ttl=0):
        result = []
        for key, size in zip(keys, sizes, strict=True):
            if key not in self.objects:
                self.objects[key] = (self.next_gva, size)
                self.next_gva += size
            result.append(self.objects[key][0])
        return result

    def batch_copy(self, gvas, addrs, sizes, direction):
        if direction == self.block_direction and self.block_address in addrs:
            self.blocked.set()
            if not self.resume.wait(timeout=5):
                raise RuntimeError("test copy was not released")
        if self.fail_copy:
            return -1
        self.copies.append((direction, list(addrs)))
        for gva, addr, size in zip(gvas, addrs, sizes, strict=True):
            if direction == 0:
                self.pool[gva : gva + size] = ctypes.string_at(addr, size)
            else:
                ctypes.memmove(addr, bytes(self.pool[gva : gva + size]), size)
        return 0

    def batch_write_finish(self, keys, results):
        self.finish_calls.append((list(keys), list(results)))
        self.published.update(key for key, result in zip(keys, results, strict=True) if result == 0)
        return [0] * len(keys)

    def batch_get_key_info(self, keys, *, for_load=False):
        return [
            SimpleNamespace(gva_list=lambda k=key: [self.objects[k][0]], size=lambda k=key: self.objects[k][1])
            if key in self.published
            else None
            for key in keys
        ]

    def batch_add_lease(self, keys, _ttl=0):
        self.leases.update(key for key in keys if key in self.published)
        return [0 if key in self.published else -1 for key in keys]

    def batch_remove_lease(self, keys):
        self.leases.difference_update(keys)
        return 0


class BoundsCheckingGVAStore(MemoryGVAStore):
    """Reject copies beyond the allocated object instead of growing the buffer."""

    def batch_copy(self, gvas, addrs, sizes, direction):
        for gva, size in zip(gvas, sizes, strict=True):
            if not any(base <= gva and gva + size <= base + length for base, length in self.objects.values()):
                raise RuntimeError("GVA copy exceeds its allocated object")
        return super().batch_copy(gvas, addrs, sizes, direction)


def make_shared_worker(
    test,
    *,
    bounded=False,
    draft=False,
    mode="NONE",
    sources=(2, 8),
    num_layers=12,
    global_num_layers=None,
    method=None,
    architecture="DeepseekV41DSparkModel",
    use_v2=True,
    segmented=False,
    pcp_size=1,
    pcp_rank=0,
    tp_size=1,
    tp_rank=0,
    use_mla=False,
    extra_config=None,
    store=None,
):
    """Register real CPU cache views with the production GVA worker and queue threads."""
    names = [
        name
        for source in sources
        for name in (f"model.layers.{source}.attn.long_kv_cache", f"model.layers.{source}.attn.indexer.k_cache")
    ]
    groups = [names]
    if not bounded:
        groups.append([f"model.layers.{layer}.attn.swa_cache" for layer in range(num_layers)])
    if draft:
        groups.append(["model.mtp.layers.0.attn.kv_cache"])
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    cache_config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=group, kv_cache_spec=spec) for group in groups],
        dspark_draft_layer_names=set(groups[-1]) if draft else set(),
        num_blocks=4,
        prefix_cache_retention_interval=0,
    )

    def create(config, use_layerwise, kv_cache_config):
        if global_num_layers is not None:
            config.model_config.get_layers_start_end_indices.return_value = (0, num_layers)
            config.model_config.get_total_num_hidden_layers.return_value = global_num_layers
        config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        config.model_config.hf_config = config.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=num_layers, kv_source_layer_ids=list(sources), compress_ratios=[2] * num_layers
        )
        config.model_config.enforce_eager = False
        config.compilation_config.cudagraph_mode = SimpleNamespace(name=mode)
        config.compilation_config.static_forward_context = {
            str(layer): SimpleNamespace(
                layer_idx=layer, kv_transfer_cache_names=(f"model.layers.{layer}.attn.swa_cache",)
            )
            for layer in range(num_layers)
        }
        config.parallel_config.use_ubatching = config.parallel_config.use_sequence_parallel_moe = False
        config.cache_config.swa_bounded_replay = bounded
        config.use_v2_model_runner = use_v2
        config.speculative_config = (
            None
            if method is None
            else SimpleNamespace(method=method, draft_model_config=SimpleNamespace(architectures=[architecture]))
        )
        return create_kv_pool_worker(config, use_layerwise, kv_cache_config)

    with (
        patch.object(test_pool_worker, "create_kv_pool_worker", side_effect=create),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.uses_hybrid_kv_cache",
            return_value=True,
        ),
    ):
        worker = test_pool_worker.make_worker(
            test,
            use_layerwise=True,
            num_layers=num_layers,
            num_hidden_layers=num_layers,
            extra_config={"backend": "memcache", "layerwise_num_shared_buffers": 1, **(extra_config or {})},
            kv_cache_config=cache_config,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_mla=use_mla,
        )
    store = MemoryGVAStore() if store is None else store
    worker.m_store = store
    worker.cache_coordinator = worker.token_database.cache_coordinator = None
    all_names = sum(groups, [])
    alignment = 2 * 1024 * 1024
    slot_bytes = 128 if segmented else 64
    backing = torch.empty(alignment + len(all_names) * slot_bytes, dtype=torch.uint8)
    start = -backing.data_ptr() % alignment
    caches = {
        name: backing[start + index * slot_bytes : start + index * slot_bytes + 64].reshape(4, 16, 1).fill_(index + 1)
        for index, name in enumerate(all_names)
    }
    if segmented:
        physical = backing[start : start + slot_bytes].reshape(8, 16, 1).fill_(165)
        caches[all_names[0]] = physical[:, :8, :].fill_(1)

    def close():
        store.resume.set()
        with suppress(RuntimeError):
            worker.shutdown()

    test.addCleanup(close)
    worker.register_kv_caches(caches)
    return worker, store, caches


def metadata(worker, *, load=False, save=True):
    result = AscendConnectorMetadata(set())
    result.add_request(
        ReqMeta(
            "r",
            token_len_chunk=16,
            num_prompt_tokens=32,
            block_hashes=[b"h"],
            block_ids_by_group=[[1] for _ in worker.grouped_block_size],
            kv_cache_group_ids=list(worker.cacheable_group_ids),
            can_save=save,
            load_spec=LoadSpec(0, 16, True) if load else None,
        )
    )
    return result


def forward_layer(worker, meta, layer):
    name = f"model.layers.{layer}.v41_attn"
    worker.wait_for_layer_load(name)
    worker.save_kv_layer(meta, name)


def save_prefix(worker):
    meta = metadata(worker)
    worker.prepare_layerwise_step(meta)
    worker.start_load_kv(meta)
    for layer in range(worker.hf_config.num_hidden_layers):
        forward_layer(worker, meta, layer)
    worker.wait_for_save(meta)
    worker.get_finished(set(), meta)


class TestSharedSourceLayerwise(unittest.TestCase):
    def test_sparse_single_group_keeps_model_layer_slots(self):
        worker, store, caches = make_shared_worker(self, bounded=True)
        self.assertIs(type(worker), LayerwiseKVPoolWorker)
        self.assertEqual(worker.num_layers, 12)
        self.assertFalse(worker.layerwise_offload)
        self.assertEqual(worker._groups_for_layerwise_transfer(2), [(0, 0)])
        self.assertEqual(worker._groups_for_layerwise_transfer(8), [(0, 1)])
        self.assertEqual(worker._groups_for_layerwise_transfer(0), [])
        save_prefix(worker)
        self.assertTrue(store.published)
        self.assertEqual({size for _, size in store.objects.values()}, {64})

    def test_roundtrip_sparse_sources_swa_and_draft(self):
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                worker, store, caches = make_shared_worker(self, bounded=bounded, draft=True)
                expected = {name: cache.clone() for name, cache in caches.items()}
                save_prefix(worker)
                for cache in caches.values():
                    cache.zero_()
                meta = metadata(worker, load=True, save=False)
                worker.prepare_layerwise_step(meta)
                worker.start_load_kv(meta)
                for layer in range(12):
                    forward_layer(worker, meta, layer)
                worker.finish_forward()
                worker.wait_for_save(meta)
                worker.get_finished(set(), meta)
                for name, cache in caches.items():
                    torch.testing.assert_close(cache[1], expected[name][1])
                    self.assertEqual(int(cache[0].sum()), 0)
                self.assertFalse(store.leases)

    def test_cut_cache_only_load_is_idempotent_and_saves_after_final_write(self):
        worker, store, caches = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        store.published.clear()
        store.finish_calls.clear()
        meta = metadata(worker)
        worker.prepare_layerwise_step(meta)
        for layer in range(8):
            forward_layer(worker, meta, layer)
        cut_name = "model.layers.8.v41_attn"
        worker.wait_for_layer_load(cut_name)  # full-batch cache-only boundary
        self.assertEqual(worker.current_layer, 8)
        self.assertNotIn(8, worker._attention_saved_layers)
        worker.wait_for_layer_load(cut_name)  # compact replay of the same layer
        cut = caches["model.layers.8.attn.long_kv_cache"]
        cut[1].fill_(77)
        worker.save_kv_layer(meta, cut_name)
        worker.kv_send_thread.request_queue.join()
        self.assertIn(8, worker._attention_saved_layers)
        self.assertTrue(any(cut[1].data_ptr() in addrs for direction, addrs in store.copies if direction == 0))
        self.assertEqual(worker.current_layer, 9)
        for layer in range(9, 12):
            forward_layer(worker, meta, layer)
        worker.wait_for_save(meta)
        self.assertTrue(store.published)

    def test_failed_load_reaches_consumer_and_never_publishes(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        meta = metadata(worker, load=True, save=False)
        worker.prepare_layerwise_step(meta)
        store.fail_copy = True
        worker.start_load_kv(meta)
        with self.assertRaises(RuntimeError):
            for layer in range(12):
                forward_layer(worker, meta, layer)

    def test_load_uses_adopted_hit_not_unrecomputed_tail(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        meta = metadata(worker, load=True, save=False)
        meta.requests[0].load_spec.kvpool_store_skip_tokens = 32
        worker.prepare_layerwise_step(meta)
        self.assertEqual(meta.requests[0].load_spec.kvpool_store_skip_tokens, 32)
        self.assertEqual(worker.layer_load_tasks[2][0].block_ranges[0].end_block, 1)

    def test_graph_replay_modes_cannot_skip_python_transfer_hooks(self):
        for mode in ("FULL", "FULL_DECODE_ONLY", "PIECEWISE", "FULL_AND_PIECEWISE"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "enforce-eager"):
                make_shared_worker(self, mode=mode)

    def test_missing_target_suffix_is_not_flushed_as_draft(self):
        for fence in ("finish_forward", "wait_for_save"):
            with self.subTest(fence=fence):
                worker, store, _ = make_shared_worker(self, bounded=True, draft=True)
                meta = metadata(worker)
                worker.prepare_layerwise_step(meta)
                for layer in range(8):
                    forward_layer(worker, meta, layer)
                with self.assertRaisesRegex(RuntimeError, "target.*hook"):
                    getattr(worker, fence)(*([meta] if fence == "wait_for_save" else []))
                self.assertFalse(store.published)

    def test_transfer_failure_cannot_be_reset_by_next_bind(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        meta = metadata(worker, load=True, save=False)
        worker.prepare_layerwise_step(meta)
        store.fail_copy = True
        with self.assertRaises(RuntimeError):
            for layer in range(12):
                forward_layer(worker, meta, layer)
        with self.assertRaises(RuntimeError):
            worker.prepare_layerwise_step(metadata(worker))

    def test_shared_prefix_acquires_one_lease_per_step(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        counts = Counter()
        add_lease, remove_lease = store.batch_add_lease, store.batch_remove_lease

        def acquire(keys, ttl):
            counts.update(keys)
            return add_lease(keys, ttl)

        def release(keys):
            counts.subtract(keys)
            return remove_lease(keys)

        store.batch_add_lease, store.batch_remove_lease = acquire, release
        meta = metadata(worker, load=True, save=False)
        other = deepcopy(meta.requests[0])
        other.req_id = "other"
        meta.requests.append(other)
        worker.prepare_layerwise_step(meta)
        self.assertEqual(set(counts.values()), {1})
        for layer in range(12):
            forward_layer(worker, meta, layer)
        worker.finish_forward()
        worker.get_finished(set(), meta)
        self.assertEqual(set(counts.values()), {0})

    def test_incompatible_object_size_fails_before_copy(self):
        worker, store, _ = make_shared_worker(self, bounded=True)
        save_prefix(worker)
        for key, (gva, size) in list(store.objects.items()):
            store.objects[key] = (gva, size - 1)
        store.copies.clear()
        with self.assertRaises(RuntimeError):
            worker.prepare_layerwise_step(metadata(worker, load=True, save=False))
        self.assertFalse(store.copies)

    def test_draft_save_fence_blocks_publication_until_final_copy(self):
        worker, store, caches = make_shared_worker(self, bounded=True, draft=True)
        meta = metadata(worker)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta, layer)
        worker.finish_forward()
        draft = caches["model.mtp.layers.0.attn.kv_cache"]
        draft[1].fill_(99)  # final draft write after target forward
        store.block_address = draft[1].data_ptr()
        store.block_direction = 0
        done = threading.Event()
        errors = []

        def save():
            try:
                worker.wait_for_save(meta)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=save)
        thread.start()
        try:
            self.assertTrue(store.blocked.wait(3))
            self.assertFalse(done.is_set())
            self.assertFalse(store.published)
        finally:
            store.resume.set()
            thread.join(3)
        self.assertTrue(done.is_set())
        self.assertFalse(errors)
        self.assertTrue(store.published)
        worker.get_finished(set(), meta)
        draft.zero_()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta, layer)
        worker.finish_forward()
        torch.testing.assert_close(draft[1], torch.full_like(draft[1], 99))

    def test_failed_draft_save_never_publishes_partial_prefix(self):
        worker, store, _ = make_shared_worker(self, bounded=True, draft=True)
        meta = metadata(worker)
        worker.prepare_layerwise_step(meta)
        for layer in range(12):
            forward_layer(worker, meta, layer)
        worker.finish_forward()
        store.fail_copy = True
        with self.assertRaises(RuntimeError):
            worker.wait_for_save(meta)
        self.assertFalse(store.published)
