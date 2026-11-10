# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise shared-source scheduling through the real layerwise GVA data plane."""

import ctypes
import threading
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from tests.ut.distributed.ascend_store import test_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import create_kv_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv import CrossLayerKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
)
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
        if direction == 1 and self.block_address in addrs:
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


def make_cross_worker(
    test,
    *,
    bounded=False,
    draft=False,
    mode="NONE",
    sources=(2, 8),
    num_layers=12,
    method=None,
    architecture="DeepseekV41DSparkModel",
    use_v2=True,
    segmented=False,
):
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
        num_blocks=4,
        prefix_cache_retention_interval=0,
    )

    def create(config, use_layerwise, kv_cache_config):
        config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        config.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=num_layers, kv_source_layer_ids=list(sources), compress_ratios=[2] * num_layers
        )
        config.model_config.enforce_eager = False
        config.compilation_config.cudagraph_mode = SimpleNamespace(name=mode)
        config.compilation_config.static_forward_context = {
            str(layer): SimpleNamespace(layer_idx=layer, supports_cross_layer_kv_transfer_hooks=True)
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
            extra_config={"backend": "memcache", "layerwise_num_shared_buffers": 1},
            kv_cache_config=cache_config,
        )
    store = MemoryGVAStore()
    worker.m_store = store
    worker.cache_coordinator = worker.token_database.cache_coordinator = None
    all_names = sum(groups, [])
    alignment = 2 * 1024 * 1024
    backing = torch.empty(alignment + len(all_names) * 64, dtype=torch.uint8)
    start = -backing.data_ptr() % alignment
    caches = {
        name: backing[start + index * 64 : start + (index + 1) * 64].reshape(4, 16, 1).fill_(index + 1)
        for index, name in enumerate(all_names)
    }
    if segmented:
        caches[all_names[0]] = caches[all_names[0]].reshape(8, 8, 1)

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


def save_prefix(worker):
    meta = metadata(worker)
    worker.prepare_cross_layer_kv_step(meta)
    for layer in range(worker.vllm_config.model_config.hf_text_config.num_hidden_layers):
        worker.on_cross_layer_kv_save_trigger(layer)
    worker.wait_for_save(meta)


class TestCrossLayerGVA(unittest.TestCase):
    def test_multiple_requests_share_first_consumer_batches(self):
        worker, store, caches = make_cross_worker(self, bounded=True)
        meta = metadata(worker)
        meta.add_request(
            ReqMeta(
                "other",
                token_len_chunk=16,
                num_prompt_tokens=32,
                block_hashes=[b"j"],
                block_ids_by_group=[[2]],
                kv_cache_group_ids=[0],
                can_save=True,
            )
        )
        for cache in caches.values():
            cache[2].fill_(47)
        worker.prepare_cross_layer_kv_step(meta)
        worker.wait_for_save(meta)
        for cache in caches.values():
            cache.zero_()
        for request in meta.requests:
            request.can_save = False
            request.load_spec = LoadSpec(0, 16, True)
        store.block_address = caches["model.layers.8.attn.long_kv_cache"][1].data_ptr()
        worker.prepare_cross_layer_kv_step(meta)
        worker.start_load_kv(meta)
        self.assertTrue(store.blocked.wait(timeout=2))
        worker.wait_for_cross_layer_kv_load(2)
        self.assertEqual(int(caches["model.layers.2.attn.long_kv_cache"][2].sum()), 47 * 16)
        store.resume.set()
        worker.finish_cross_layer_kv_loads()
        self.assertFalse(store.leases)

    def test_changed_sources_and_segmented_descriptors_roundtrip(self):
        worker, store, caches = make_cross_worker(self, sources=(1, 5, 9), num_layers=16, segmented=True)
        expected = {name: cache.flatten()[16:32].clone() for name, cache in caches.items()}
        save_prefix(worker)
        for cache in caches.values():
            cache.zero_()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_kv_step(meta)
        worker.start_load_kv(meta)
        for layer in range(16):
            worker.wait_for_cross_layer_kv_load(layer)
        worker.finish_cross_layer_kv_loads()
        for name, cache in caches.items():
            self.assertTrue(torch.equal(cache.flatten()[16:32], expected[name]), name)
        self.assertFalse(store.leases)

    def test_roundtrip_reuses_gva_protocol_builders_and_threads(self):
        worker, store, caches = make_cross_worker(self)
        self.assertIsInstance(worker, CrossLayerKVPoolWorker)
        self.assertIs(type(worker.kv_recv_thread), KVCacheStoreLayerRecvingThread)
        self.assertIs(type(worker.kv_send_thread), KVCacheStoreLayerSendingThread)
        self.assertFalse(worker.layerwise_offload)
        expected = {name: value[1].clone() for name, value in caches.items()}
        save_prefix(worker)
        self.assertTrue(store.published)
        self.assertTrue(all("@group:" not in key and "@head_or_tp_rank:" not in key for key in store.published))
        for group in worker.cacheable_group_ids:
            self.assertIn(worker._make_layerwise_full_key(group, b"h".hex()), store.published)
        for value in caches.values():
            value.zero_()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_kv_step(meta)
        worker.start_load_kv(meta)
        for layer in range(12):
            worker.wait_for_cross_layer_kv_load(layer)
        worker.finish_cross_layer_kv_loads()
        self.assertFalse(store.leases)
        for name, value in caches.items():
            self.assertTrue(torch.equal(value[1], expected[name]), name)
            self.assertEqual(int(value[0].sum()), 0)

    def test_early_consumer_runs_while_later_source_is_copying(self):
        worker, store, caches = make_cross_worker(self, bounded=True)
        save_prefix(worker)
        store.block_address = caches["model.layers.8.attn.long_kv_cache"][1].data_ptr()
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_kv_step(meta)
        worker.start_load_kv(meta)
        self.assertTrue(store.blocked.wait(timeout=2))
        worker.wait_for_cross_layer_kv_load(2)
        self.assertFalse(worker.layer_load_finished_events[worker._global_to_local_layer[8]].is_set())
        store.resume.set()
        worker.finish_cross_layer_kv_loads()

    def test_replay_cut_and_publication_wait_for_step_end(self):
        worker, store, _ = make_cross_worker(self, bounded=True)
        meta = metadata(worker)
        worker.prepare_cross_layer_kv_step(meta)
        worker.on_cross_layer_kv_save_trigger(2)
        worker.kv_send_thread.request_queue.join()
        self.assertTrue(store.copies)
        self.assertFalse(store.published)
        worker.on_cross_layer_kv_save_trigger(8)
        self.assertFalse(store.published)
        worker.wait_for_save(meta)
        self.assertTrue(store.published)
        self.assertTrue(all(result == 0 for _, results in store.finish_calls for result in results))

    def test_copy_failure_survives_error_notification_collection(self):
        worker, store, _ = make_cross_worker(self, bounded=True)
        save_prefix(worker)
        meta = metadata(worker, load=True, save=False)
        worker.prepare_cross_layer_kv_step(meta)
        store.fail_copy = True
        worker.start_load_kv(meta)
        with self.assertRaisesRegex(RuntimeError, "state cannot be reused"):
            worker.wait_for_cross_layer_kv_load(2)
        worker.get_failed_requests()
        with self.assertRaisesRegex(RuntimeError, "state cannot be reused"):
            worker.prepare_cross_layer_kv_step(metadata(worker))
        self.assertFalse(store.leases)

    def test_missing_or_wrong_sized_objects_release_read_leases(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                worker, store, _ = make_cross_worker(self)
                save_prefix(worker)
                key = worker._make_layerwise_full_key(1, b"h".hex())
                if missing:
                    store.published.remove(key)
                else:
                    address, size = store.objects[key]
                    store.objects[key] = (address, size + 1)
                with self.assertRaisesRegex(RuntimeError, "state cannot be reused"):
                    worker.prepare_cross_layer_kv_step(metadata(worker, load=True, save=False))
                self.assertFalse(store.leases)

    def test_failed_save_does_not_publish_or_delete_shared_keys(self):
        worker, store, _ = make_cross_worker(self, bounded=True)
        meta = metadata(worker)
        worker.prepare_cross_layer_kv_step(meta)
        store.fail_copy = True
        worker.wait_for_save(meta)
        self.assertFalse(store.published)
        self.assertFalse(store.finish_calls)

    def test_uncertain_publication_never_compensates_with_write_fail(self):
        worker, store, _ = make_cross_worker(self, bounded=True)
        publish = store.batch_write_finish

        def lost_reply(keys, results):
            publish(keys, results)
            raise RuntimeError("publication reply lost")

        store.batch_write_finish = lost_reply
        meta = metadata(worker)
        worker.prepare_cross_layer_kv_step(meta)
        with self.assertRaises(RuntimeError):
            worker.wait_for_save(meta)
        self.assertTrue(store.published)
        self.assertTrue(all(result == 0 for _, results in store.finish_calls for result in results))
