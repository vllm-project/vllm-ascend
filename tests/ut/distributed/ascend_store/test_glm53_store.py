# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash pool lifecycle with real cache specs and CPU payloads."""

import ctypes
import queue
import threading
import unittest
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store.test_pool_worker import make_worker, start_patch
from tests.ut.models.test_glm5next_cache_config import make_config, make_specs
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


def make_glm53_plan(layer_offset=0):
    register_all_kvcache_specs(None)
    specs = {}
    for name, spec in make_specs(pool=4).items():
        parts = name.split(".")
        parts[2] = str(int(parts[2]) + layer_offset)
        specs[".".join(parts)] = replace(spec, mamba_cache_mode="align") if isinstance(spec, MambaSpec) else spec
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, specs)
    return get_glm5_next_kv_cache_config(config, groups, 24 * get_glm5_next_pool_bytes_per_block(groups))


def make_glm53_caches(plan):
    # Match the runner's padded, aliased storage and 2 MiB NPU alignment.
    storage = {}
    alignment = 2 * 1024 * 1024
    for tensor in plan.kv_cache_tensors:
        buffer = np.zeros(tensor.size + alignment, dtype=np.uint8)
        offset = -buffer.ctypes.data % alignment
        raw = torch.from_numpy(buffer[offset : offset + tensor.size])
        for name in get_kv_cache_tensor_layers(tensor):
            storage[name] = raw
    caches = {}
    for group in plan.kv_cache_groups:
        specs = group.kv_cache_spec
        specs = (
            specs.kv_cache_specs
            if isinstance(specs, UniformTypeKVCacheSpecs)
            else dict.fromkeys(group.layer_names, specs)
        )
        for name, spec in specs.items():
            raw = storage[name]
            if isinstance(spec, MambaSpec):
                views = []
                offset = 0
                for shape, dtype in zip(spec.shapes, spec.dtypes):
                    view = raw[offset:].view(dtype)
                    strides = torch.empty(shape).stride()
                    views.append(
                        view.as_strided(
                            (plan.num_blocks, *shape), (spec.page_size_bytes // view.element_size(), *strides)
                        )
                    )
                    offset += views[-1][0].numel() * view.element_size()
                caches[name] = tuple(views)
            else:
                shape = (
                    (2, 4, 128)
                    if isinstance(spec, AscendIndexerKPoolTailSpec)
                    else ((128, 1, 128) if "k_cache" in name else (512, 1, 512))
                )
                view = raw.view(spec.dtype)
                strides = torch.empty(shape).stride()
                view = view.as_strided(
                    (plan.num_blocks, *shape), (spec.page_size_bytes // view.element_size(), *strides)
                )
                caches[name] = (view, view[..., :0]) if name.endswith(".attn") else (view,)
    return caches


class TestGLM53Store(unittest.TestCase):
    def test_hash_geometry_and_safe_full_hit(self):
        plan = make_glm53_plan()
        for prefix_unit in (None, 128, 512):
            with self.subTest(prefix_unit=prefix_unit):
                worker = make_worker(
                    self, kv_cache_config=plan, prefix_match_unit=prefix_unit, use_mla=True, num_layers=4
                )
                start_patch(self, "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib")
                scheduler = KVPoolScheduler(worker.vllm_config, False, plan)
                self.assertEqual(worker.hash_block_size, prefix_unit or 512)
                self.assertEqual(scheduler.hash_block_size, worker.hash_block_size)
                self.assertEqual(worker.cacheable_group_ids, [0, 2, 3, 4])
                self.assertEqual(scheduler.kv_cache_group_ids, [0, 1, 2, 3, 4])
                self.assertEqual(worker.cache_transfer_granularity, 512)
                for prompt_len, hit, expected in ((512, 512, 0), (1024, 1024, 512), (1025, 1024, 1024)):
                    scheduler.client = MagicMock()
                    scheduler.client.lookup.return_value = hit
                    request = SimpleNamespace(
                        request_id="hit", prompt_token_ids=[1] * prompt_len, num_tokens=prompt_len, block_hashes=[]
                    )
                    self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (expected, False))
                self.doCleanups()

    def test_nonlayerwise_round_trip_keeps_each_tp_state_and_private_tail(self):
        plan = make_glm53_plan()
        hashes = [bytes([i]) * 32 for i in (1, 2)]
        source = [[1, 2], [3], [4, 5], [6, 7], [8, 9]]
        target = [[10, 11], [12], [13, 14], [15, 16], [17, 18]]
        for tp_size, load_async in ((1, False), (1, True), (2, False), (2, True)):
            with self.subTest(tp_size=tp_size, load_async=load_async):
                stored: dict[str, list[bytes]] = {}
                workers = []

                def put(keys, addresses, sizes, stored=stored):
                    for key, row, lengths in zip(keys, addresses, sizes):
                        stored[key] = [ctypes.string_at(address, length) for address, length in zip(row, lengths)]
                    return [0] * len(keys)

                def get(keys, addresses, sizes, stored=stored):
                    for key, row, lengths in zip(keys, addresses, sizes):
                        for address, length, payload in zip(row, lengths, stored[key]):
                            self.assertEqual(length, len(payload))
                            ctypes.memmove(address, payload, length)
                    return [0] * len(keys)

                for rank in range(tp_size):
                    worker = make_worker(
                        self,
                        kv_cache_config=plan,
                        use_mla=True,
                        num_layers=4,
                        tp_size=tp_size,
                        tp_rank=rank,
                        extra_config={"load_async": load_async},
                    )
                    caches = make_glm53_caches(plan)
                    worker.m_store.requires_exists_before_put = False
                    worker.m_store.put.side_effect = put
                    worker.m_store.get.side_effect = get
                    worker.m_store.exists.side_effect = lambda keys, stored=stored: [int(key in stored) for key in keys]
                    worker.register_kv_caches(caches)
                    self.assertEqual(worker.kv_send_thread.block_size, [512, 4, 512, 512, 512])
                    self.assertTrue(
                        all(address > 0 for row in worker.group_kv_caches_base_addr.values() for address in row)
                    )
                    self.assertEqual(worker.group_block_len[0], [512 * 512 * 2, 128 * 128 * 2])
                    for group_id, group in enumerate(plan.kv_cache_groups):
                        for name in group.layer_names:
                            for cache in caches[name]:
                                cache[source[group_id]] = 10 + group_id + (rank if group_id >= 2 else 0)
                                cache[target[group_id]] = -1
                    metadata = AscendConnectorMetadata(set())
                    metadata.add_request(
                        ReqMeta(
                            "save",
                            token_len_chunk=1024,
                            block_ids_by_group=source,
                            block_hashes=hashes,
                            kv_cache_group_ids=list(range(5)),
                            can_save=True,
                        )
                    )
                    worker.wait_for_save(metadata)
                    worker.wait_for_previous_save()
                    workers.append((worker, caches))
                    self.doCleanups()

                self.assertTrue(stored)
                self.assertFalse(any("@group:1@" in key for key in stored))
                mla_keys = [key for key in stored if "@group:0@" in key]
                self.assertEqual(len(mla_keys), len(hashes))
                self.assertTrue(all("@head_or_tp_rank:0@" in key for key in mla_keys))
                for worker, caches in workers:
                    self.assertEqual(worker.lookup_scheduler(1024, hashes, list(range(5))), 1024)
                    metadata = AscendConnectorMetadata(set())
                    metadata.add_request(
                        ReqMeta(
                            "load",
                            token_len_chunk=1024,
                            block_ids_by_group=target,
                            block_hashes=hashes,
                            kv_cache_group_ids=list(range(5)),
                            load_spec=LoadSpec(0, 1024, True),
                        )
                    )
                    worker.start_load_kv(metadata)
                    if load_async:
                        worker.kv_recv_thread.request_queue.join()
                    self.assertTrue(worker.m_store.get.called)
                    masks = worker.token_database.load_mask(hashes, 1024)
                    for group_id, group in enumerate(plan.kv_cache_groups):
                        for name in group.layer_names:
                            for cache in caches[name]:
                                if group_id == 1:
                                    self.assertTrue(torch.all(cache[target[group_id]] == -1))
                                else:
                                    for block_idx, allowed in enumerate(masks[group_id]):
                                        if allowed:
                                            torch.testing.assert_close(
                                                cache[target[group_id][block_idx]],
                                                cache[source[group_id][block_idx]],
                                            )
                for rank in range(tp_size):
                    missing = next(
                        key
                        for key in stored
                        if "@group:4@" in key and f"@head_or_tp_rank:{rank}@" in key and key.endswith(hashes[-1].hex())
                    )
                    payload = stored.pop(missing)
                    self.assertLess(workers[0][0].lookup_scheduler(1024, hashes, list(range(5))), 1024)
                    stored[missing] = payload

    def test_memcache_layerwise_round_trip_keeps_each_tp_state(self):
        plan = make_glm53_plan()
        block_hash = bytes([1]) * 32
        buffers = {}
        readable = set()

        def alloc(keys, sizes, ttl):
            for key, size in zip(keys, sizes):
                self.assertNotIn(key, buffers)
                buffers[key] = ctypes.create_string_buffer(size)
            return [ctypes.addressof(buffers[key]) for key in keys]

        def infos(keys, **kwargs):
            return [
                SimpleNamespace(
                    size=lambda key=key: int(key in readable),
                    gva_list=lambda key=key: [ctypes.addressof(buffers[key])],
                )
                for key in keys
            ]

        def finish(keys, results):
            self.assertEqual(results, [0] * len(keys))
            readable.update(keys)
            return results

        def copy(gvas, addresses, sizes, direction):
            for gva, address, size in zip(gvas, addresses, sizes):
                self.assertTrue(
                    any(
                        ctypes.addressof(buf) <= gva and gva + size <= ctypes.addressof(buf) + len(buf)
                        for buf in buffers.values()
                    )
                )
                ctypes.memmove(gva if direction == 0 else address, address if direction == 0 else gva, size)
            return 0

        source = [[1], [2], [3], [4], [5]]
        target = [[6], [7], [8], [9], [10]]
        workers = []
        for rank, pp_rank in ((0, 0), (1, 0), (0, 1), (1, 1)):
            plan = make_glm53_plan(layer_offset=4 * pp_rank)
            worker = make_worker(
                self,
                kv_cache_config=plan,
                use_layerwise=True,
                use_mla=True,
                num_layers=4,
                tp_size=2,
                tp_rank=rank,
                pp_size=2,
                pp_rank=pp_rank,
                layer_offset=4 * pp_rank,
                extra_config={"backend": "memcache"},
            )
            worker.m_store.batch_alloc.side_effect = alloc
            worker.m_store.batch_get_key_info.side_effect = infos
            worker.m_store.batch_write_finish.side_effect = finish
            worker.m_store.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
            worker.m_store.store.batch_copy.side_effect = copy
            caches = make_glm53_caches(plan)
            worker.register_kv_caches(caches)
            for group_id, group in enumerate(plan.kv_cache_groups):
                for name in group.layer_names:
                    for cache in caches[name]:
                        cache[source[group_id]] = 10 + group_id + (rank if group_id >= 2 else 0)
                        cache[target[group_id]] = -1
            metadata = AscendConnectorMetadata(set())
            request = ReqMeta(
                "save",
                token_len_chunk=512,
                block_ids_by_group=source,
                block_ids_by_group_np=[np.asarray(ids) for ids in source],
                block_hashes=[block_hash],
                kv_cache_group_ids=list(range(5)),
                can_save=True,
            )
            metadata.add_request(request)
            worker.start_load_kv(metadata)
            task_groups = {task.group_id for tasks in worker.layer_save_tasks for task in tasks}
            self.assertEqual(task_groups, {0, 2, 3, 4} if rank == 0 else {2, 3, 4})
            self.assertEqual(len(request.block_gvas_by_group_np), 5)
            for _ in range(4):
                worker.wait_for_layer_load()
                worker.save_kv_layer(metadata)
            worker.kv_send_thread.request_queue.join()
            worker.kv_send_thread.raise_if_failed()
            self.assertEqual(worker.current_layer, 4)
            workers.append((worker, caches))
            self.doCleanups()

        self.assertEqual(len(readable), 14)  # Per PP stage: one MLA shard and three KDA groups on both TP ranks.
        self.assertFalse(any("@1@" in key.split(block_hash.hex())[0] for key in readable))
        start_patch(self, "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib")
        scheduler = KVPoolScheduler(workers[0][0].vllm_config, True, workers[0][0].kv_cache_config)
        scheduler.store_scheduler.batch_is_readable.side_effect = lambda keys: [key in readable for key in keys]
        hit_request = SimpleNamespace(
            request_id="hit", prompt_token_ids=[1] * 513, num_tokens=513, block_hashes=[block_hash]
        )
        self.assertEqual(scheduler.get_num_new_matched_tokens(hit_request, 0), (512, False))
        self.assertEqual(len(scheduler._make_layerwise_hit_check_keys(2, block_hash.hex())), 4)

        for worker, caches in workers:
            plan = worker.kv_cache_config
            metadata = AscendConnectorMetadata(set())
            request = ReqMeta(
                "load",
                token_len_chunk=512,
                block_ids_by_group=target,
                block_ids_by_group_np=[np.asarray(ids) for ids in target],
                block_hashes=[block_hash],
                kv_cache_group_ids=list(range(5)),
                load_spec=LoadSpec(0, 512, True),
                can_save=False,
            )
            metadata.add_request(request)
            worker.start_load_kv(metadata)
            self.assertEqual(len(request.load_block_gvas_by_group_np), 5)
            self.assertEqual(len(request.load_block_gvas_by_group_np[1]), 0)
            for _ in range(4):
                worker.wait_for_layer_load()
                worker.save_kv_layer(metadata)
            worker.kv_recv_thread.request_queue.join()
            worker.kv_recv_thread.raise_if_failed()
            for group_id, group in enumerate(plan.kv_cache_groups):
                for name in group.layer_names:
                    for cache in caches[name]:
                        if group_id == 1:
                            self.assertTrue(torch.all(cache[target[group_id]] == -1))
                        else:
                            torch.testing.assert_close(cache[target[group_id]], cache[source[group_id]])
        readable.remove(workers[1][0]._make_layerwise_full_key(4, block_hash.hex()))
        hit_request.request_id = "missing"
        self.assertEqual(scheduler.get_num_new_matched_tokens(hit_request, 0), (0, False))

    def test_mooncake_layerwise_still_rejects_hybrid(self):
        with self.assertRaisesRegex(ValueError, "AscendStore private KV state requires non-layerwise transfer"):
            make_worker(self, kv_cache_config=make_glm53_plan(), use_layerwise=True, use_mla=True)

    def test_empty_final_layer_waits_for_pending_save_before_reusing_events(self):
        pending = queue.Queue()
        pending.put("earlier-layer-save")
        entered = threading.Event()
        finished = threading.Event()
        errors = []
        events = [threading.Event(), threading.Event()]
        worker = SimpleNamespace(
            use_block_key_layerwise=False,
            num_layers=2,
            current_layer=1,
            sync_save_events=[MagicMock(), MagicMock()],
            layer_save_finished_events=events,
            layer_save_tasks=[[], []],
            prefetch_layer_map={},
            kv_send_thread=SimpleNamespace(request_queue=pending, raise_if_failed=MagicMock()),
        )
        worker._wait_for_final_layer_save = MethodType(KVPoolWorker._wait_for_final_layer_save, worker)
        worker.sync_save_events[1].record.side_effect = entered.set

        def save():
            try:
                KVPoolWorker.save_kv_layer(worker, AscendConnectorMetadata(set()))
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        caller = threading.Thread(target=save, daemon=True)
        caller.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertFalse(finished.wait(timeout=0.1))
        finally:
            events[0].set()
            pending.task_done()
            caller.join(timeout=2)
        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(any(event.is_set() for event in events))
        self.assertEqual(worker.current_layer, 2)

    def test_empty_final_layer_propagates_pending_transfer_failure(self):
        pending = queue.Queue()
        pending.put("failed-save")
        worker = SimpleNamespace(
            use_block_key_layerwise=False,
            num_layers=1,
            current_layer=0,
            sync_save_events=[MagicMock()],
            layer_save_finished_events=[threading.Event()],
            layer_save_tasks=[[]],
            prefetch_layer_map={},
            kv_send_thread=SimpleNamespace(
                request_queue=pending,
                raise_if_failed=MagicMock(side_effect=[None, RuntimeError("save failed")]),
            ),
        )
        worker._wait_for_final_layer_save = MethodType(KVPoolWorker._wait_for_final_layer_save, worker)
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            KVPoolWorker.save_kv_layer(worker, AscendConnectorMetadata(set()))

    def test_layerwise_null_kda_slots_never_publish_readable_keys(self):
        plan = make_glm53_plan()
        worker = make_worker(
            self,
            kv_cache_config=plan,
            use_layerwise=True,
            use_mla=True,
            num_layers=4,
            extra_config={"backend": "memcache"},
        )
        worker.register_kv_caches(make_glm53_caches(plan))
        worker.m_store.batch_alloc.side_effect = lambda keys, sizes, ttl: [100000 + i * 4096 for i in range(len(keys))]
        hashes = [bytes([i]) * 32 for i in (1, 2, 3)]
        blocks = [[1, 2, 3], [4], [0, 5, 0], [6, 0, 0], [0, 0, 7]]
        request = ReqMeta(
            "null-kda-slots",
            token_len_chunk=1536,
            block_ids_by_group=blocks,
            block_ids_by_group_np=[np.asarray(ids) for ids in blocks],
            block_hashes=hashes,
            kv_cache_group_ids=list(range(5)),
            can_save=True,
        )
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(request)
        worker.start_load_kv(metadata)
        expected = {2: [False, True, False], 3: [True, False, False], 4: [False, False, True]}
        for group_id, mask in expected.items():
            self.assertEqual(request.store_masks[group_id], mask)
            for index, valid in enumerate(mask):
                key = worker._make_layerwise_full_key(group_id, hashes[index].hex())
                self.assertEqual(key in worker._allocated_gvas, valid)
        for tasks in worker.layer_save_tasks:
            for task in tasks:
                if task.group_id in expected:
                    for block_range in task.block_ranges:
                        self.assertTrue(all(expected[task.group_id][block_range.start_block : block_range.end_block]))
