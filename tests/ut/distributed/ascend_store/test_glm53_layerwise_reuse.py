# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real shared CPU storage and GVA copies, with NPU events mocked."""

import ctypes
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from tests.ut.distributed.ascend_store.test_glm53_store import make_glm53_caches
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker, start_patch
from tests.ut.models.test_glm5next_cache_config import make_config, make_specs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import create_kv_pool_scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    validate_glm5_next_layerwise_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)


def make_plan(num_layers=45):
    specs = {}
    for offset in range(0, num_layers, 4):
        for name, spec in make_specs(pool=4).items():
            parts = name.split(".")
            index = int(parts[2]) + offset
            if index >= num_layers:
                continue
            parts[2] = str(index)
            specs[".".join(parts)] = replace(spec, mamba_cache_mode="align") if isinstance(spec, MambaSpec) else spec
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, specs)
    return get_glm5_next_kv_cache_config(config, groups, 24 * get_glm5_next_pool_bytes_per_block(groups))


class TestGLM53LayerwiseReuse(unittest.TestCase):
    def worker(self, num_layers=45):
        plan = make_plan(num_layers)
        worker = make_worker(
            self,
            kv_cache_config=plan,
            use_layerwise=True,
            use_mla=True,
            num_layers=num_layers,
            extra_config={"backend": "memcache"},
        )
        config = worker.vllm_config
        config.model_config.enforce_eager = True
        config.speculative_config = None
        config.parallel_config.data_parallel_size = 1
        config.kv_transfer_config.kv_connector = "AscendStoreConnector"
        config.kv_transfer_config.kv_connector_extra_config.update(
            use_layerwise=True,
            layerwise_num_shared_buffers=3,
        )
        return worker, plan

    def test_full_model_allocation_preserves_groups_and_small_backings(self):
        worker, plan = self.worker()
        groups = plan.kv_cache_groups
        old_tensors = list(plan.kv_cache_tensors)
        self.assertEqual(len(old_tensors), 23)
        apply_layerwise_kv_cache_plan(plan, worker.vllm_config)
        self.assertIs(plan.kv_cache_groups, groups)
        self.assertEqual([len(group.layer_names) for group in groups], [22, 11, 12, 11, 11])
        self.assertEqual(plan.kv_cache_tensors[-11:], old_tensors[-11:])
        self.assertEqual(len(plan.kv_cache_tensors), 18)
        self.assertEqual(
            sum(t.size for t in old_tensors) - sum(t.size for t in plan.kv_cache_tensors), 5 * old_tensors[0].size
        )
        caches = make_glm53_caches(plan)
        self.assertEqual(
            caches["model.layers.1.linear_attn"][0].data_ptr(), caches["model.layers.5.linear_attn"][0].data_ptr()
        )
        self.assertNotEqual(
            caches["model.layers.0.linear_attn"][0].data_ptr(), caches["model.layers.4.linear_attn"][0].data_ptr()
        )
        self.assertEqual(set(caches), {name for group in groups for name in group.layer_names})

    def install_store(self, worker):
        buffers, committed = {}, set()

        def alloc(keys, sizes, ttl):
            for key, size in zip(keys, sizes):
                buffers.setdefault(key, ctypes.create_string_buffer(size))
            return [ctypes.addressof(buffers[key]) for key in keys]

        def infos(keys, **kwargs):
            return [
                SimpleNamespace(
                    size=lambda key=key: int(key in committed),
                    gva_list=lambda key=key: [ctypes.addressof(buffers[key])],
                )
                if key in buffers
                else None
                for key in keys
            ]

        def finish(keys, results):
            self.assertEqual(results, [0] * len(keys))
            committed.update(keys)
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

        worker.m_store.batch_alloc.side_effect = alloc
        worker.m_store.batch_is_exist.side_effect = lambda keys: [int(key in committed) for key in keys]
        worker.m_store.batch_get_key_info.side_effect = infos
        worker.m_store.batch_write_finish.side_effect = finish
        worker.m_store.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
        worker.m_store.store.batch_copy.side_effect = copy
        return buffers, committed

    def test_chunk_roundtrip_restores_each_layer_state_before_overwrite(self):
        worker, plan = self.worker(num_layers=17)
        apply_layerwise_kv_cache_plan(plan, worker.vllm_config)
        worker._init_layerwise_config()
        caches = make_glm53_caches(plan)
        buffers, committed = self.install_store(worker)
        worker.register_kv_caches(caches)
        self.addCleanup(worker.kv_send_thread.stop)
        self.addCleanup(worker.kv_recv_thread.stop)
        # IDs are disjoint across groups. Old state slots remain available for
        # the existing per-layer align pre-copy when a chunk crosses a page.
        blocks = [[1, 2, 3], [4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
        hashes = [bytes([i]) * 32 for i in (1, 2, 3)]
        names = {name: gid for gid, group in enumerate(plan.kv_cache_groups) for name in group.layer_names}
        saved_values = {}
        previous = 0
        for step, end in enumerate((137, 700, 1024, 1301), start=1):
            with self.subTest(end=end):
                request = ReqMeta(
                    "chunked",
                    block_ids_by_group=blocks,
                    block_ids_by_group_np=[np.array(ids) for ids in blocks],
                    block_hashes=hashes,
                    target_token_len=end,
                    save_end_token=end // 512 * 512,
                    save_start_token=previous // 512 * 512,
                    can_save=True,
                    num_prompt_tokens=1536,
                    layerwise_request_id="generation-one",
                    load_spec=LoadSpec(previous, previous, True, layerwise_previous_step=True) if previous else None,
                )
                metadata = AscendConnectorMetadata(set())
                metadata.add_request(request)
                worker.prepare_layerwise_step(metadata)
                for layer in range(worker.num_layers):
                    worker.wait_for_layer_load()
                    for name, group_id in names.items():
                        if int(name.split(".")[2]) != layer:
                            continue
                        for component, view in enumerate(caches[name]):
                            if not view.numel():
                                continue
                            value = layer * 4 + component + 1 + step * 40
                            if previous:
                                old_ids = (
                                    [blocks[group_id][(previous - 1) // 512]]
                                    if group_id >= 2
                                    else ([blocks[1][0]] if group_id == 1 else blocks[0][: (previous + 511) // 512])
                                )
                                for block_id in old_ids:
                                    self.assertTrue(
                                        torch.all(view[block_id] == saved_values[name, component, block_id]),
                                        (end, layer, name, block_id),
                                    )
                            new_ids = (
                                [blocks[group_id][(end - 1) // 512]]
                                if group_id >= 2
                                else (
                                    [blocks[1][0]] if group_id == 1 else blocks[0][previous // 512 : (end + 511) // 512]
                                )
                            )
                            # Represent the kernel's post-update conv/SSM and
                            # attention writes. Later layers overwrite aliases.
                            for block_id in new_ids:
                                view[block_id].fill_(value)
                                saved_values[name, component, block_id] = value
                    worker.save_kv_layer(metadata)
                worker.kv_recv_thread.request_queue.join()
                previous = end
        self.assertTrue(buffers)
        self.assertFalse(any("@1@" in key.split(hashes[0].hex())[0] for key in committed if "@partial@" not in key))
        # Ending at 700 never produced a KDA checkpoint at 512.
        for group in (2, 3, 4):
            self.assertNotIn(worker._make_layerwise_full_key(group, hashes[0].hex()), committed)
            self.assertIn(worker._make_layerwise_full_key(group, hashes[1].hex()), committed)
        worker._save_disabled.set()
        worker.current_layer = 0
        with self.assertRaisesRegex(RuntimeError, "incomplete KV/state"):
            worker._check_glm5_next_reuse_errors()
        committed.remove(worker._make_layerwise_partial_key(request, 3, 2, 1301))
        request.load_spec = LoadSpec(1301, 1301, True, layerwise_previous_step=True)
        request.target_token_len = 1400
        with self.assertRaisesRegex(RuntimeError, "incomplete KV/state"):
            worker.prepare_layerwise_step(metadata)

    def test_local_prefix_requires_external_endpoint_and_rebuilds_tail(self):
        worker, plan = self.worker()
        start_patch(self, "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib")
        scheduler = create_kv_pool_scheduler(worker.vllm_config, True, plan)
        request = SimpleNamespace(request_id="hot", prompt_token_ids=[1] * 1536, num_tokens=1536, block_hashes=[])
        scheduler._lookup_external_hit_tokens = MagicMock(return_value=1536)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 512), (512, False))
        self.assertEqual(scheduler.load_specs["hot"].kvpool_store_skip_tokens, 1024)
        scheduler._lookup_external_hit_tokens.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "no complete Memcache restore point"):
            scheduler.get_num_new_matched_tokens(request, 512)

    def test_prefix_probe_checks_the_replay_endpoint_before_accepting_hit(self):
        worker, plan = self.worker()
        start_patch(self, "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib")
        scheduler = create_kv_pool_scheduler(worker.vllm_config, True, plan)
        scheduler._lookup_layerwise_with_coordinator = MagicMock(return_value=512)
        request = SimpleNamespace(request_id="aligned-prompt", num_tokens=1536)
        self.assertEqual(scheduler._get_layerwise_hit_tokens(request, 1536, 0), 512)
        scheduler._lookup_layerwise_with_coordinator.assert_called_once_with(request, 1024, 0)

    def test_snapshot_namespace_changes_on_request_restart(self):
        worker, plan = self.worker()
        start_patch(self, "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib")
        scheduler = create_kv_pool_scheduler(worker.vllm_config, True, plan)
        first = RequestTracker("same-id", 137, allocated_block_ids_by_group=[[1]] * 5)
        second = RequestTracker("same-id", 137, allocated_block_ids_by_group=[[1]] * 5)
        snapshots = [scheduler._build_req_meta(tracker, [], None, [1] * 1000, False) for tracker in (first, second)]
        self.assertNotEqual(snapshots[0].layerwise_request_id, snapshots[1].layerwise_request_id)
        self.assertEqual(snapshots[0].layerwise_request_id, first.layerwise_request_id)

    def test_unsupported_modes_fail_before_allocation(self):
        worker, plan = self.worker()
        worker._init_layerwise_config()
        layout = worker._layerwise_reuse_layout
        for owner, attr, value in (
            (worker.vllm_config, "speculative_config", object()),
            (worker.vllm_config.model_config, "enforce_eager", False),
            (worker.vllm_config.parallel_config, "pipeline_parallel_size", 2),
            (worker.vllm_config.kv_transfer_config, "kv_connector", "MultiConnector"),
        ):
            original = getattr(owner, attr)
            try:
                setattr(owner, attr, value)
                with self.assertRaises(ValueError):
                    validate_glm5_next_layerwise_reuse(worker.vllm_config, layout)
            finally:
                setattr(owner, attr, original)
