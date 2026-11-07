# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    create_kv_pool_scheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv import CrossLayerKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_reuse_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.transfer_mode import KVPoolTransferMode
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
# isort: on


def _config(*, shared=True):
    config = MagicMock()
    config.additional_config = {}
    config.kv_events_config = config.speculative_config = None
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": "memcache",
        "use_layerwise": True,
        "load_async": True,
        "layerwise_num_shared_buffers": 1,
    }
    config.kv_transfer_config.get_from_extra_config.return_value = True
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        setattr(config.parallel_config, name, 1)
    config.parallel_config.world_size = 2
    config.parallel_config.rank = config.parallel_config.data_parallel_rank = 0
    config.cache_config.block_size = config.cache_config.prefix_match_unit = 16
    config.model_config.model = "org/shared-kv-model"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = SimpleNamespace(
        num_hidden_layers=16,
        kv_source_layer_ids=[1, 5, 9] if shared else None,
    )
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 16
    config.model_config.get_layers_start_end_indices.return_value = (0, 16)
    return config


class TestCrossLayerScheduler(unittest.TestCase):
    def _scheduler(self):
        config = _config()
        backend = get_backend_class("memcache")
        with (
            patch.object(backend, "create_scheduler_client"),
            patch.object(backend, "bind_layerwise_keys") as bind_keys,
            patch.object(LayerwiseKVPoolScheduler, "_init_layerwise_offload") as init_offload,
        ):
            scheduler = create_kv_pool_scheduler(config, True)
        bind_keys.assert_not_called()
        init_offload.assert_not_called()
        return scheduler

    def test_layerwise_scheduler_uses_group_lookup_and_synchronous_admission(self):
        scheduler = self._scheduler()
        self.assertIsInstance(scheduler, LayerwiseKVPoolScheduler)
        self.assertIs(scheduler.transfer_mode, KVPoolTransferMode.CROSS_LAYER_KV)
        self.assertTrue(scheduler.use_layerwise)
        self.assertFalse(scheduler.load_async)
        self.assertFalse(scheduler.layerwise_offload)
        scheduler.client = MagicMock(closed=False)
        scheduler.client.lookup.return_value = 48
        request = SimpleNamespace(
            request_id="r", prompt_token_ids=list(range(64)), num_tokens=64, block_hashes=[b"h"] * 4
        )
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 32), (16, False))
        scheduler.client.lookup.assert_called_once_with(64, request.block_hashes, [0], hbm_hit_tokens=32)
        self.assertFalse(scheduler.load_specs["r"].can_load)
        self.assertIsNone(scheduler._running_load_spec(MagicMock(), 32))
        scheduler.client.lookup.return_value = 32
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 32), (0, False))

    def test_pending_saves_pin_blocks_until_every_worker_completes(self):
        scheduler = self._scheduler()
        scheduler.use_hybrid = True
        scheduler.num_speculative_blocks_by_group = {0: 1}
        pool = MagicMock()
        pool.blocks = [object() for _ in range(4)]
        scheduler.bind_gpu_block_pool(pool)
        request = ReqMeta("r", block_ids_by_group=[[0, 1, 2, 3]], block_hashes=[b"h"], can_save=True)
        scheduler.retain_blocks_for_save(request)
        pool.touch.assert_called_once_with(pool.blocks[1:3])
        self.assertTrue(scheduler.has_pending_push_work())
        self.assertFalse(scheduler.reset_store())
        output = SimpleNamespace(
            finished_recving=None, kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata({0: 1})
        )
        scheduler.update_connector_output(output)
        pool.free_blocks.assert_not_called()
        scheduler.update_connector_output(output)
        pool.free_blocks.assert_called_once_with([pool.blocks[2], pool.blocks[1]])
        self.assertFalse(scheduler.has_pending_push_work())
        scheduler._async_loading_req_ids = {"other"}
        scheduler.update_connector_output(SimpleNamespace(finished_recving={"other"}, kv_connector_worker_meta=None))
        self.assertFalse(scheduler._async_loading_req_ids)

    def test_shared_sources_never_enable_physical_buffer_reuse(self):
        config = _config()
        cache = SimpleNamespace(kv_cache_tensors=object())
        original = cache.kv_cache_tensors
        self.assertIsNone(get_layerwise_reuse_config(config.kv_transfer_config, vllm_config=config))
        apply_layerwise_kv_cache_plan(cache, config)
        self.assertIs(cache.kv_cache_tensors, original)
        physical = _config(shared=False)
        self.assertIs(
            get_layerwise_reuse_config(physical.kv_transfer_config, vllm_config=physical),
            physical.kv_transfer_config.kv_connector_extra_config,
        )
        extra = config.kv_transfer_config.kv_connector_extra_config
        config.kv_transfer_config.kv_connector = "MultiConnector"
        config.kv_transfer_config.kv_connector_extra_config = {
            "connectors": [{"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": extra}],
        }
        self.assertIsNone(get_layerwise_reuse_config(config.kv_transfer_config, vllm_config=config))

    def test_public_switch_selects_shared_hooks_and_keeps_save_fence(self):
        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
        worker = MagicMock(spec=CrossLayerKVPoolWorker)
        with (
            patch(module + ".create_kv_pool_worker", return_value=worker),
            patch(module + ".LookupKeyServer") as lookup,
        ):
            connector = AscendStoreConnector(_config(), KVConnectorRole.WORKER)
        self.assertTrue(connector.use_layerwise)
        self.assertTrue(connector.use_cross_layer_kv)
        self.assertFalse(connector.use_classic_layerwise)
        lookup.assert_called_once()
        metadata = AscendConnectorMetadata(set())
        connector.bind_connector_metadata(metadata)
        worker.prepare_cross_layer_kv_step.assert_called_once_with(metadata)
        connector.wait_for_layer_load("model.layers.1.self_attn")
        connector.save_kv_layer("model.layers.1.self_attn", MagicMock(), None)
        connector.wait_for_cross_layer_kv_load(1)
        worker.wait_for_cross_layer_kv_load.assert_called_once_with(1)
        connector.wait_for_save()
        worker.wait_for_save.assert_called_once_with(metadata)
        connector.finish_forward()
        worker.finish_cross_layer_kv_loads.assert_called_once_with()
