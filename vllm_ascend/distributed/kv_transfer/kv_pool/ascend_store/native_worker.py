# SPDX-License-Identifier: Apache-2.0
"""Ascend device bindings for the upstream Mooncake Store worker.

The queue, transfer loops, key rotation, pressure handling and save ledger are
owned by upstream. Only device setup and the checked Store boundary live here.
"""

from contextlib import contextmanager
from dataclasses import replace

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    MooncakeStoreWorker,
)
from vllm.distributed.mooncake_store import MooncakeStoreConfig
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
    NativeMooncakeStoreAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeStoreConfig as AscendStoreConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_cache_views import register_native_cache_views


class AscendNativeStoreSendingThread(KVCacheStoreSendingThread):
    def set_finished_request(self, req_id):
        self.store.raise_if_failed()
        super().set_finished_request(req_id)

    def finish_store_job(self, req_meta):
        # Upstream also calls this from finally when a Store operation raises.
        self.store.raise_if_failed()
        super().finish_store_job(req_meta)


class AscendNativeStoreRecvingThread(KVCacheStoreRecvingThread):
    def set_finished_request(self, req_id):
        self.store.raise_if_failed()
        super().set_finished_request(req_id)


class AscendNativeStoreWorker(MooncakeStoreWorker):
    preferred_segment: str | None
    _supports_host_cache_tp_layout = True
    _sending_thread_cls = AscendNativeStoreSendingThread
    _recving_thread_cls = AscendNativeStoreRecvingThread

    @staticmethod
    def _create_mem_pool(extra_config, model_config):
        if extra_config.get("custom_mem_pool"):
            raise ValueError("custom_mem_pool is not supported by the Ascend Mooncake Store worker")
        return None

    def __init__(self, vllm_config, kv_cache_config):
        self._ascend_vllm_config = vllm_config
        super().__init__(vllm_config, kv_cache_config)
        backend = self.store.backend
        # Keep upstream's explicit preferred_segment and group-semantics
        # capability detection. Only fill Ascend's legacy allocation policy.
        self.store_replicate_config.prefer_alloc_in_same_node = backend.config.prefer_alloc_in_same_node
        if self.preferred_segment is None and backend.config.preferred_segment:
            self.preferred_segment = backend.local_seg
            self.store_replicate_config.preferred_segment = backend.local_seg

    def _load_store_config(self) -> MooncakeStoreConfig:
        config = AscendStoreConfig.load_from_env()
        self._ascend_store_config = config
        return MooncakeStoreConfig(
            metadata_server=config.metadata_server,
            master_server_address=config.master_server_address,
            protocol=config.protocol,
            device_name=config.device_name,
            mode=config.mode or ("standalone-store" if config.global_segment_size == 0 else "embedded"),
            global_segment_size=config.global_segment_size,
            local_buffer_size=config.local_buffer_size,
            enable_offload=config.enable_offload or config.enable_ssd_offload,
            tenant_id=config.tenant_id,
        )

    def _setup_store(self, store_config, local_hostname):
        config = self._ascend_vllm_config
        backend = MooncakeBackend(
            config.parallel_config,
            extra_config=config.kv_transfer_config.kv_connector_extra_config,
            store_config=self._ascend_store_config,
        )
        self.store = NativeMooncakeStoreAdapter(backend)

    def _make_store_group_plan(self, group, store_tp_size, attention_layout_cls, replication_factor):
        # Ascend MLA records query-head count separately from its one latent
        # KV head. Only Store planning needs the KV-head interpretation.
        def store_spec(spec):
            return replace(spec, num_heads=spec.num_kv_heads) if isinstance(spec, AscendMLAAttentionSpec) else spec

        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = replace(
                spec, kv_cache_specs={name: store_spec(inner) for name, inner in spec.kv_cache_specs.items()}
            )
        else:
            spec = store_spec(spec)
        return super()._make_store_group_plan(
            replace(group, kv_cache_spec=spec), store_tp_size, attention_layout_cls, replication_factor
        )

    def register_kv_caches(self, kv_caches):
        if self._capacity_only or not kv_caches:
            return
        self._check_completion()
        register_native_cache_views(self, kv_caches)
        self._init_lookup_key_prefixes()
        self._start_transfer_threads()

    @contextmanager
    def _transfer_context(self):
        self._check_completion()
        try:
            self.store.backend.set_device()
        except Exception as exc:
            self.store.fail_transfer(exc)
            raise
        yield

    def _record_save_event(self):
        event = torch.npu.Event()
        event.record()
        return event

    def _check_completion(self):
        self.store.raise_if_failed()

    def start_load_kv(self, metadata):
        self._check_completion()
        return super().start_load_kv(metadata)

    def wait_for_save(self, metadata):
        self._check_completion()
        return super().wait_for_save(metadata)

    def get_finished(self, finished_req_ids, meta):
        self._check_completion()
        return super().get_finished(finished_req_ids, meta)

    def get_block_ids_with_load_errors(self):
        self._check_completion()
        return super().get_block_ids_with_load_errors()

    def build_connector_worker_meta(self):
        self._check_completion()
        return super().build_connector_worker_meta()

    def _close_store(self, store, transfers_pending):
        store.raise_if_failed()
        if transfers_pending:
            raise RuntimeError("Cannot close Ascend Store while transfers are pending")
        store.close()
