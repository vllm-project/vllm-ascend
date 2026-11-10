# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-source scheduling on the existing Memcache GVA data plane."""

from copy import copy
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv_hooks import (
    supports_cross_layer_kv_hooks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerLoadTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.transfer_mode import (
    config_value,
    kv_source_layer_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

CROSS_LAYER_KV_GATE_POLL_INTERVAL_S = 0.1


class CrossLayerKVPoolWorker(LayerwiseKVPoolWorker):
    """Reuse GVA keys, leases, masks, builders and receiver; change submission order.

    Source/SWA tasks prefetch in consumer order. Draft and unknown groups
    restore synchronously before target forward. Save tasks use the ordinary
    GVA sender and group publication, with source-ready events or the final
    step boundary for replay cuts, unmapped groups and speculative writes.
    """

    def __init__(self, vllm_config: VllmConfig, use_layerwise: bool, kv_cache_config: KVCacheConfig | None = None):
        if not vllm_config.use_v2_model_runner:
            raise ValueError("cross_layer_kv requires Model Runner V2")
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        if self.pp_size != 1 or self.pcp_size != 1 or self.dcp_size != 1 or self.use_kvpp:
            raise ValueError("cross_layer_kv currently supports Memcache TP deployments without PP/PCP/DCP/KVPP")
        self.load_async = False
        self._load_started = False
        self._pending_layers: set[int] = set()
        self._step_error: Exception | None = None
        self._pipeline_groups: set[int] = set()
        mode = vllm_config.compilation_config.cudagraph_mode
        mode_name = getattr(mode, "name", str(mode))
        speculative = vllm_config.speculative_config
        dspark = (
            speculative is not None
            and speculative.method == "dspark"
            and getattr(speculative.draft_model_config, "architectures", []) == ["DeepseekV41DSparkModel"]
        )
        self._pipeline_enabled = (
            (vllm_config.model_config.enforce_eager or mode_name in ("NONE", "PIECEWISE"))
            and not getattr(vllm_config.parallel_config, "use_ubatching", False)
            and not getattr(vllm_config.parallel_config, "use_sequence_parallel_moe", False)
            and (speculative is None or dspark)
            and self._extra_config.get("discard_partial_chunks", True) is True
        )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        # Keep draft descriptors registered too; they use synchronous GVA loads.
        BaseKVPoolWorker.register_kv_caches(self, kv_caches)

    def _groups_for_layerwise_transfer(self, local_layer: int) -> list[tuple[int, int]]:
        # Shared-source layouts can have no persistent member at a layer.
        if self.num_kv_cache_groups == 1:
            return [(0, local_layer)] if local_layer < self.group_num_layers.get(0, self.num_layers) else []
        return self.physical_layer_to_group_layers.get(local_layer, [])

    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return
        super()._start_kv_transfer_threads()
        self._pipeline_enabled = self._pipeline_enabled and supports_cross_layer_kv_hooks(
            self.vllm_config.compilation_config.static_forward_context,
            config_value(self.hf_config, "num_hidden_layers", 0),
        )

    def _pipeline_group_ids(self) -> set[int]:
        if not self._pipeline_enabled or self.kv_cache_config is None:
            return set()
        sources = set(kv_source_layer_ids(self.hf_config))
        groups = set()
        for group_id in self.cacheable_group_ids:
            names = self.kv_cache_config.kv_cache_groups[group_id].layer_names
            if not names or any(".mtp." in f".{name}" for name in names):
                continue
            if all(name.endswith(".swa_cache") for name in names) or (
                sources
                and all(name.endswith(".long_kv_cache") or ".indexer.k_cache" in name for name in names)
                and {self._extract_physical_layer_index(name) for name in names} == sources
            ):
                groups.add(group_id)
        return groups

    def prepare_cross_layer_kv_step(self, metadata: AscendConnectorMetadata) -> None:
        # Save-only steps also bind before forward. Drain old H2D writes
        # before replacing block snapshots or completion events.
        self.finish_cross_layer_kv_loads()
        self._load_started = False
        self._pending_layers.clear()
        for event in self.layer_load_finished_events:
            event.clear()
        snapshot_meta = copy(metadata)
        requests = []
        for request in metadata.requests:
            snapshot = copy(request)
            # Only restore the adopted hit, preserving local recomputation of
            # private compressor state when the raw store hit was longer.
            if request.load_spec is not None:
                snapshot.load_spec = copy(request.load_spec)
                snapshot.load_spec.kvpool_store_skip_tokens = snapshot.load_spec.kvpool_cached_tokens
            snapshot.block_ids_by_group_np = [
                np.asarray(blocks, dtype=np.int64) for blocks in request.block_ids_by_group
            ]
            snapshot.block_ids_np = snapshot.block_ids_by_group_np[0]
            requests.append(snapshot)
        snapshot_meta.requests = requests
        try:
            super().prepare_layerwise_step(snapshot_meta)
            if any(
                original.load_spec is not None and original.load_spec.can_load and not snapshot.load_spec.can_load
                for original, snapshot in zip(metadata.requests, requests, strict=True)
            ):
                raise RuntimeError("Cross-layer GVA load preparation failed")
            self._pipeline_groups = self._pipeline_group_ids()
        except Exception as exc:
            self._step_error = exc
            self.finish_cross_layer_kv_loads()

    def start_load_kv(self, metadata: AscendConnectorMetadata) -> None:
        if self._load_started:
            return
        self._load_started = True
        try:
            # Restore unmapped/draft groups before target compute through the
            # same GVA builders and copy routine as the background receiver.
            for layer_id, tasks in enumerate(self.layer_load_tasks):
                synchronous = [task for task in tasks if task.group_id not in self._pipeline_groups]
                if synchronous and not self.kv_recv_thread._copy_gva_layers(LayerLoadTask(None, synchronous, layer_id)):
                    raise RuntimeError("Cross-layer synchronous GVA load failed")
                self.layer_load_tasks[layer_id] = [task for task in tasks if task.group_id in self._pipeline_groups]
            # Bundle requests by their first consumer, rather than draining
            # every dependency of one request before serving another.
            for layer_id, tasks in enumerate(self.layer_load_tasks):
                if tasks:
                    self._pending_layers.add(layer_id)
                    self.kv_recv_thread.add_request(LayerLoadTask(None, tasks, layer_id))
        except Exception as exc:
            self._step_error = exc
            self.finish_cross_layer_kv_loads()

    def wait_for_cross_layer_kv_load(self, layer_idx: int) -> None:
        layer_id = self._global_to_local_layer.get(layer_idx)
        if layer_id not in self._pending_layers:
            return
        try:
            while not self.layer_load_finished_events[layer_id].wait(CROSS_LAYER_KV_GATE_POLL_INTERVAL_S):
                self.kv_recv_thread.raise_if_failed()
                if not self.kv_recv_thread.is_alive():
                    raise RuntimeError("GVA receiver stopped before completing a cross-layer load")
            self.kv_recv_thread.raise_if_failed()
            if self.kv_recv_thread.load_disabled.is_set():
                raise RuntimeError("Cross-layer GVA copy failed")
        except Exception as exc:
            self._step_error = exc
            self.finish_cross_layer_kv_loads()

    def finish_cross_layer_kv_loads(self) -> None:
        receiver = self.kv_recv_thread
        try:
            if receiver is not None:
                self._drain_attention_transfers(drain_recv=True, full=True)
                if receiver.load_disabled.is_set():
                    raise RuntimeError("Cross-layer GVA copy failed")
        except Exception as exc:
            self._step_error = self._step_error or exc
            # Fatal receivers exit without consuming the remaining queue.
            # Join before releasing leases so no target write can outlive them.
            if receiver is not None:
                receiver.stop()
                if receiver.is_alive():
                    receiver.join()
        finally:
            if self._step_load_keys:
                keys = list(self._step_load_keys)
                try:
                    if self.m_store.batch_remove_lease(keys) != 0:
                        raise RuntimeError("Could not release GVA read leases")
                except Exception as exc:
                    self._step_error = self._step_error or exc
                else:
                    self._step_load_keys.clear()
        if self._step_error is not None:
            raise RuntimeError("Cross-layer GVA load failed; this step's state cannot be reused") from self._step_error

    def reset_store(self) -> None:
        self.finish_cross_layer_kv_loads()
        super().reset_store()

    def on_cross_layer_kv_save_trigger(self, trigger_layer: int) -> None:
        # DSpark verification/draft preparation can still change cache pages.
        # The final step boundary is their publication boundary.
        if self.vllm_config.speculative_config is not None or not self.can_put:
            return
        sources = kv_source_layer_ids(self.hf_config)
        if self.vllm_config.cache_config.swa_bounded_replay and sources and trigger_layer == max(sources):
            return
        layer_id = self._global_to_local_layer.get(trigger_layer)
        if layer_id is None or layer_id in self._attention_saved_layers:
            return
        # GVA attaches publication to its final task. Keep it behind every
        # replay-cut/unmapped task that will only be submitted at step end.
        if any(task.write_finish_keys for task in self.layer_save_tasks[layer_id]):
            return
        if all(task.group_id in self._pipeline_groups for task in self.layer_save_tasks[layer_id]):
            self._submit_attention_save(layer_id)

    def wait_for_save(self, metadata: AscendConnectorMetadata) -> None:
        # Reuse GVA events, copy tasks, deferred WRITE_OK and the existing
        # step fence. Missing hooks/replay-cut/draft tasks get final events.
        for layer_id in range(self.num_layers):
            if layer_id not in self._attention_saved_layers:
                self._submit_attention_save(layer_id)
        self._wait_for_final_layer_save(self.num_layers, self.kv_send_thread)

    def shutdown(self) -> None:
        try:
            self.finish_cross_layer_kv_loads()
        finally:
            super().shutdown()
