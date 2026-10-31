# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compose a paired sparse-prefix cache with the existing PD transport.

vLLM's offloader owns hashes, admission, LRU, block fences and publication.
Sparse KV offload continues to own the live Host main KV and Hot buffers.
The local cache restores a prefix; PD fills the remaining prompt. Neither
completion alone makes that request ready to decode.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiKVConnectorMetadata
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector import SfaRemoteD2HConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import get_external_request_id
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import AscendOffloadingConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass
class SparsePrefixMetadata(MultiKVConnectorMetadata):
    paired_requests: tuple[str, ...] = ()
    remote_requests: tuple[str, ...] = ()


class SfaPrefixOffloadingConnector(AscendMultiConnector):
    """CPU prefix cache for sparse KV offload, followed by remote PD fallback."""

    @classmethod
    def _get_connector_classes_and_configs(cls, vllm_config: VllmConfig):
        transfer = vllm_config.kv_transfer_config
        if transfer is None or not transfer.is_kv_consumer or transfer.is_kv_producer:
            raise ValueError("SfaPrefixOffloadingConnector is Decode-consumer only")
        sparse = (vllm_config.additional_config or {}).get("sparse_kv_offload_config", {})
        if not sparse.get("enabled", False) or sparse.get("keep_device_kv_cache", False):
            raise ValueError("Sparse prefix caching requires sparse_kv_offload with keep_device_kv_cache=False")
        extra = dict(transfer.kv_connector_extra_config)
        extra.pop("connectors", None)
        prefix = dict(extra.pop("prefix_cache", {}))
        if int(prefix.get("cpu_bytes_to_use", 0)) <= 0:
            raise ValueError("prefix_cache.cpu_bytes_to_use must be positive")
        if prefix.get("spec_name", "CPUOffloadingSpec") != "CPUOffloadingSpec" or prefix.get("canonical_layout", False):
            raise ValueError("Sparse prefix caching currently requires CPUOffloadingSpec without canonical_layout")
        if prefix.get("secondary_tiers"):
            raise ValueError("Sparse prefix caching does not support secondary tiers")
        if prefix.get("spec_module_path"):
            raise ValueError("Sparse prefix caching requires the built-in CPU offloading spec")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Sparse prefix caching requires --enable-prefix-caching")
        if transfer.kv_load_failure_policy != "fail":
            raise ValueError(
                "Sparse prefix caching requires kv_load_failure_policy='fail'; local Prefill is unavailable"
            )
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.method != "mtp":
            raise ValueError("Sparse prefix caching supports ordinary decode and MTP only")
        # Child order also keeps the existing PD registration/lifecycle first.
        result = []
        for connector, role, child_extra in (
            (SfaRemoteD2HConnector, "kv_consumer", extra),
            (AscendOffloadingConnector, "kv_both", {**prefix, "sparse_prefix_cache": True}),
        ):
            config = copy(vllm_config)
            config.kv_transfer_config = copy(transfer)
            config.kv_transfer_config.kv_connector = (
                "SfaRemoteD2HConnector" if connector is SfaRemoteD2HConnector else "OffloadingConnector"
            )
            config.kv_transfer_config.kv_role = role
            config.kv_transfer_config.kv_connector_module_path = None
            config.kv_transfer_config.kv_connector_extra_config = child_extra
            result.append((connector, config))
        return result

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig):
        # MultiConnector also validates HMA against the declared child configs.
        # Populate that declaration without mutating the engine configuration.
        config = copy(vllm_config)
        config.kv_transfer_config = copy(vllm_config.kv_transfer_config)
        config.kv_transfer_config.kv_connector_extra_config = {
            **vllm_config.kv_transfer_config.kv_connector_extra_config,
            "connectors": [
                {"kv_connector": "SfaRemoteD2HConnector", "kv_role": "kv_consumer"},
                {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"},
            ],
        }
        super().__init__(config, role, kv_cache_config)
        self._prefix_matches: dict[str, tuple[int, int]] = {}
        self._prefix_hits: dict[str, int] = {}
        self._new_pairs: set[str] = set()
        self._new_receives: set[str] = set()
        self._receives: dict[str, set[int]] = {}
        self._expected: dict[str, set[int]] = {}
        self._cancelled: set[str] = set()
        self._failed: set[str] = set()
        self._blocked_errors: dict[str, set[int]] = {}
        self._ready_errors: set[int] = set()
        self._scheduler_pending: set[str] = set()
        if role == KVConnectorRole.SCHEDULER:
            self._connectors[0].connector_scheduler.prefix_cache_enabled = True

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        cached, _ = self._connectors[1].get_num_new_matched_tokens(request, num_computed_tokens)
        if cached is None:
            return None, False
        remote, asynchronous = self._connectors[0].get_num_new_matched_tokens(request, num_computed_tokens)
        if remote is None:
            return None, False
        if not remote:
            # Sparse offload has no general local-prefill path. A local cache
            # hit must not bypass the PD protocol for the uncached suffix.
            return 0, False
        self._prefix_matches[request.request_id] = (num_computed_tokens, min(cached, remote))
        self._requests_to_connector[request.request_id] = 0
        return remote, asynchronous

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        # request.num_computed_tokens is updated by core only AFTER this hook.
        # The lookup boundary already includes the newly adopted local hit.
        local, cached = self._prefix_matches.pop(request.request_id, (request.num_computed_tokens, 0))
        if not num_external_tokens:
            cached = 0
        self._prefix_hits[request.request_id] = cached
        self._connectors[1].update_state_after_alloc(request, blocks, cached)
        pd = self._connectors[0].connector_scheduler
        if num_external_tokens and local + cached:
            pd.prefix_cache_tokens[request.request_id] = local + cached
        if cached:
            self._new_pairs.add(request.request_id)
        if num_external_tokens:
            self._scheduler_pending.add(request.request_id)
            self._new_receives.add(request.request_id)
        self._connectors[0].update_state_after_alloc(request, blocks, num_external_tokens)

    def get_external_cache_hit_sources(self, request: Request, num_external_tokens: int) -> dict[CacheHitSource, int]:
        cached = min(self._prefix_hits.get(request.request_id, 0), num_external_tokens)
        sources = {CacheHitSource.HOST: cached} if cached else {}
        if num_external_tokens > cached:
            sources.update(self._connectors[0].get_external_cache_hit_sources(request, num_external_tokens - cached))
        return sources

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> SparsePrefixMetadata:
        metadata = super().build_connector_meta(scheduler_output)
        pairs = tuple(self._new_pairs)
        receives = tuple(self._new_receives)
        self._new_pairs.clear()
        self._new_receives.clear()
        return SparsePrefixMetadata(metadata.metadata, metadata.extra_async_saves, pairs, receives)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, SparsePrefixMetadata)
        super().bind_connector_metadata(connector_metadata)
        for request_id in connector_metadata.remote_requests:
            self._receives.setdefault(request_id, set())
            self._expected[request_id] = {0, 1} if request_id in connector_metadata.paired_requests else {0}

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        self._cancelled.update(finished_req_ids & self._receives.keys())
        # Do not drop PD DMA destinations while either load is in flight.
        cleanup = finished_req_ids - self._receives.keys()
        output = KVConnectorTransferResults()
        completed = set()
        for index, connector in enumerate(self._connectors):
            result = connector.get_transfer_results(cleanup)
            for request_id in result.finished_sending:
                pending = self._extra_async_saves.get(request_id, 0)
                if not pending:
                    output.finished_sending.add(request_id)
                elif pending == 1:
                    del self._extra_async_saves[request_id]
                else:
                    self._extra_async_saves[request_id] = pending - 1
            self._failed.update(result.failed_recving & self._receives.keys())
            output.failed_recving.update(result.failed_recving - self._receives.keys())
            for request_id in result.finished_recving:
                if request_id in self._receives:
                    self._receives[request_id].add(index)
                else:
                    output.finished_recving.add(request_id)
            errors = connector.get_block_ids_with_load_errors()
            for request_id in self._receives if errors else ():
                destinations = self._connectors[0].connector_worker._dest_blocks_by_req.get(
                    get_external_request_id(request_id), ((), ())
                )
                affected = errors.intersection(block for group in destinations for block in group)
                if affected:
                    self._blocked_errors.setdefault(request_id, set()).update(affected)
                    self._failed.add(request_id)
                    errors.difference_update(affected)
            self._ready_errors.update(errors)
        if self._receives:
            local_completed = {
                request_id for request_id, arrivals in self._receives.items() if arrivals >= self._expected[request_id]
            }
            pd_worker = self._connectors[0].connector_worker
            # TP0 owns the shared main-KV write. Other TP ranks must not
            # initialize their Hot rows or report completion before that write.
            # PD already joins all TP ranks. An additional collective is only
            # necessary when a native paired restore is also in flight.
            statuses = (
                pd_worker._gather_tp_read_status(local_completed, self._failed)
                if any(1 in expected for expected in self._expected.values())
                else [(local_completed, self._failed)]
            )
            completed = set.intersection(*(done for done, _ in statuses))
            self._failed.update(set().union(*(failed for _, failed in statuses)))
            pd_worker.finish_prefix_cache_loads(completed - self._cancelled - self._failed)
        output.finished_recving.update(completed)
        output.failed_recving.update(completed & self._failed)
        cancelled = completed & self._cancelled
        output.finished_sending.update(cancelled)
        if cancelled:
            # Cleanup only: polling children again would consume completions
            # for unrelated requests arriving between the two polls.
            self._connectors[0].connector_worker._cleanup_request_state(cancelled)
        for request_id in completed:
            del self._receives[request_id]
            del self._expected[request_id]
            self._ready_errors.update(self._blocked_errors.pop(request_id, ()))
        self._failed.difference_update(completed)
        self._cancelled.difference_update(completed)
        return output

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Recompute/cleanup may only reuse destinations after both writers drain.
        errors, self._ready_errors = self._ready_errors, set()
        return errors

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._prefix_matches.pop(request.request_id, None)
        self._prefix_hits.pop(request.request_id, None)
        pending, params = super().request_finished_all_groups(request, block_ids)
        return pending or request.request_id in self._scheduler_pending, params

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        self._scheduler_pending.difference_update(connector_output.finished_recving or ())
