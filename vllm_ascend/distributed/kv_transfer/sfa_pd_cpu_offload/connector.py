# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""PD-disaggregated SFA KV-transfer connector.

On the Decode node (``kv_consumer``), remote Prefill exposes its KV and Decode
pulls the bulk MLA KV into a CPU pinned offload pool; the indexer KV lands in
HBM. The CPU pool and sparse resident-cache load path are owned by
:class:`KVOffloadDecodeManager`.
"""

from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (
    SFAPDCpuOffloadScheduler,
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (
    SFAPDCpuOffloadConsumerWorker,
    SFAPDCpuOffloadProducerWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


class SFAPDCpuOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """One connector class branching on ``role`` and ``kv_role``.

    * SCHEDULER + producer : P-side metadata for memfabric pull notifications.
    * SCHEDULER + consumer : D-side vLLM block-id / advertisement tracking.
    * WORKER + producer    : P-side layer-wise READ_READY notifications.
    * WORKER + consumer    : D-side manager CPU-pool + memfabric pull read +
      indexer/main split registration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_consumer = vllm_config.kv_transfer_config.is_kv_consumer
        self.requires_full_blocks_on_update_after_alloc = (
            role == KVConnectorRole.SCHEDULER and self.is_producer
        )
        # SFA path is layer-wise on both sides.
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", True)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        # Decode offload is asymmetric: P exposes regular paged KV while D owns
        # the KVOffloadDecodeManager CPU pool.
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        # AscendConfig may not be initialized yet at connector construction
        # time; init_ascend_config is idempotent (no-op if already done).
        init_ascend_config(vllm_config)
        decode_offload_enabled = get_ascend_config().kv_offload_decode_config.enabled
        if self.is_producer:
            assert not decode_offload_enabled, (
                "SFAPDCpuOffloadConnector producer (P) must run with "
                "kv_offload_decode_config.enabled=false."
            )
        if self.is_consumer:
            assert decode_offload_enabled, (
                "SFAPDCpuOffloadConnector consumer (D) must run with "
                "kv_offload_decode_config.enabled=true."
            )

        if role == KVConnectorRole.SCHEDULER:
            # Producer scheduler prepares P-side metadata; consumer scheduler
            # allocates and advertises D-side CPU blocks.
            if self.is_producer:
                self.connector_scheduler = SFAPDProducerScheduler(vllm_config, kv_cache_config, str(self.engine_id))
            else:
                self.connector_scheduler = SFAPDCpuOffloadScheduler(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            if self.is_producer:
                self.connector_worker = SFAPDCpuOffloadProducerWorker(vllm_config, kv_cache_config, str(self.engine_id))
            else:
                self.connector_worker = SFAPDCpuOffloadConsumerWorker(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        if self.is_consumer:
            return self.connector_worker.get_finished(finished_req_ids)
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Per-layer gate called before each layer's attention computation.

        D-side: no-op (SFA loads through ``KVOffloadDecodeManager`` directly).
        P-side: before this layer writes HBM, wait for D to finish every pending
        read from the physical main/indexer storage slots touched by this layer.

        The per-layer send-done events are cleared when a READ_READY_BATCH is
        sent and set again by the pipelined MembPull send thread when READ_DONE
        arrives. Events are initially set, so the first occupant does not block.
        """
        if not self.is_producer:
            return
        match = _LAYER_IDX_RE.search(layer_name)
        if match is None:
            return
        layer_idx = int(match.group(1))
        self.wait_for_layer_send(layer_idx)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        # SFA attention calls this every forward, including profiling / graph
        # capture where no per-step connector metadata is bound. Nothing to save
        # then; skip rather than trip _get_connector_metadata's assert.
        if not self.has_connector_metadata():
            return
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._get_connector_metadata())

    def wait_for_save(self):
        # Decode KV writes are synchronized by KVOffloadDecodeManager. P-side
        # completion is tracked by READ_DONE/storage_send_done_events.
        if self.is_consumer and self.connector_worker is not None:
            self.connector_worker.wait_for_save()

    # Phase 3: real per-req CPU-block count for the solution-1 threshold.
    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_num_cpu_blocks(req_ids)

    def shutdown(self) -> None:
        for component in (self.connector_worker, self.connector_scheduler):
            shutdown = getattr(component, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def close(self) -> None:
        self.shutdown()

    # P-side buffer-reuse gate: block until D has read a layer's source KV buffer,
    # so the buffer may be reused by a later layer.
    def wait_for_layer_send(self, layer_idx: int) -> None:
        worker = self.connector_worker
        if worker is None or not hasattr(worker, "wait_for_layer_send"):
            return
        worker.wait_for_layer_send(layer_idx)
