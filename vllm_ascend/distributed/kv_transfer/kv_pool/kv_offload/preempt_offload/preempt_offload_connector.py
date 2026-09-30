# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PreemptOffloadConnector: minimal CPU KV cache offloading."""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.manager import (
    PreemptOffloadScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.metadata import (
    PreemptOffloadMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.worker import (
    PreemptOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class PreemptOffloadConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """CPU KV cache preservation for recompute-preempted requests."""

    @staticmethod
    def _resolve_offload_capacity(extra_config: dict[str, Any], world_size: int) -> tuple[int | None, float]:
        offload_host_memory_ratio = float(extra_config.get("offload_host_memory_ratio", 1))
        if not math.isfinite(offload_host_memory_ratio) or offload_host_memory_ratio <= 0:
            raise ValueError(
                f"offload_host_memory_ratio must be a positive finite number, got {offload_host_memory_ratio!r}"
            )

        if "cpu_bytes_to_use_per_rank" in extra_config:
            cpu_capacity_per_rank = int(extra_config["cpu_bytes_to_use_per_rank"])
        elif "cpu_bytes_to_use" in extra_config:
            cpu_capacity_per_rank = int(extra_config["cpu_bytes_to_use"]) // world_size
        else:
            cpu_capacity_per_rank = None

        if cpu_capacity_per_rank is not None and cpu_capacity_per_rank <= 0:
            raise ValueError(
                f"The effective per-rank CPU offload memory must be positive, got {cpu_capacity_per_rank} bytes"
            )
        return cpu_capacity_per_rank, offload_host_memory_ratio

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        enable_offload_prefix_caching = extra_config.get("enable_offload_prefix_caching", False)
        if not isinstance(enable_offload_prefix_caching, bool):
            raise ValueError(f"enable_offload_prefix_caching must be a boolean, got {enable_offload_prefix_caching!r}")
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity_per_rank, offload_host_memory_ratio = self._resolve_offload_capacity(extra_config, world_size)

        self.scheduler_manager: PreemptOffloadScheduler | None = None
        self.worker_handler: PreemptOffloadWorker | None = None

        logger.info(
            "PreemptOffloadConnector: role=%s, per_rank_bytes=%s, "
            "host_memory_ratio=%s, world_size=%d, offload_prefix_caching=%s",
            role.name,
            cpu_capacity_per_rank,
            offload_host_memory_ratio,
            world_size,
            enable_offload_prefix_caching,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = PreemptOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                enable_offload_prefix_caching,
                offload_host_memory_ratio,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = PreemptOffloadWorker(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                offload_host_memory_ratio,
            )

    # --- Worker-side methods ---

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, PreemptOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(kv_connector_metadata, PreemptOffloadMetadata)
            self.worker_handler.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.worker_handler is not None:
            self.worker_handler.wait_for_layer_load()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    # --- Scheduler-side methods ---

    # NOTE: New API only for PreemptOffloadConnector.
    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(request, blocks, num_external_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.update_state_before_preempt(
                request,
                block_ids,
                num_computed_tokens,
            )
        return False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return PreemptOffloadMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(request, block_ids)
        return False, None

    # NOTE: New API only for PreemptOffloadConnector.
    def has_pending_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_transfers()
        return False

    def has_preempted_request(self, req_id: str) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_preempted_request(req_id)
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []

    def reset_cache(self) -> bool | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.reset_cache()
        return None
