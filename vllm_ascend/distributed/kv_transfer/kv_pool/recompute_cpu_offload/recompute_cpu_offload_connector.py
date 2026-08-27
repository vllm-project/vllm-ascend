# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecomputeCPUOffloadConnector: minimal CPU KV cache offloading."""

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

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.manager import (
    RecomputeCPUOffloadScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (
    RecomputeCPUOffloadMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker import (
    RecomputeCPUOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)


class RecomputeCPUOffloadConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """CPU KV cache preservation for recompute-preempted requests."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        cpu_capacity_bytes = int(extra_config.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES))
        enable_offload_prefix_caching = extra_config.get("enable_offload_prefix_caching", False)
        if not isinstance(enable_offload_prefix_caching, bool):
            raise ValueError(f"enable_offload_prefix_caching must be a boolean, got {enable_offload_prefix_caching!r}")
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity_per_rank = cpu_capacity_bytes // world_size
        if "cpu_bytes_to_use_per_rank" in extra_config:
            explicit = int(extra_config["cpu_bytes_to_use_per_rank"])
            if explicit != cpu_capacity_per_rank:
                logger.warning(
                    "cpu_bytes_to_use_per_rank (%.2f GB) != "
                    "cpu_bytes_to_use/world_size (%.2f GB). Using per-rank value.",
                    explicit / (1024**3),
                    cpu_capacity_per_rank / (1024**3),
                )
            cpu_capacity_per_rank = explicit

        self.scheduler_manager: RecomputeCPUOffloadScheduler | None = None
        self.worker_handler: RecomputeCPUOffloadWorker | None = None

        logger.info(
            "RecomputeCPUOffloadConnector: role=%s, per_rank=%.2f GB, world_size=%d, offload_prefix_caching=%s",
            role.name,
            cpu_capacity_per_rank / (1024**3),
            world_size,
            enable_offload_prefix_caching,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = RecomputeCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                enable_offload_prefix_caching,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = RecomputeCPUOffloadWorker(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
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
            assert isinstance(connector_metadata, RecomputeCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(kv_connector_metadata, RecomputeCPUOffloadMetadata)
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

    # NOTE: New API only for RecomputeCPUOffloadConnector.
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
        return RecomputeCPUOffloadMetadata()

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

    # NOTE: New API only for RecomputeCPUOffloadConnector.
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
