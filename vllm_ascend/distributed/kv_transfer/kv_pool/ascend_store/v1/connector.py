"""Adapt vLLM hooks to AscendStore v1."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.v1.core import kv_cache_utils

from .planning.availability import LookupQuery
from .planning.ownership import StoreSourceLeases
from .planning.planner import TransferPlanner
from .protocol.rpc import LookupServer
from .protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    StateCheckpointSource,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from .runtime.result import LoadResult
from .runtime.runtime import KVPoolRuntime
from .vllm_adapter import (
    adapt_scheduler_output,
    create_kv_pool_runtime,
    create_transfer_planner,
    group_uses_align_state,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class AscendStoreV1Connector(KVConnectorBase_V1, SupportsHMA):
    """Adapt vLLM hooks to the AscendStore v1 planner and KV Pool runtime."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)

        self.planner: TransferPlanner | None = None
        self.runtime: KVPoolRuntime | None = None
        self.lookup_server: LookupServer | None = None
        self._pending_load_result: LoadResult | None = None
        self._requests: dict[str, Request] = {}
        self._released_store_job_ids: set[int] = set()
        self._finished_checkpoint_stores: list[CheckpointStoreCommand] = []
        transfer_config = vllm_config.kv_transfer_config
        extra_config = transfer_config.kv_connector_extra_config
        self._store_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or extra_config.get(
            "consumer_is_to_put", False
        )
        self._align_state_group_ids = frozenset(
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if group_id in kv_cache_config.transfer_group_ids and group_uses_align_state(group)
        )
        self._transfer_group_ids = frozenset(kv_cache_config.transfer_group_ids)
        self._store_source_leases = StoreSourceLeases(
            self._transfer_group_ids,
            self._align_state_group_ids,
            vllm_config.parallel_config.world_size,
        )
        _, self._hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        if role == KVConnectorRole.SCHEDULER:
            lookup_address = self._resolve_lookup_address(vllm_config)
            self.planner = create_transfer_planner(vllm_config, kv_cache_config, lookup_address)
        else:
            self.runtime = create_kv_pool_runtime(vllm_config, kv_cache_config)
            if vllm_config.parallel_config.rank == 0:
                lookup_address = self._resolve_lookup_address(vllm_config)
                self.lookup_server = LookupServer(self.runtime.lookup, lookup_address)

    @staticmethod
    def _resolve_lookup_address(vllm_config: VllmConfig) -> str:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        rpc_port = extra_config.get("lookup_rpc_port", extra_config.get("mooncake_rpc_port", 0))
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        return f"ipc://{envs.VLLM_RPC_BASE_PATH}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict[tuple[int, int], Any]) -> None:
        """Pool keys, not the vLLM P/D handshake, identify PP shards."""
        return

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        assert self.planner is not None
        lookup_query = LookupQuery(
            request_id=request.request_id,
            prompt_token_len=request.num_prompt_tokens,
            request_token_len=request.num_tokens,
            block_hashes=request.block_hashes,
            local_cached_tokens=num_computed_tokens,
        )
        prefix_plan = self.planner.lookup(lookup_query)
        return prefix_plan.num_new_matched_tokens, prefix_plan.load_is_deferred

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        assert self.planner is not None
        self._requests[request.request_id] = request
        self.planner.confirm_allocation(
            request.request_id,
            tuple(tuple(block_ids) for block_ids in blocks.get_block_ids()),
            tuple(request.block_hashes),
            request.num_prompt_tokens,
            num_external_tokens,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if not isinstance(metadata, StoreSourceReleaseMetadata):
            return
        self._store_source_leases.release(metadata.released_store_jobs)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        assert self.planner is not None
        planning_step = adapt_scheduler_output(scheduler_output, self._requests, store_enabled=self._store_enabled)
        step = self.planner.build_step(planning_step)
        for request_id in planning_step.finished_request_ids | planning_step.preempted_request_ids:
            self._requests.pop(request_id, None)
        pending_checkpoint_stores = tuple(self._finished_checkpoint_stores)
        commands = tuple(self._store_source_leases.acquire(command) for command in step.store.commands)
        self._finished_checkpoint_stores.clear()
        return replace(step, store=StoreCommandBatch(commands + pending_checkpoint_stores))

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, None]:
        assert self.planner is not None
        return False, None

    def request_finished_all_groups(self, request: Request, block_ids: tuple[list[int], ...]) -> tuple[bool, None]:
        assert self.planner is not None
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self.planner is not None
        if not self._store_enabled or not partial_tail_offloads:
            return False
        snapshot = self.planner.request_progress.get(request.request_id)
        if snapshot is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("Finished state checkpoints must share one token boundary")
        boundary = next(iter(boundaries))
        if boundary <= 0 or boundary > snapshot.num_prompt_tokens:
            return False
        if boundary % self._hash_block_size or boundary > len(request.block_hashes) * self._hash_block_size:
            return False
        sources = tuple(
            StateCheckpointSource(group_id, block_id, boundary)
            for group_id, block_id, boundary in partial_tail_offloads
            if group_id in self._transfer_group_ids and group_id in self._align_state_group_ids and block_id > 0
        )
        if len(sources) != len(partial_tail_offloads):
            return False
        command = CheckpointStoreCommand(
            request.request_id,
            tuple(tuple(group_block_ids) for group_block_ids in block_ids),
            tuple(request.block_hashes),
            snapshot.published_store_end_token,
            sources,
        )
        self._finished_checkpoint_stores.append(self._store_source_leases.acquire(command))
        # Exact job-level references own the source blocks, so vLLM can release the request's own references on time.
        return False

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._store_source_leases.bind_block_pool(gpu_block_pool)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.runtime is not None
        self.runtime.bind_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.runtime is None:
            super().bind_connector_metadata(connector_metadata)
            return
        if not isinstance(connector_metadata, KVTransferStep):
            raise TypeError(f"Expected KVTransferStep, got {type(connector_metadata).__name__}")
        super().bind_connector_metadata(connector_metadata)
        self.runtime.begin_step(connector_metadata)

    def clear_connector_metadata(self) -> None:
        if self.runtime is not None:
            self.runtime.end_step()
        super().clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        del kv_connector_metadata
        assert self.runtime is not None
        try:
            self.runtime.fence_previous_store()
        finally:
            self._released_store_job_ids.update(self.runtime.take_released_store_job_ids())

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.runtime is not None
        self.runtime.start_load()

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.runtime is not None
        self.runtime.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        del kv_layer, attn_metadata, kwargs
        assert self.runtime is not None
        self.runtime.save_layer(layer_name)

    def wait_for_save(self) -> None:
        assert self.runtime is not None
        self.runtime.finish_step()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.runtime is not None
        if self._pending_load_result is not None:
            raise RuntimeError("Previous Load result has not been fully consumed")
        # A finished request can still own blocks held for an in-flight async Load; its late completion releases them.
        load_result = self.runtime.collect_load_result()
        if load_result.failed_request_ids:
            raise RuntimeError(
                "Hybrid KV Load failed, but this vLLM version cannot report request-level Load failures: "
                f"{sorted(load_result.failed_request_ids)}"
            )
        self._pending_load_result = load_result
        return set(), set(self._pending_load_result.completed_request_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.runtime is not None
        if self._pending_load_result is None:
            return set()
        failed_block_ids = set(self._pending_load_result.failed_block_ids)
        self._pending_load_result = None
        return failed_block_ids

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if self.runtime is not None:
            self._released_store_job_ids.update(self.runtime.take_released_store_job_ids())
        if not self._released_store_job_ids:
            return None
        metadata = StoreSourceReleaseMetadata({store_job_id: 1 for store_job_id in self._released_store_job_ids})
        self._released_store_job_ids.clear()
        return metadata

    def has_pending_push_work(self) -> bool:
        return self._store_source_leases.has_pending() or bool(self._finished_checkpoint_stores)

    def shutdown(self) -> None:
        if self.planner is not None:
            self.planner.close()
        if self.lookup_server is not None:
            self.lookup_server.close()
        if self.runtime is not None:
            self.runtime.close()
