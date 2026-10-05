"""Adapt vLLM hooks to AscendStore v1."""

from __future__ import annotations

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

from .protocol.rpc import LookupServer
from .protocol.transfer import KVTransferStep, StoreSourceReleaseMetadata
from .runtime.result import LoadResult
from .scheduler import KVPoolScheduler
from .vllm_adapter import create_kv_pool_scheduler, create_kv_pool_worker
from .worker import KVPoolWorker

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
    """Delegate vLLM hooks to one Scheduler or Worker owner."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)

        self.scheduler: KVPoolScheduler | None = None
        self.worker: KVPoolWorker | None = None
        self.lookup_server: LookupServer | None = None
        self._pending_load_result: LoadResult | None = None
        self._released_store_job_ids: set[int] = set()
        if role == KVConnectorRole.SCHEDULER:
            lookup_address = self._resolve_lookup_address(vllm_config)
            self.scheduler = create_kv_pool_scheduler(vllm_config, kv_cache_config, lookup_address)
        else:
            self.worker = create_kv_pool_worker(vllm_config, kv_cache_config)
            if vllm_config.parallel_config.rank == 0:
                lookup_address = self._resolve_lookup_address(vllm_config)
                self.lookup_server = LookupServer(self.worker.lookup, lookup_address)

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
        assert self.scheduler is not None
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        assert self.scheduler is not None
        self.scheduler.confirm_allocation(request, blocks, num_external_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if self.scheduler is not None:
            self.scheduler.accept_worker_metadata(connector_output.kv_connector_worker_meta)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        assert self.scheduler is not None
        return self.scheduler.build_step(scheduler_output)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, None]:
        assert self.scheduler is not None
        return False, None

    def request_finished_all_groups(self, request: Request, block_ids: tuple[list[int], ...]) -> tuple[bool, None]:
        assert self.scheduler is not None
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self.scheduler is not None
        return self.scheduler.register_finished_partial_tail(request, block_ids, partial_tail_offloads)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        assert self.scheduler is not None
        self.scheduler.bind_gpu_block_pool(gpu_block_pool)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.worker is not None
        self.worker.bind_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.worker is None:
            super().bind_connector_metadata(connector_metadata)
            return
        if not isinstance(connector_metadata, KVTransferStep):
            raise TypeError(f"Expected KVTransferStep, got {type(connector_metadata).__name__}")
        super().bind_connector_metadata(connector_metadata)
        try:
            self.worker.begin_step(connector_metadata)
        except BaseException:
            super().clear_connector_metadata()
            raise

    def clear_connector_metadata(self) -> None:
        if self.worker is not None:
            self.worker.end_step()
        super().clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        del kv_connector_metadata
        assert self.worker is not None
        try:
            self.worker.fence_previous_store()
        finally:
            self._released_store_job_ids.update(self.worker.take_released_store_job_ids())

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.worker is not None
        self.worker.start_load()

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.worker is not None
        self.worker.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        del kv_layer, attn_metadata, kwargs
        assert self.worker is not None
        self.worker.save_layer(layer_name)

    def wait_for_save(self) -> None:
        assert self.worker is not None
        self.worker.finish_step()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.worker is not None
        if self._pending_load_result is not None:
            raise RuntimeError("Previous Load result has not been fully consumed")
        # A finished request can still own blocks held for an in-flight async Load; its late completion releases them.
        load_result = self.worker.collect_load_result()
        if load_result.failed_request_ids:
            failed_locations = sorted(
                (location.group_id, location.block_id) for location in load_result.failed_locations
            )
            raise RuntimeError(
                "Hybrid KV Load failed, but this vLLM version cannot report request-level Load failures: "
                f"{sorted(load_result.failed_request_ids)}; cache group/block failures {failed_locations}"
            )
        self._pending_load_result = load_result
        return set(), set(self._pending_load_result.completed_request_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.worker is not None
        if self._pending_load_result is None:
            return set()
        failed_block_ids = set(self._pending_load_result.failed_block_ids)
        self._pending_load_result = None
        return failed_block_ids

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if self.worker is not None:
            self._released_store_job_ids.update(self.worker.take_released_store_job_ids())
        if not self._released_store_job_ids:
            return None
        metadata = StoreSourceReleaseMetadata({store_job_id: 1 for store_job_id in self._released_store_job_ids})
        self._released_store_job_ids.clear()
        return metadata

    def has_pending_push_work(self) -> bool:
        return self.scheduler is not None and self.scheduler.has_pending_push_work()

    def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        if self.lookup_server is not None:
            self.lookup_server.close()
        if self.worker is not None:
            self.worker.close()
