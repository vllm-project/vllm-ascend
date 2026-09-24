"""vLLM hooks for the extracted AscendStore classic business path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from .metadata import AscendStoreV1Metadata
from .scheduler.lookup import SchedulerLookupRequest
from .scheduler.service import SchedulerService
from .worker.lookup import LookupKeyServer
from .worker.service import WorkerService

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendStoreV1Connector(KVConnectorBase_V1, SupportsHMA):
    """Adapt vLLM hooks to the classic AscendStore v1 services."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("AscendStore v1 classic path requires one KV cache group")
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if extra_config.get("use_layerwise", False) or extra_config.get("load_async", False):
            raise ValueError("AscendStore v1 classic path requires non-Layerwise synchronous Load")
        if extra_config.get("backend", "mooncake").lower() != "mooncake":
            raise ValueError("AscendStore v1 classic path requires Mooncake")

        self.scheduler: SchedulerService | None = None
        self.worker: WorkerService | None = None
        self.lookup_server: LookupKeyServer | None = None
        self._store_hook = self._submit_store
        if role == KVConnectorRole.SCHEDULER:
            lookup_address = self._resolve_lookup_address(vllm_config)
            self.scheduler = SchedulerService(vllm_config, kv_cache_config, lookup_address)
        else:
            self.worker = WorkerService(vllm_config, kv_cache_config)
            if not self.worker.can_store:
                self._store_hook = self._skip_store
            if vllm_config.parallel_config.rank == 0:
                lookup_address = self._resolve_lookup_address(vllm_config)
                self.lookup_server = LookupKeyServer(self.worker.lookup, lookup_address)

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
        lookup_request = SchedulerLookupRequest(
            req_id=request.request_id,
            prompt_token_len=len(request.prompt_token_ids),
            num_tokens=request.num_tokens,
            block_hashes=request.block_hashes,
            num_computed_tokens=num_computed_tokens,
        )
        return self.scheduler.lookup(lookup_request), False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        assert self.scheduler is not None
        self.scheduler.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendStoreV1Metadata:
        assert self.scheduler is not None
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, None]:
        assert self.scheduler is not None
        return False, None

    def request_finished_all_groups(self, request: Request, block_ids: tuple[list[int], ...]) -> tuple[bool, None]:
        assert self.scheduler is not None
        return False, None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.worker is not None
        self.worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        assert self.worker is not None
        self.worker.wait_for_previous_store()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, AscendStoreV1Metadata)
        self.worker.load(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        self._store_hook()

    def _submit_store(self) -> None:
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, AscendStoreV1Metadata)
        self.worker.submit_store(metadata)

    @staticmethod
    def _skip_store() -> None:
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, AscendStoreV1Metadata)
        self.worker.clear_store_completion_bookkeeping(metadata.preempted_req_ids)
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.worker is not None
        return self.worker.get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        if self.lookup_server is not None:
            self.lookup_server.close()
