"""Adapt vLLM hooks to AscendStore v1."""

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

from .assembly import build_kv_pool_runtime, build_transfer_planner
from .backend import BACKEND_IMPORTS
from .graph.evaluation import LoadResult
from .planning.availability import LookupQuery
from .protocol.rpc import LookupServer
from .protocol.transfer import KVTransferStep

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from .execution.runtime import KVPoolRuntime
    from .graph.evaluation import KVPoolStepEvaluation
    from .planning.planner import TransferPlanner


class AscendStoreV1Connector(KVConnectorBase_V1, SupportsHMA):
    """Adapt vLLM hooks to the AscendStore v1 planner and KV Pool runtime."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if extra_config.get("use_layerwise", False):
            raise ValueError("AscendStore v1 currently requires non-Layerwise Load")
        backend_name = extra_config.get("backend", "mooncake").strip().lower()
        if backend_name not in BACKEND_IMPORTS:
            raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")

        self.planner: TransferPlanner | None = None
        self.runtime: KVPoolRuntime | None = None
        self.lookup_server: LookupServer | None = None
        self._step_evaluation: KVPoolStepEvaluation | None = None
        self._pending_load_result: LoadResult | None = None
        if role == KVConnectorRole.SCHEDULER:
            lookup_address = self._resolve_lookup_address(vllm_config)
            self.planner = build_transfer_planner(vllm_config, kv_cache_config, lookup_address)
        else:
            self.runtime = build_kv_pool_runtime(vllm_config, kv_cache_config)
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
            prompt_token_len=len(request.prompt_token_ids),
            request_token_len=request.num_tokens,
            block_hashes=request.block_hashes,
            local_cached_tokens=num_computed_tokens,
        )
        prefix_plan = self.planner.lookup(lookup_query)
        return prefix_plan.num_new_matched_tokens, prefix_plan.load_is_deferred

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        assert self.planner is not None
        self.planner.confirm_allocation(request, blocks.get_block_ids(), num_external_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        return

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        assert self.planner is not None
        return self.planner.build_step(scheduler_output)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, None]:
        assert self.planner is not None
        return False, None

    def request_finished_all_groups(self, request: Request, block_ids: tuple[list[int], ...]) -> tuple[bool, None]:
        assert self.planner is not None
        return False, None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.runtime is not None
        self.runtime.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.runtime is None:
            super().bind_connector_metadata(connector_metadata)
            return
        if not isinstance(connector_metadata, KVTransferStep):
            raise TypeError(f"Expected KVTransferStep, got {type(connector_metadata).__name__}")
        if self._step_evaluation is not None:
            raise RuntimeError("Previous KV Pool step evaluation has not been cleared")
        super().bind_connector_metadata(connector_metadata)
        self._step_evaluation = self.runtime.begin_step(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._step_evaluation = None
        super().clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        assert self.runtime is not None
        self.runtime.fence_previous_store()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.runtime is not None
        self.runtime.start_load(self._require_step_evaluation())

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        assert self.runtime is not None
        self.runtime.finish_step(self._require_step_evaluation())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.runtime is not None
        if self._pending_load_result is not None:
            raise RuntimeError("Previous Load result has not been fully consumed")
        # A finished request can still own blocks held for an in-flight async Load; its late completion releases them.
        load_result = self.runtime.collect_load_result(self._require_step_evaluation())
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

    def shutdown(self) -> None:
        if self.planner is not None:
            self.planner.close()
        if self.lookup_server is not None:
            self.lookup_server.close()
        if self.runtime is not None:
            self.runtime.close()

    def _require_step_evaluation(self) -> KVPoolStepEvaluation:
        if self._step_evaluation is None:
            raise RuntimeError("KV Pool step evaluation has not been created")
        return self._step_evaluation
