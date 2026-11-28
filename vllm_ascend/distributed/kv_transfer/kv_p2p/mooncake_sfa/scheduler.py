# SPDX-License-Identifier: Apache-2.0
"""SFA row admission and lifetime policy on the shared Mooncake scheduler."""

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata


class MooncakeSFAConnectorScheduler(MooncakePullConnectorScheduler):
    def __init__(self, vllm_config, engine_id, kv_cache_config) -> None:
        self._copy_sfa_slot_allocator = None
        self._copy_sfa_deferred_releases: set[str] = set()
        sparse_cfg = init_ascend_config(vllm_config).sparse_kv_offload_config
        transfer_config = vllm_config.kv_transfer_config
        if (
            transfer_config is not None
            and transfer_config.is_kv_consumer
            and sparse_cfg is not None
            and sparse_cfg.use_fused_copy_sfa
        ):
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
                CopySfaTopkSlotAllocator,
                copy_sfa_pool_capacity,
            )

            self._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(
                copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs)
            )

        # Complete policy initialization before the parent starts threads.
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        """Check SFA row capacity before accepting remote prompt tokens."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            allocator = getattr(self, "_copy_sfa_slot_allocator", None)
            token_ids = request.prompt_token_ids or []
            remote_prompt_tokens = params.get("remote_num_prompt_tokens")
            if allocator is not None and remote_prompt_tokens is not None and len(token_ids) != remote_prompt_tokens:
                raise ValueError(
                    "Mooncake V2 Sparse KV offload requires the same original prompt on P and D; "
                    "appending the producer's first output token is not supported."
                )
            if allocator is not None and not allocator.can_bind(request.request_id):
                # Waiting async receivers do not count towards max_num_seqs.
                # Defer before KV allocation, matching SFA's row admission.
                return None, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens) -> None:
        params = request.kv_transfer_params
        if (
            params is not None
            and params.get("do_remote_prefill")
            and params.get("remote_block_ids")
            and all(key in params for key in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id"))
        ):
            if self._copy_sfa_slot_allocator is not None:
                self._copy_sfa_slot_allocator.bind(request.request_id)
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output) -> MooncakeSFAConnectorMetadata:
        allocator = self._copy_sfa_slot_allocator
        bindings = allocator.bound_slots() if allocator is not None else {}
        if allocator is not None:
            # Validate before the public builder consumes its pending queue.
            missing = self._reqs_need_recv.keys() - bindings.keys()
            if missing:
                raise ValueError(f"Mooncake SFA requests have no reserved slots: {sorted(missing)}")
        metadata = super().build_connector_meta(scheduler_output)
        return MooncakeSFAConnectorMetadata.from_mooncake(metadata, bindings)

    def request_finished(self, request, block_ids):
        allocator = getattr(self, "_copy_sfa_slot_allocator", None)
        if allocator is not None:
            if request.request_id in self._reqs_recv_info:
                # An aborted async PD may still write this rank-private row.
                # Reuse it only after completion from every D worker.
                deferred = getattr(self, "_copy_sfa_deferred_releases", None)
                if deferred is None:
                    deferred = self._copy_sfa_deferred_releases = set()
                deferred.add(request.request_id)
            else:
                allocator.release(request.request_id)
        return super().request_finished(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Release deferred SFA rows after worker aggregation and before the ACK."""
        deferred = self._copy_sfa_deferred_releases
        for req_id in connector_output.finished_recving or ():
            if req_id in deferred:
                assert self._copy_sfa_slot_allocator is not None
                self._copy_sfa_slot_allocator.release(req_id)
                deferred.remove(req_id)
        super().update_connector_output(connector_output)
