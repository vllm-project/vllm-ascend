# SPDX-License-Identifier: Apache-2.0
"""SFA row admission and lifetime policy on the shared Mooncake scheduler."""

from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata


class MooncakeSFAConnectorScheduler(MooncakePullConnectorScheduler):
    def _initialize_request_state(self, vllm_config) -> None:
        self._copy_sfa_slot_allocator = None
        self._copy_sfa_deferred_releases: set[str] = set()
        sparse_cfg = getattr(self.ascend_config, "sparse_kv_offload_config", None)
        if self.kv_transfer_config.is_kv_consumer and sparse_cfg is not None and sparse_cfg.use_fused_copy_sfa:
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
                CopySfaTopkSlotAllocator,
                copy_sfa_pool_capacity,
            )

            self._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(
                copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs)
            )

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

    def _create_connector_metadata(self) -> MooncakeSFAConnectorMetadata:
        allocator = self._copy_sfa_slot_allocator
        if allocator is not None:
            # Preserve the original build-time rejection of an unreserved row.
            for request_id in self._reqs_need_recv:
                allocator.get(request_id)
        return MooncakeSFAConnectorMetadata(allocator.bound_slots() if allocator is not None else {})

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

    def _on_receive_finished(self, req_id: str) -> None:
        deferred: set[str] | None = getattr(self, "_copy_sfa_deferred_releases", None)
        if deferred is not None and req_id in deferred:
            assert self._copy_sfa_slot_allocator is not None
            self._copy_sfa_slot_allocator.release(req_id)
            deferred.remove(req_id)
