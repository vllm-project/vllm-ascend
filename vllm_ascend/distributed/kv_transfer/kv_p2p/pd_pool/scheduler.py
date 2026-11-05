# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPD handoff composed with ordinary full-block pool scheduling."""

from vllm.logger import logger
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import truncate_stateful_prefill_request
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.metadata import (
    POOL_PD_PROTOCOL,
    PoolPDMetadata,
    PoolPDRequest,
    prompt_digest,
)


class PoolPDScheduler:
    def __init__(self, pool, *, need_truncate=False, mtp_tokens=0):
        if type(mtp_tokens) is not int or mtp_tokens < 0:
            raise ValueError("Pool PD requires a nonnegative speculative token count")
        self.pool = pool
        self.mtp_tokens = mtp_tokens
        self.need_truncate = need_truncate
        self.original_prompts = {}
        self.requests = {}
        self.receives = {}
        self.matches = {}
        self.tail_scheduled = set()
        self.pd_load_requests = set()

    def on_new_request(self, request):
        params = request.kv_transfer_params or {}
        if self.need_truncate and params.get("do_remote_decode"):
            if request.request_id not in self.original_prompts:
                self.original_prompts[request.request_id] = (len(request.prompt_token_ids), prompt_digest(request))
            truncate_stateful_prefill_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.requests[request.request_id] = request
        self.pd_load_requests.discard(request.request_id)
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return self.pool.get_num_new_matched_tokens(request, num_computed_tokens)
        if params.get("pool_pd_protocol") != POOL_PD_PROTOCOL:
            raise ValueError("Pool PD receive requires a compatible completed producer handoff")
        remote_mtp = params.get("pool_pd_mtp_tokens", 0)
        if type(remote_mtp) is not int or remote_mtp != self.mtp_tokens:
            raise ValueError("Pool PD requires matching P/D speculative token counts")
        length = len(request.prompt_token_ids or ())
        if params.get("remote_num_prompt_tokens") != length:
            raise ValueError("Pool PD handoff prompt length differs from the decode request")
        # Keep the existing CPD last-token recomputation. Draft cache is local.
        count = max(0, length - 1 - num_computed_tokens)
        if count:
            self.pd_load_requests.add(request.request_id)
        self.matches[request.request_id] = num_computed_tokens
        return count, count > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.requests[request.request_id] = request
        if request.request_id not in self.matches:
            self.pool.update_state_after_alloc(request, blocks, num_external_tokens)
            return
        local = self.matches.pop(request.request_id)
        if num_external_tokens:
            self.receives[request.request_id] = self._request(request, blocks.get_block_ids(), local)
        # Ordinary pool scheduling tracks subsequent decode and writeback,
        # while this module exclusively owns the CPD external load.
        self.pool.update_state_after_alloc(request, blocks, 0)
        request.kv_transfer_params["do_remote_prefill"] = False

    def _request(self, request, blocks, local=0):
        length, digest = self.original_prompts.get(
            request.request_id, (len(request.prompt_token_ids), prompt_digest(request))
        )
        cache_tokens = length - 1 if self.need_truncate and length > 1 else length
        return PoolPDRequest(
            request.request_id,
            length,
            digest,
            [list(group) for group in blocks],
            list(request.block_hashes),
            local,
            cache_tokens,
        )

    def build_connector_meta(self, output):
        released = set(output.finished_req_ids or ()) | set(output.preempted_req_ids or ())
        self.tail_scheduled.difference_update(released)
        for rid in output.finished_req_ids:
            self.pd_load_requests.discard(rid)
            self.original_prompts.pop(rid, None)
            self.requests.pop(rid, None)
            self.matches.pop(rid, None)
            self.receives.pop(rid, None)
        for rid in output.preempted_req_ids or ():
            self.receives.pop(rid, None)
        metadata = PoolPDMetadata(self.pool.build_connector_meta(output), list(self.receives.values()))
        self.receives.clear()
        for rid, scheduled in output.num_scheduled_tokens.items():
            request = self.requests.get(rid)
            if request is None or rid in self.tail_scheduled:
                continue
            params = request.kv_transfer_params or {}
            if not params.get("do_remote_decode"):
                continue
            prompt_len = len(request.prompt_token_ids)
            if request.num_computed_tokens + scheduled < prompt_len:
                continue
            tracker = self.pool._request_trackers.get(rid)
            if tracker is None:
                raise RuntimeError("Pool PD producer has no allocated block table")
            metadata.tails.append(self._request(request, tracker.allocated_block_ids_by_group))
            self.tail_scheduled.add(rid)
        return metadata

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params or {}
        if params.get("do_remote_decode") and request.status == RequestStatus.FINISHED_LENGTH_CAPPED:
            if request.request_id not in self.tail_scheduled:
                raise RuntimeError("Cannot hand off a prompt without its pool publication step")
            # wait_for_save completed before this model result was delivered.
            # Objects remain in the pool; no HBM hold or remote delete ACK.
            handoff = {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "pool_pd_protocol": POOL_PD_PROTOCOL,
                "remote_num_prompt_tokens": self.original_prompts.get(
                    request.request_id, (len(request.prompt_token_ids), None)
                )[0],
            }
            if self.need_truncate or self.mtp_tokens:
                # This verifies the cache contract, not draft work requested
                # from D. Each endpoint initializes its own draft state.
                handoff["pool_pd_mtp_tokens"] = self.mtp_tokens
            logger.info(
                "Pool PD handoff req=%s prompt_tokens=%d cache_tokens=%d",
                request.request_id,
                handoff["remote_num_prompt_tokens"],
                len(request.prompt_token_ids),
            )
            return False, handoff
        return False, None
