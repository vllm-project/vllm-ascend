#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Skip PP sampled-token broadcasts that no later step consumes.

Non-last PP ranks use the broadcast tokens only to build the inputs of a
follow-up step for the same request (decode or spec verification). When a
request leaves the engine right after its final prefill chunk -- the
disaggregated-prefill case, on both pure prefill nodes and hybrid
producer-consumer instances, where such requests carry ``max_tokens=1``
-- the payload is discarded by the deferred consumer anyway
(``get_prev_sampled_outputs`` drops freed requests). The collective
still couples all PP ranks at every request boundary and contends with
the next chunk's activation p2p transfer, which serializes chunked
prefill under pipeline parallelism.

The broadcast is skipped when every sample-producing request is a final
prefill chunk that reaches its length cap with its single sampled token.
The condition is per-request and role-agnostic, so it also fires for
``max_tokens=1`` requests on hybrid (producer-consumer) deployments,
while requests that keep decoding on this engine always keep the
broadcast. Both PP ranks derive the verdict from identical per-step
state, so the sender and the receivers always agree.
"""

from __future__ import annotations

from functools import wraps

_PATCHED_ATTR = "_vllm_ascend_pp_token_broadcast_patched"

_original_receive = None
_original_broadcast = None
_original_broadcast_drafts = None


def broadcast_has_no_consumer(pp_handler, input_batch) -> bool:
    """Whether no later step on this engine reads the PP token broadcast."""
    from vllm.v1.worker.gpu import pp_utils

    request_states = getattr(pp_handler, "ascend_request_states", None)
    if request_states is None:
        return False

    produces_sample = pp_utils.compute_need_sampled_mask(input_batch)
    if produces_sample is None:
        return False

    # A final prefill chunk samples exactly one token. With
    # ``RequestState.max_seq_len == prompt_len + max_tokens``, the request
    # leaves this engine once ``prefill_len + 1 >= max_seq_len``, so no
    # follow-up step can read the broadcast payload.
    final_prefill_chunk = input_batch.is_prefilling_np & (
        input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens >= input_batch.prefill_len_np
    )
    reaches_length_cap = input_batch.prefill_len_np + 1 >= request_states.max_seq_len[input_batch.idx_mapping_np]
    finished_with_sample = final_prefill_chunk & reaches_length_cap
    return bool(finished_with_sample[produces_sample].all())


def _patch_pp_handler() -> None:
    global _original_receive, _original_broadcast, _original_broadcast_drafts

    from vllm.v1.worker.gpu import pp_utils

    pp_handler_cls = getattr(pp_utils, "PPHandler", None)
    if pp_handler_cls is None or not hasattr(pp_utils, "compute_need_sampled_mask"):
        # Older vLLM without the deferred PP token broadcast; nothing to skip.
        return
    if getattr(pp_handler_cls.receive, _PATCHED_ATTR, False):
        return

    _original_receive = pp_handler_cls.receive
    _original_broadcast = pp_handler_cls.broadcast
    _original_broadcast_drafts = pp_handler_cls.broadcast_drafts

    @wraps(_original_receive)
    def receive(self, input_batch):
        if broadcast_has_no_consumer(self, input_batch):
            # Mirror the upstream no-sample path: reserve no deferred
            # postprocess slot and report that not all requests decode next.
            return False
        return _original_receive(self, input_batch)

    @wraps(_original_broadcast)
    def broadcast(self, sampled_token_ids, num_sampled, num_rejected, input_batch):
        if broadcast_has_no_consumer(self, input_batch):
            return
        return _original_broadcast(self, sampled_token_ids, num_sampled, num_rejected, input_batch)

    @wraps(_original_broadcast_drafts)
    def broadcast_drafts(self, draft_tokens, input_batch):
        if broadcast_has_no_consumer(self, input_batch):
            return
        return _original_broadcast_drafts(self, draft_tokens, input_batch)

    for patched in (receive, broadcast, broadcast_drafts):
        setattr(patched, _PATCHED_ATTR, True)  # type: ignore[attr-defined]
    pp_handler_cls.receive = receive
    pp_handler_cls.broadcast = broadcast
    pp_handler_cls.broadcast_drafts = broadcast_drafts


def _apply_patch() -> None:
    _patch_pp_handler()


_apply_patch()
