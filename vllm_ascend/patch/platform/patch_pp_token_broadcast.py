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

``PPHandler.receive``/``broadcast``/``broadcast_drafts`` all consult
``compute_need_sampled_mask`` and take their no-op path when it returns
``None``, so patching that one function skips the whole collective.

Non-last PP ranks use the broadcast tokens only to build the inputs of a
follow-up step for the same request (decode or spec verification). On a
pure disaggregated-prefill node every request leaves the engine right
after its final prefill chunk, so the payload is discarded by the
deferred consumer anyway (``get_prev_sampled_outputs`` drops freed
requests). The collective still couples all PP ranks at every request
boundary and contends with the next chunk's activation p2p transfer,
which serializes chunked prefill under pipeline parallelism.

On a prefill node the broadcast is skipped when every sample-producing
request is a final prefill chunk. Both PP ranks derive the verdict from
identical per-step batch state, so the sender and the receivers always
agree. This relies on the disaggregated-prefill convention that
prefill-node requests stop after one sampled token (routers submit
``max_tokens=1``); engines that keep decoding requests are unaffected
because their batches keep the broadcast.
"""

from __future__ import annotations

from functools import wraps

from vllm_ascend.patch.platform.patch_pp_mtp import (
    _is_pd_prefill_node as _is_pd_prefill_node_for_config,
)

_PATCHED_ATTR = "_vllm_ascend_pp_token_broadcast_patched"


def _is_pd_prefill_node() -> bool:
    """Whether this engine is a pure disaggregated-prefill producer."""
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    return vllm_config is not None and _is_pd_prefill_node_for_config(vllm_config)


def _apply_patch() -> None:
    from vllm.v1.worker.gpu import pp_utils

    original = getattr(pp_utils, "compute_need_sampled_mask", None)
    if original is None or getattr(original, _PATCHED_ATTR, False):
        # Older vLLM without the deferred PP token broadcast; nothing to skip.
        return

    @wraps(original)
    def compute_need_sampled_mask_without_pd_consumers(input_batch):
        produces_sample = original(input_batch)
        if produces_sample is None or not _is_pd_prefill_node():
            return produces_sample

        final_prefill_chunk = input_batch.is_prefilling_np & (
            input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens >= input_batch.prefill_len_np
        )
        if final_prefill_chunk[produces_sample].all():
            # Every sample belongs to a request that finishes prefill and
            # leaves this engine; no follow-up step reads the payload.
            return None
        return produces_sample

    setattr(
        compute_need_sampled_mask_without_pd_consumers,
        _PATCHED_ATTR,
        True,
    )  # type: ignore[attr-defined]
    pp_utils.compute_need_sampled_mask = compute_need_sampled_mask_without_pd_consumers


_apply_patch()
