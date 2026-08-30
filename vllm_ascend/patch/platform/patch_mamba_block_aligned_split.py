# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend compatibility patch for vLLM PR #51113.

PR #51113 changed Mamba align-mode chunk splitting for every non-final
prefill chunk. Kimi-K3 P/D deployments need the previous behavior while the
upstream change is being investigated with externally supplied KV state.
"""

import inspect

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)


def _mamba_block_aligned_split_before_vllm_pr_51113(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Use the Mamba align splitting behavior from before vLLM PR #51113."""
    start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return num_new_tokens

    block_size = self.cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    if end < last_cache_position:
        max_prefill_tokens = self.max_num_scheduled_tokens
        long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
        if long_prefill_threshold > 0:
            max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
        aligned_end = end // block_size * block_size
        if aligned_end > start or block_size <= max_prefill_tokens:
            end = aligned_end

    next_block_boundary = (start // block_size + 1) * block_size
    tail_boundary = (
        request.num_prompt_tokens // self.hash_block_size * self.hash_block_size if self.mamba_partial_cache_hit else 0
    )
    stops = (
        next_block_boundary if start % block_size != 0 and next_block_boundary <= last_cache_position else 0,
        last_cache_position,
        tail_boundary if last_cache_position < tail_boundary < request.num_prompt_tokens else 0,
        start + (request.shared_prefix_boundary - start) // block_size * block_size
        if start < request.shared_prefix_boundary < end
        else 0,
    )
    end = min((stop for stop in stops if start < stop < end), default=end)
    return max(end - start, 0)


current_parameters = tuple(inspect.signature(Scheduler._mamba_block_aligned_split).parameters)
if current_parameters != _EXPECTED_PARAMETERS:
    raise RuntimeError(
        "Cannot apply the vLLM PR #51113 compatibility patch: unexpected "
        f"Scheduler._mamba_block_aligned_split signature {current_parameters}"
    )

Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split_before_vllm_pr_51113
logger.warning("Applied Ascend compatibility patch reverting vLLM PR #51113 Mamba align-mode chunk splitting behavior.")
