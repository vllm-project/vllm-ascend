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

"""Apply Ascend-specific Mamba chunk-boundary handling.

A newly admitted KV-consumer request can have one prompt token left after the
external cache hit. The waiting scheduler pads that token to a ``1 + K``
speculative verifier window before Mamba alignment is applied. If the window
starts mid-block, the alignment split can shorten its physical width while the
request still advertises all ``K`` speculative placeholders.

Decode consumers preserve the complete verifier window. Sparse index-kpool
producers align against the resolved common cache-group boundary because their
physical indexer-state block is smaller than the Mamba checkpoint interval.
Other models retain the upstream behavior.
"""

import functools
import inspect

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
)

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split


def _split_without_shared_prefix_junction(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> int:
    """Apply upstream Mamba alignment without a shared-prefix chunk stop."""
    start = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens

    block_size = self.cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    if end < prefill_end:
        max_prefill_tokens = self.max_num_scheduled_tokens
        long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
        if long_prefill_threshold > 0:
            max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
        aligned_end = end // block_size * block_size
        if aligned_end > start or block_size <= max_prefill_tokens:
            end = aligned_end

    next_block_boundary = (start // block_size + 1) * block_size
    tail_boundary = (
        request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
        if self.mamba_partial_cache_hit
        else 0
    )
    stops = (
        next_block_boundary if start % block_size != 0 else 0,
        last_cache_position,
        tail_boundary
        if last_cache_position < tail_boundary < request.num_prompt_tokens
        else 0,
    )
    end = min((stop for stop in stops if start < stop < end), default=end)
    return max(end - start, 0)


@functools.wraps(_original_mamba_block_aligned_split)
def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Preserve PD windows and align sparse index-kpool cache groups."""
    kv_transfer_config = self.vllm_config.kv_transfer_config
    # A consumer only needs the unsplit verifier window after some prefix has
    # already been computed locally or loaded from the connector.  ``kv_both``
    # also handles cold prefills; bypassing alignment for those requests means
    # no reusable Mamba state is ever materialized, so neither HBM nor the KV
    # pool can cache the prefix.
    has_computed_prefix = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens > 0
    if kv_transfer_config is not None and kv_transfer_config.is_kv_consumer and has_computed_prefix:
        return num_new_tokens

    if _get_sparse_index_kpool(self.vllm_config.model_config) is not None:
        num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        if num_computed_tokens < max(
            request.num_prompt_tokens,
            request.num_tokens - 1,
        ):
            block_size = self.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            scheduled_end = num_computed_tokens + num_new_tokens
            if scheduled_end < last_cache_position:
                chunked_tokens = num_new_tokens // block_size * block_size
                if chunked_tokens > 0:
                    num_new_tokens = chunked_tokens
            elif num_computed_tokens < last_cache_position < scheduled_end:
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens

    return _split_without_shared_prefix_junction(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
    )


current_parameters = tuple(inspect.signature(_original_mamba_block_aligned_split).parameters)
if current_parameters != _EXPECTED_PARAMETERS:
    raise RuntimeError(
        "Cannot apply the PD consumer Mamba split patch: unexpected "
        "Scheduler._mamba_block_aligned_split signature "
        f"{current_parameters}"
    )

Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
