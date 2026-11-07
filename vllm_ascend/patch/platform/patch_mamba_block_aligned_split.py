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

Decode consumers preserve the complete verifier window. When core enables
partial cache hits, native splitting chooses prompt-tail and shared-prefix
checkpoints for continued prefill. Coarse-cache sparse index-kpool producers
retain the resolved common cache-group alignment. Other models retain the
upstream behavior.

Current vLLM resolves MTP lookahead hashing and checkpoint backoff together.
Only the v0.32 pairing retains its legacy producer/standalone exemption.
"""

import functools
import inspect

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _skips_eagle_block_drop,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
)
from vllm_ascend.utils import vllm_version_is

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split


@functools.wraps(_original_mamba_block_aligned_split)
def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Preserve Ascend verifier windows and native checkpoint boundaries."""
    kv_transfer_config = self.vllm_config.kv_transfer_config
    # A consumer only needs the unsplit verifier window after some prefix has
    # already been computed locally or loaded from the connector.  ``kv_both``
    # also handles cold prefills; bypassing alignment for those requests means
    # no reusable Mamba state is ever materialized, so neither HBM nor the KV
    # pool can cache the prefix.
    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    if (
        kv_transfer_config is not None
        and kv_transfer_config.is_kv_consumer
        and num_computed_tokens > 0
        and (not self.mamba_partial_cache_hit or num_computed_tokens >= request.num_tokens - 1)
    ):
        return num_new_tokens

    # Native partial-cache eligibility also owns the prompt-tail/shared-prefix
    # checkpoint stops. Do not bypass those stops with physical-page splitting.
    if not self.mamba_partial_cache_hit and _get_sparse_index_kpool(self.vllm_config.model_config) is not None:
        if num_computed_tokens < max(
            request.num_prompt_tokens,
            request.num_tokens - 1,
        ):
            block_size = self.block_size
            drop_eagle = getattr(self, "use_eagle_block_drop", self.use_eagle)
            if vllm_version_is("0.32.0"):
                last_cache_position = request.num_tokens - request.num_tokens % block_size
                if drop_eagle and not _skips_eagle_block_drop(kv_transfer_config):
                    last_cache_position = max(last_cache_position - block_size, 0)
            else:
                last_cache_position = get_mamba_prefill_checkpoint_position(
                    request.num_tokens,
                    block_size,
                    drop_eagle,
                    self.num_hash_lookahead_tokens,
                )
            scheduled_end = num_computed_tokens + num_new_tokens
            if scheduled_end < last_cache_position:
                chunked_tokens = num_new_tokens // block_size * block_size
                if chunked_tokens > 0:
                    num_new_tokens = chunked_tokens
            elif num_computed_tokens < last_cache_position < scheduled_end:
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens

    if not _skips_eagle_block_drop(kv_transfer_config):
        return _original_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
    # vLLM 0.28.x and newer revisions use different names.  Some
    # transitional scheduler implementations expose both and different
    # wrapper layers consult different attributes, so clear every
    # attribute that exists and restore all of them after the call.
    drop_attrs = tuple(name for name in ("use_eagle", "use_eagle_block_drop") if hasattr(self, name))
    original_drop_values = {name: getattr(self, name) for name in drop_attrs}
    for name in drop_attrs:
        setattr(self, name, False)
    try:
        result = _original_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        # A resend must recompute its last input token for logits. When the
        # prompt ends exactly on a state-page boundary, its reusable state is
        # therefore one page earlier. Materialize that checkpoint before the
        # final chunk; retaining only the terminal state makes the first hot
        # replay miss every Mamba group. PD crops one token before scheduling,
        # so this also covers API prompts of k * block_size + 1 tokens.
        block_size = self.cache_config.block_size
        if (
            getattr(self.cache_config, "enable_prefix_caching", True)
            and self.hash_block_size == block_size
            and not getattr(self, "mamba_has_prefill_checkpoint_blocks", False)
            and request.num_prompt_tokens % block_size == 0
        ):
            start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
            replay_boundary = max(request.num_prompt_tokens - block_size, 0)
            if start < replay_boundary < start + result:
                result = replay_boundary - start
        return result
    finally:
        for name, value in original_drop_values.items():
            setattr(self, name, value)


current_parameters = tuple(inspect.signature(_original_mamba_block_aligned_split).parameters)
if current_parameters != _EXPECTED_PARAMETERS:
    raise RuntimeError(
        "Cannot apply the PD consumer Mamba split patch: unexpected "
        "Scheduler._mamba_block_aligned_split signature "
        f"{current_parameters}"
    )

Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
