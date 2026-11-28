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

On a pure PD prefill producer or a standalone instance the EAGLE one-block
backoff of the last cacheable position is suppressed: matched content-hash
blocks are always verified prompt blocks there, while the backoff would leave
the final full mamba-align state page unmaterialized across the chunk boundary
(copy-on-write in MambaManager.allocate_new_blocks), pinning hybrid prefix
hits one full page (or entirely) short.
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

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split


def _elide_first_tail_checkpoint(
    scheduler: Scheduler, request: Request, start: int, num_new_tokens: int, aligned_tokens: int
) -> int:
    if getattr(request, "_ascend_gdn_promotion", 0):
        request._ascend_gdn_promotion = 0
    if not getattr(scheduler, "mamba_partial_cache_hit", False) or scheduler.mamba_has_prefill_checkpoint_blocks:
        return aligned_tokens

    end = start + num_new_tokens
    page_size = scheduler.cache_config.block_size
    if (
        end != request.num_prompt_tokens
        or end != request.num_tokens
        or start // page_size != (end - 1) // page_size
        or (scheduler.mamba_shared_prefix_checkpoint and start < request.shared_prefix_boundary < end)
    ):
        return aligned_tokens

    tail = get_mamba_prefill_checkpoint_position(
        request.num_prompt_tokens,
        scheduler.hash_block_size,
        False,
        num_hash_lookahead_tokens=scheduler.num_hash_lookahead_tokens,
    )
    if not start < tail < end or aligned_tokens != num_new_tokens and start + aligned_tokens != tail:
        return aligned_tokens

    from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager, MambaManager

    coordinator = scheduler.kv_cache_manager.coordinator
    managers = [m for m in coordinator.single_type_managers if isinstance(m, MambaManager)]
    if managers and all(
        isinstance(m, AscendMambaManager)
        and m.block_size == page_size
        and not m.has_prefill_checkpoint_blocks
        and m.checkpoint_candidate_key(request, tail) is not None
        for m in managers
    ):
        manager = managers[0]
        step = scheduler.hash_block_size
        lower_bound = start if getattr(coordinator, "supports_gdn_interior_promotion", False) else tail - step
        for position in range(tail, lower_bound, -step):
            if manager.has_repeated_checkpoint(request, position):
                if position != tail:
                    request._ascend_gdn_promotion = position
                return position - start
        return num_new_tokens
    return aligned_tokens


@functools.wraps(_original_mamba_block_aligned_split)
def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Preserve PD windows and align sparse index-kpool cache groups.

    On a pure PD prefill producer or a standalone instance the EAGLE
    one-block backoff of the last cacheable position is suppressed.
    ``Scheduler._mamba_block_aligned_split`` backs the last cacheable
    mamba-align page off by one block (and, on newer vLLM revisions, shifts
    the partial-tail checkpoint boundary) whenever the EAGLE block drop is
    active. Consequently a producer prefill can never end a chunk at its
    final full-page boundary. In mamba "align" mode the recurrent state of a
    page is only materialized across a chunk boundary (copy-on-write in
    ``MambaManager.allocate_new_blocks``), so the suppressed split leaves the
    final full state page unhashed. Hybrid coordinator hits reconcile to the
    per-group minimum: even with the full-attention group fixed by the
    coordinator exemption, the mamba groups report one full page less
    (1600-token prompts -> 0 hit, 3200-token prompts -> 1536 with 1536-token
    align pages) - the observed MTP prefix-cache kill band.

    Rather than copy the scheduler method (its body moves between vLLM
    revisions), invoke the original with the drop bit temporarily cleared:
    every read of the bit inside the method exists solely to compensate for
    the block drop, and on the producer matched blocks are always verified
    prompt blocks and the coordinator never drops. Scheduling is
    single-threaded per scheduler instance, so the temporary toggle is safe.
    vLLM 0.28.x names the bit ``use_eagle``; newer revisions expose the
    dedicated ``use_eagle_block_drop`` knob.

    The suppression self-gates at call time on
    ``self.vllm_config.kv_transfer_config`` via ``_skips_eagle_block_drop``
    (the same PD role source used by the coordinator and the neighboring
    mamba split logic), so consumers and ``kv_both`` instances pass straight
    through with upstream behavior. Standalone instances (no connector)
    suppress the backoff too: the coordinator never drops there, so backing
    the split off would only erase cacheable hit length.
    """
    kv_transfer_config = self.vllm_config.kv_transfer_config
    # Extra verifier or asynchronous output rows need an unsplit window.
    # Ordinary prefills must still stop at reusable checkpoints.
    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    remaining_tokens = max(request.num_tokens - num_computed_tokens, 0)
    if (
        kv_transfer_config is not None
        and kv_transfer_config.is_kv_consumer
        and num_computed_tokens > 0
        and num_new_tokens > remaining_tokens
    ):
        return num_new_tokens

    # Pure PD prefill producer or standalone instance: suppress the EAGLE
    # one-block backoff (see the docstring). The sparse index-kpool
    # early-return path below must observe it too (it previously ran under
    # a temporary bit clear from the outer producer wrapper).
    if _get_sparse_index_kpool(self.vllm_config.model_config) is not None:
        if num_computed_tokens < max(
            request.num_prompt_tokens,
            request.num_tokens - 1,
        ):
            block_size = self.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            if self.use_eagle and not _skips_eagle_block_drop(kv_transfer_config):
                last_cache_position = max(last_cache_position - block_size, 0)
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
        return _elide_first_tail_checkpoint(self, request, num_computed_tokens, num_new_tokens, result)
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
