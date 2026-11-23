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
from collections import OrderedDict

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request

from vllm_ascend.core.gdn_checkpoint_policy import GDNCheckpointPolicy
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _skips_eagle_block_drop,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager

logger = init_logger(__name__)

_EXPECTED_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split
_original_scheduler_init = Scheduler.__init__
_original_add_request = Scheduler.add_request
_original_reset_prefix_cache = Scheduler.reset_prefix_cache
_original_schedule = Scheduler.schedule


@functools.wraps(_original_scheduler_init)
def _init_scheduler(self: Scheduler, *args, **kwargs) -> None:
    _original_scheduler_init(self, *args, **kwargs)
    self._ascend_gdn_checkpoint_policy = GDNCheckpointPolicy.from_config(self.vllm_config)
    policy = self._ascend_gdn_checkpoint_policy
    self._ascend_seen_mamba_tails = None
    self._ascend_short_prefill_interval = 1
    options = self.vllm_config.additional_config or {}
    if (
        options.get("gdn_short_prefill_schedule_interval") == 2
        and policy is not None
        and policy.max_tokens
        and self.mamba_partial_cache_hit
        and self.local_prefill_interval == 1
        and self.parallel_config.data_parallel_size == 1
    ):
        self._ascend_short_prefill_interval = 2
    if policy is None or not self.mamba_has_prefill_checkpoint_blocks:
        return
    for manager in self.kv_cache_manager.coordinator.single_type_managers:
        if not getattr(manager, "has_prefill_checkpoint_blocks", False):
            continue
        if not isinstance(manager, AscendMambaManager):
            raise RuntimeError("Native GDN checkpoint policy requires the Ascend Mamba cache manager")
        manager.gdn_checkpoint_policy = policy
    if policy.elide_singleton_warm_tail and self.mamba_partial_cache_hit:
        self._ascend_seen_mamba_tails = OrderedDict()


@functools.wraps(_original_schedule)
def _schedule(self: Scheduler, throttle_prefills: bool = False):
    interval = getattr(self, "_ascend_short_prefill_interval", 1)
    requests = self.requests
    if interval == 1 or not requests:
        return _original_schedule(self, throttle_prefills)

    limit = self._ascend_gdn_checkpoint_policy.max_tokens
    smallest_prompt, largest_prompt = limit, 0
    for request in requests.values():
        prompt_tokens = request.num_prompt_tokens
        if prompt_tokens + request.max_tokens > limit:
            if not getattr(self, "_ascend_cadence_long_seen", False):
                logger.info("Local PMU prefill cadence retains default for context above %d tokens", limit)
                self._ascend_cadence_long_seen = True
            return _original_schedule(self, throttle_prefills)
        smallest_prompt = min(smallest_prompt, prompt_tokens)
        largest_prompt = max(largest_prompt, prompt_tokens)

    # Spacing prefills can help uniform short inputs. When active prompts
    # span more than one hash-block length, a single deferred step can
    # instead delay a continuation behind a different prefill shape.
    if largest_prompt - smallest_prompt >= self.hash_block_size:
        if not getattr(self, "_ascend_cadence_mixed_seen", False):
            logger.info("Local PMU retains default prefill cadence for mixed prompt lengths")
            self._ascend_cadence_mixed_seen = True
        return _original_schedule(self, throttle_prefills)

    # A batch of one-page prompts needs very little prefill after a warm
    # partial hit. One additional decode step reduces prefill/decode mixing.
    if smallest_prompt == largest_prompt == self.cache_config.block_size:
        interval = 3
        if not getattr(self, "_ascend_cadence_one_page_seen", False):
            logger.info("Local PMU uses prefill cadence %d for uniform one-page prompts", interval)
            self._ascend_cadence_one_page_seen = True
    if not getattr(self, "_ascend_cadence_short_seen", False):
        logger.info("Local PMU uses prefill cadence %d for context at most %d tokens", interval, limit)
        self._ascend_cadence_short_seen = True
    # Reuse the core cadence and its capacity escape for a uniform short batch.
    # The next scheduling call restores ordinary behavior if a long request arrived.
    previous_interval = self.local_prefill_interval
    self.local_prefill_interval = interval
    try:
        return _original_schedule(self, throttle_prefills)
    finally:
        self.local_prefill_interval = previous_interval


@functools.wraps(_original_add_request)
def _add_request(self: Scheduler, request: Request) -> None:
    history = self._ascend_seen_mamba_tails
    if (
        history is not None
        and request.request_id not in self.requests
        and not request.resumable
        and self.cache_config.block_size <= request.num_prompt_tokens
    ):
        block_size = self.cache_config.block_size
        tail = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            self.hash_block_size,
            self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )
        hash_index = tail // self.hash_block_size - 1
        if (
            (tail > block_size or request.num_prompt_tokens == block_size)
            and tail % block_size
            and 0 <= hash_index < len(request.block_hashes)
        ):
            key = request.block_hashes[hash_index]
            request._ascend_first_observed_mamba_tail = key not in history
            history[key] = None
            history.move_to_end(key)
            if len(history) > 4096:
                history.popitem(last=False)
    _original_add_request(self, request)


@functools.wraps(_original_reset_prefix_cache)
def _reset_prefix_cache(self: Scheduler, *args, **kwargs) -> bool:
    success = _original_reset_prefix_cache(self, *args, **kwargs)
    if success and self._ascend_seen_mamba_tails is not None:
        self._ascend_seen_mamba_tails.clear()
    return success


def _split_with_gdn_policy(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> int:
    def upstream_split() -> int:
        return _original_mamba_block_aligned_split(
            self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )

    policy = getattr(self, "_ascend_gdn_checkpoint_policy", None)
    if policy is None or not self.mamba_has_prefill_checkpoint_blocks:
        return upstream_split()
    start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    end = start + num_new_tokens
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if not start < prefill_end <= end:
        return upstream_split()
    block_size = self.cache_config.block_size
    partial_cache_hit = self.mamba_partial_cache_hit
    checkpoint = get_mamba_prefill_checkpoint_position(
        prefill_end,
        self.hash_block_size if partial_cache_hit else block_size,
        self.use_eagle_block_drop,
        num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
    )
    next_page_boundary = (start // block_size + 1) * block_size
    skip_tail = (
        partial_cache_hit
        and getattr(request, "_ascend_first_observed_mamba_tail", False)
        and start > 0
        and (start == num_new_local_computed_tokens or (policy.max_tokens and prefill_end > policy.max_tokens))
        and (
            start % block_size == 0
            or prefill_end == next_page_boundary
            or (start >= block_size and 0 < policy.max_tokens >= prefill_end)
        )
        and start < checkpoint < prefill_end <= next_page_boundary
        and prefill_end == request.num_prompt_tokens
    )
    # An unaligned warm hit cannot export a second internal checkpoint
    # into the same state page. For first-observed short continuations after
    # the first physical page, avoid a separate tail-only forward. Keep the
    # existing checkpoint behavior for repeats and for nonaligned first-page
    # endings, where no completed physical page precedes the continuation.
    self.mamba_has_prefill_checkpoint_blocks = not skip_tail and policy.allows_internal_checkpoint(
        checkpoint, end, block_size
    )
    self.mamba_partial_cache_hit = partial_cache_hit and not skip_tail
    try:
        return upstream_split()
    finally:
        self.mamba_has_prefill_checkpoint_blocks = True
        self.mamba_partial_cache_hit = partial_cache_hit


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
        result = _split_with_gdn_policy(
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
Scheduler.__init__ = _init_scheduler
Scheduler.schedule = _schedule
Scheduler.add_request = _add_request
Scheduler.reset_prefix_cache = _reset_prefix_cache
