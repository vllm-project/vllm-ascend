#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Helpers for the DP metadata sync that the vLLM Ascend model runner skips."""

from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def align_skip_drafts_across_dp(
    scheduler_output: "SchedulerOutput",
    dp_size: int,
    skip_dp_metadata_sync: bool,
) -> "SchedulerOutput":
    """Return a scheduler output whose ``skip_drafts`` flag is safe for the DP group.

    Upstream vLLM agrees on ``skip_drafts`` across the DP group inside
    ``sync_cudagraph_and_dp_padding``: the flag travels in the same tensor
    that carries the padding metadata, and the all-reduce over that tensor
    turns a per-rank value into a group-wide one. When vLLM Ascend skips that
    all-reduce (``skip_dp_coordination()``), every rank keeps its own value,
    so two DP ranks can disagree.

    The disagreement is fatal for speculative decoding. The rank that
    finished its last decode step reports ``skip_drafts=True`` and skips the
    ``K - 1`` drafter passes, while a peer that is still decoding runs all
    ``K`` of them. The unmatched EP/MC2 collectives then deadlock the engine
    at the next synchronization point.

    Clearing the flag on every rank restores the agreement that the skipped
    all-reduce would have produced: all ranks run the drafter, so they
    enqueue the same number of collectives. ``skip_drafts`` is a pure
    optimization (it only tells the drafter that no scheduled request can use
    the proposed drafts), so dropping it does not change the sampled tokens.

    Args:
        scheduler_output: The scheduler output of the current step.
        dp_size: Number of ranks in the DP group.
        skip_dp_metadata_sync: Whether this rank skips the DP metadata
            all-reduce.

    Returns:
        ``scheduler_output`` unchanged when the flag is already safe, and a
        shallow copy with ``skip_drafts`` cleared otherwise.
    """
    if not skip_dp_metadata_sync or dp_size <= 1:
        return scheduler_output
    if not getattr(scheduler_output, "skip_drafts", False):
        return scheduler_output
    aligned = copy(scheduler_output)
    aligned.skip_drafts = False
    return aligned
