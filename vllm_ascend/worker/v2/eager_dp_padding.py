# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.
#
"""DP padding for fine-grained TP eager steps.

Fine-grained TP features whose collectives cross DP ranks (o_proj TP & mlp TP)
require every rank of the group to forward the same number of tokens per step.
Cudagraph dispatch guarantees that natively: the DP sync agrees one group-wide
token count and pads every rank to it. A step that degrades to eager keeps
per-rank token counts and the cross-DP collectives would hang.

Instead of rejecting eager steps, keep them DP-aligned with four pieces in
NPUModelRunner, without replacing any upstream binding:

1. `execute_model` agrees the step's DP-group token max G with one CPU
   all_reduce before dispatch (real steps report their token count, dummy
   steps report zero) — the same technique model runner v1's
   `_sync_metadata_across_dp` applied on every step.
2. `gather_batch_req_state` reports G as the batch's token count, so the
   dispatch input already carries the padded size. The eager branch then
   hands every rank a descriptor at G AND publishes counts[dp_rank] == G,
   keeping the forward-context invariant (counts[dp_rank] equals the padded
   forward row count) intact without touching the dispatch result.
3. `prepare_inputs` restores the real token extent for everything that
   treats num_tokens as real work (query_start_loc trailing fill,
   `InputBatch.num_tokens`); padded rows keep flowing through the
   cudagraph-padding machinery (PAD_SLOT_ID slot mappings, sampler reads
   only real request rows).
4. Dummy steps rewrite their synthetic scheduler output to G, so an idle
   rank's dummy batch forwards exactly G rows and joins the collectives at
   the same shape.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.core.sched.output import SchedulerOutput


def sync_dp_group_max_tokens(num_tokens: int, dp_size: int, dp_rank: int) -> int:
    """Agree the DP-group token-count max with one CPU all_reduce.

    Mirrors the one-hot reduce `dp_utils.sync_cudagraph_and_dp_padding`
    performs: every rank writes its own count into its slot, the sum
    all_reduce yields the full per-rank vector, and the max of that vector
    is identical on every rank. Dummy steps report zero so only real work
    raises the bar.
    """
    cpu_group = get_dp_group().cpu_group
    counts = torch.zeros(dp_size, dtype=torch.int32)
    counts[dp_rank] = num_tokens
    dist.all_reduce(counts, group=cpu_group)
    return int(counts.max().item())


def make_dp_padded_dummy_output(scheduler_output: SchedulerOutput, group_max: int) -> SchedulerOutput:
    """Rewrite a dummy scheduler output to forward exactly group_max tokens.

    The dummy output is synthetic (built by `_dummy_run`), so replacing
    its token accounting is safe. One synthetic request keeps the dummy
    uniform-decode shape gather_batch_req_state derives from it, and makes
    the eager dispatch report exactly group_max for this rank.
    """
    return replace(
        scheduler_output,
        num_scheduled_tokens={"_dummy_dp_padding": group_max},
        total_num_scheduled_tokens=group_max,
    )
