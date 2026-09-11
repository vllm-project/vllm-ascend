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

Cross-DP collectives (o_proj TP) need every rank to forward the same token
count per step. Cudagraph dispatch pads natively, eager dispatch would hang,
so `NPUModelRunner` aligns eager steps to the group max.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.core.sched.output import SchedulerOutput


def sync_dp_group_max_tokens(num_tokens: int, dp_size: int, dp_rank: int) -> int:
    """
    Agree the DP-group token-count max with one CPU all_reduce. Mirrors the one-hot
    reduce in `dp_utils.sync_cudagraph_and_dp_padding`.
    """
    cpu_group = get_dp_group().cpu_group
    counts = torch.zeros(dp_size, dtype=torch.int32)
    counts[dp_rank] = num_tokens
    dist.all_reduce(counts, group=cpu_group)
    return int(counts.max().item())


def make_dp_padded_dummy_output(
    scheduler_output: SchedulerOutput, group_max: int, decode_query_len: int, max_num_reqs: int
) -> SchedulerOutput:
    """
    Rewrite a dummy scheduler output to forward exactly `group_max` tokens. The output
    is synthetic, so rewriting it is safe.
    """
    # decode_query_len-sized requests stay decode-graph matchable; past max_num_reqs use _dummy_run's even split.
    num_full, remainder = divmod(group_max, decode_query_len)
    per_request = [decode_query_len] * num_full
    if remainder:
        per_request.append(remainder)
    if len(per_request) > max_num_reqs:
        num_reqs = min(group_max, max_num_reqs)
        per_request = [group_max // num_reqs + (i >= num_reqs - group_max % num_reqs) for i in range(num_reqs)]
    return replace(
        scheduler_output,
        num_scheduled_tokens={f"_dummy_dp_padding_{i}": n for i, n in enumerate(per_request)},
        total_num_scheduled_tokens=group_max,
    )
