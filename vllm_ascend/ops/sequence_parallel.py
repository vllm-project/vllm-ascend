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

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)

from vllm_ascend.sequence_parallel import (
    SequenceParallelCollective,
    SequenceParallelRuntimeState,
    plan_local_sequence_shard,
    plan_partial_reduction,
    plan_sequence_gather,
)


def _pad_tokens(tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    padding = tensor.new_zeros((pad_size, *tensor.shape[1:]))
    return torch.cat((tensor, padding), dim=0)


def reduce_partial_activation(
    tensor: torch.Tensor,
    state: SequenceParallelRuntimeState,
) -> tuple[torch.Tensor, SequenceParallelRuntimeState]:
    """Reduce a TP partial sum according to the runtime SP policy."""
    plan = plan_partial_reduction(state)
    if plan.collective is SequenceParallelCollective.ALL_REDUCE:
        output = tensor_model_parallel_all_reduce(tensor)
    else:
        padded = _pad_tokens(tensor, plan.pad_size)
        output = tensor_model_parallel_reduce_scatter(padded, 0)
    return output, plan.output_state


def gather_sequence_activation(
    tensor: torch.Tensor,
    state: SequenceParallelRuntimeState,
) -> tuple[torch.Tensor, SequenceParallelRuntimeState]:
    """Gather sequence shards and remove communication padding."""
    plan = plan_sequence_gather(state)
    output = tensor_model_parallel_all_gather(tensor, 0)
    if plan.unpad_size > 0:
        output = output[: -plan.unpad_size]
    return output, plan.output_state


def shard_sequence_activation(
    tensor: torch.Tensor,
    state: SequenceParallelRuntimeState,
) -> tuple[torch.Tensor, SequenceParallelRuntimeState]:
    """Pad and select the local TP token shard without communication."""
    plan = plan_local_sequence_shard(state)
    padded = _pad_tokens(tensor, plan.pad_size)
    rank = get_tensor_model_parallel_rank()
    output = torch.chunk(padded, state.world_size, dim=0)[rank]
    return output, plan.output_state
