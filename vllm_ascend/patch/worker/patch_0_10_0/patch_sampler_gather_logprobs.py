#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor,
                               values: torch.Tensor) -> torch.Tensor:
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.
    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).
    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    return (x >= values).sum(-1)


def gather_logprobs(
    self,
    logprobs: torch.Tensor,
    num_logprobs: int,
    token_ids: torch.Tensor,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        num_logprobs: minimum number of logprobs to
                    retain per token
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
                    Must be int64.

    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    assert token_ids.dtype == torch.int64
    # Find the topK values.
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

    # Get with the logprob of the prompt or sampled token.
    token_ids = token_ids.unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, token_ids)

    # Compute the ranks of the actual token.
    token_ranks = batched_count_greater_than(logprobs, token_logprobs)

    # Concatenate together with the topk.
    indices = torch.cat((token_ids, topk_indices), dim=1)
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

    # Use int32 to reduce the tensor size.
    indices = indices.to(torch.int32)

    return LogprobsTensors(indices, logprobs, token_ranks)


Sampler.gather_logprobs = gather_logprobs
