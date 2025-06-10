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

from typing import Optional

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler

from vllm_ascend import envs


def apply_min_p(
    self,
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """
    Filters logits using adaptive probability thresholding.
    """
    # Convert logits to probability distribution
    probability_values = torch.nn.functional.softmax(logits, dim=-1)
    # Calculate maximum probabilities per sequence
    max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
    # Reshape min_p for broadcasting
    adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
    # Identify valid tokens using threshold comparison
    # Apply mask using boolean indexing
    logits = logits.masked_fill(probability_values < adjusted_min_p,
                                -float('inf'))
    return logits


def _apply_top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def topk_topp_forward_native(
    self,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    logits = _apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, generators)


Sampler.apply_min_p = apply_min_p
if envs.VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:
    TopKTopPSampler.forward_native = topk_topp_forward_native
