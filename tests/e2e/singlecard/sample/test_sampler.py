#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
# Copyright 2023 The vLLM team.
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

# Set tolerance to 1 for quant ops
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_top_k_top_p_new(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Apply top-k.
    boundary = logits_sort.gather(1, (vocab_size - k).unsqueeze(dim=1))
    top_k_mask = logits_sort < boundary
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        cutoff = top_k_mask.sum(dim=-1).min()
        probs_sort = logits_sort.softmax(dim=-1)[:, cutoff:]
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum > 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = True
        strides = torch.arange(0,
                               batch_size * vocab_size,
                               vocab_size,
                               device=logits.device)
        flatten_idx = logits_idx[:, cutoff:] + strides.unsqueeze(dim=1)
        valid_idx = torch.masked_select(flatten_idx, top_p_mask)
        logits_flatten = logits.flatten()
        valid_logits = torch.index_select(logits_flatten, 0, valid_idx)
        logits = torch.empty_like(logits_flatten).fill_(-float("inf"))
        logits[valid_idx] = valid_logits
    return logits.reshape(batch_size, vocab_size)


# test with leading dimension and merge seqlen and batch_size as num_tokens
@torch.inference_mode()
def test_apply_top_k_top_p() -> None:
    logits = torch.randn((128, 7168)).npu()
    k = torch.Tensor([-1]).int().npu()
    p = torch.Tensor([1]).int().npu()
    logits_new = apply_top_k_top_p_new(logits, k, p)
    logits_old = apply_top_k_top_p(logits, k, p)
    # Compare the results.
    torch.testing.assert_close(logits_new,
                               logits_old,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
