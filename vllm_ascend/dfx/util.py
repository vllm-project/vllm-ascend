#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Small shared helpers used across DFX modules (no DFX dependencies)."""

from __future__ import annotations

from typing import Any

import torch


def is_int_list(value: Any) -> bool:
    """True when ``value`` is a non-empty ``list[int]`` (bool excluded)."""
    return (
        isinstance(value, list) and bool(value) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def is_list_of_int_lists(value: Any) -> bool:
    """True when ``value`` is a non-empty list of int lists."""
    return isinstance(value, list) and bool(value) and all(is_int_list(x) for x in value)


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token-id list to text (``skip_special_tokens=False``)."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def accepted_token_counts(
    sampled_token_ids: Any,
    *,
    placeholder_token_id: int = -1,
) -> Any:
    """Count accepted tokens per request from rejection-sampler output.

    Used for non-hybrid MTP / speculative paths where accepted counts are
    derived from ``PLACEHOLDER_TOKEN_ID`` padding rather than a dedicated
    ``num_accepted_tokens`` buffer.
    """
    if sampled_token_ids is None:
        return []
    if torch.is_tensor(sampled_token_ids):
        if sampled_token_ids.numel() == 0:
            return torch.zeros(sampled_token_ids.size(0), dtype=torch.int32)
        return (sampled_token_ids != placeholder_token_id).sum(dim=-1).to(dtype=torch.int32).cpu()
    counts: list[int] = []
    for row in sampled_token_ids:
        if row is None:
            counts.append(0)
            continue
        if torch.is_tensor(row):
            counts.append(int((row != placeholder_token_id).sum().item()))
        else:
            counts.append(sum(1 for t in row if t != placeholder_token_id))
    return counts
