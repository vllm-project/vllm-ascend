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

"""Token-id normalization / sampling-row helpers shared across runtime_guard."""

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


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize tensor / nested tensors / sequences to ``list[int]``."""
    if token_ids is None:
        return []
    if torch.is_tensor(token_ids):
        return [int(x) for x in token_ids.tolist()]
    out: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
            out.append(int(token_id.item()))
        else:
            out.append(int(token_id))
    return out


def filter_valid_token_ids(token_ids: Any) -> list[int]:
    """Normalize and drop async / pad placeholders (``-1``)."""
    return [tid for tid in normalize_token_ids(token_ids) if tid != -1]


def trim_sampled_rows(sampled_rows: Any, num_sampled: Any) -> list[list[int]]:
    """Trim each per-req row to ``num_sampled[i]`` (mirrors ``AsyncOutput.get_output``).

    v2 sampler tensors are padded to a fixed width; only the first
    ``num_sampled[i]`` ids are valid output for request ``i``. Used by UTs and
    any path that must consume pre-trim rows safely.
    """
    if sampled_rows is None:
        return []
    if hasattr(sampled_rows, "tolist"):
        rows = sampled_rows.tolist()
    else:
        rows = list(sampled_rows)
    if num_sampled is None:
        return [filter_valid_token_ids(row) for row in rows]
    if hasattr(num_sampled, "tolist"):
        counts = num_sampled.tolist()
    else:
        counts = list(num_sampled)
    out: list[list[int]] = []
    for i, row in enumerate(rows):
        n = int(counts[i]) if i < len(counts) else 0
        if n <= 0:
            out.append([])
            continue
        if isinstance(row, (list, tuple)):
            out.append(filter_valid_token_ids(row[:n]))
        else:
            # Scalar / 0-d: keep only when n > 0.
            out.append(filter_valid_token_ids([row]))
    return out


def freeze_sampled_rows(req_ids: list[str] | None, sampled_rows: Any) -> list[list[int]]:
    """Host copy of this step's per-req sampled ids (no shared mutation)."""
    ids = list(req_ids or [])
    out: list[list[int]] = []
    for i, _rid in enumerate(ids):
        try:
            row = sampled_rows[i] if sampled_rows is not None else None
        except (IndexError, TypeError, KeyError):
            out.append([])
            continue
        out.append(filter_valid_token_ids(row))
    return out


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


def load_model_tokenizer(runner: Any) -> Any | None:
    """Load model tokenizer via ``cached_tokenizer_from_config``.

    Returns ``None`` if runner/config missing; raises if the load itself fails.
    """
    if runner is None:
        return None
    from vllm.tokenizers import cached_tokenizer_from_config

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    if model_config is None:
        return None
    return cached_tokenizer_from_config(model_config)
