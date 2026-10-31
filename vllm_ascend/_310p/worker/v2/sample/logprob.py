# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors
"""Triton-free logprobs gather (MRV1 raw_logprobs layout)."""

from __future__ import annotations

import torch
from vllm.v1.outputs import LogprobsTensors


def gather_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: int,
    token_ids: torch.Tensor,
    cu_num_generated_tokens: list[int] | None = None,
) -> LogprobsTensors:
    token_ids_i64 = token_ids.to(torch.int64).view(-1, 1)
    token_logprobs = logprobs.gather(-1, token_ids_i64)
    token_ranks = (logprobs > token_logprobs).sum(dim=-1).to(torch.int32)
    if num_logprobs > 0:
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
        indices = torch.cat((token_ids_i64.to(torch.int32), topk_indices.to(torch.int32)), dim=1)
        values = torch.cat((token_logprobs, topk_logprobs), dim=1)
    else:
        indices = token_ids_i64.to(torch.int32)
        values = token_logprobs
    return LogprobsTensors(indices, values, token_ranks, cu_num_generated_tokens=cu_num_generated_tokens)
