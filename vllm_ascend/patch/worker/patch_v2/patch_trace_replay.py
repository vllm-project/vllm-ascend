# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Add the Ascend-specific empty-batch guard to upstream trace replay."""

from __future__ import annotations

from functools import wraps

from vllm.v1.worker.gpu.sample import trace_replay


_upstream_apply_trace_tokens = trace_replay.apply_trace_tokens


@wraps(_upstream_apply_trace_tokens)
def apply_trace_tokens(
    sampled,
    idx_mapping,
    trace_token_ids,
    trace_len,
    total_len,
    prompt_len,
) -> None:
    # CUDA accepts a zero-size Triton grid, while Ascend rejects coreDim=0.
    if sampled.shape[0] == 0:
        return

    _upstream_apply_trace_tokens(
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_len,
        total_len,
        prompt_len,
    )


trace_replay.apply_trace_tokens = apply_trace_tokens
