# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/logprob.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val, propagate_nan=tl.PropagateNan.ALL))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - lse - max_val
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=12944,
        PADDED_TOPK=max(triton.next_power_of_2(num_logprobs), 2),
        multibuffer=False,
    )
    return logprobs


# v35 partitioned collect (RinshannKaihou collect.py `_v35_partials` / `_v35_finish`):
# https://github.com/RinshannKaihou/self_evolving_inference_detector/blob/main/src/v35_detector/serving/collect.py
# Gate: Ascend Triton (not CUDA-only). Same math as full log_softmax max/gather:
# top1 = -log S, chosen = (x_c - M) - log S.
_V35_BLOCK = 4096
# device -> (partial max, partial sum) workspace reused across steps.
_v35_workspace: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _v35_partials(X, stride, PMax, PSum, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * stride + offs, mask=offs < N, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    s = tl.sum(tl.exp(x - m), axis=0)
    s = tl.where(m == -float("inf"), 0.0, s)
    tl.store(PMax + row * PARTS + part, m)
    tl.store(PSum + row * PARTS + part, s)


@triton.jit
def _v35_finish(PMax, PSum, X, stride, Sampled, Out, N: tl.constexpr, PARTS: tl.constexpr, BLOCK_P: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    pm = tl.load(PMax + row * PARTS + offs, mask=offs < PARTS, other=-float("inf"))
    ps = tl.load(PSum + row * PARTS + offs, mask=offs < PARTS, other=0.0)
    m = tl.max(pm, axis=0)
    s = tl.sum(tl.where(pm == -float("inf"), 0.0, ps * tl.exp(pm - m)), axis=0)
    log_s = tl.log(s)
    tok = tl.load(Sampled + row).to(tl.int64)
    ok = (tok >= 0) & (tok < N)
    xc = tl.load(X + row.to(tl.int64) * stride + tok, mask=ok, other=float("nan")).to(tl.float32)
    bad = (m == -float("inf")) | (m == float("inf"))
    tl.store(Out + row * 2, tl.where(bad, float("nan"), -log_s))
    tl.store(Out + row * 2 + 1, tl.where(bad, float("nan"), (xc - m) - log_s))


def compute_top1_and_chosen_logprobs(
    logits: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    *,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Sampler-side narrow collect: top-1 and chosen logprobs from raw logits.

    Uses the v35 two-kernel Triton path (partitioned max+sum-exp, then LSE +
    gather) — no full logprob table, no top-k list, no ranks. Always launches
    Ascend Triton (does not gate on ``is_cuda``). Intended for Guard
    ``rare_chars`` / rc3-style detectors that only need ``(lp1, lpc)``.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [rows, vocab], got shape={tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    ids = chosen_token_ids.reshape(-1).to(dtype=torch.int64, device=logits.device)
    if ids.numel() != batch_size:
        raise ValueError(f"chosen_token_ids length {ids.numel()} != logits rows {batch_size}")

    raw = logits if logits.stride(1) == 1 else logits.contiguous()
    if ids.stride(0) != 1:
        ids = ids.contiguous()

    parts = triton.cdiv(vocab_size, _V35_BLOCK)
    device_key = str(raw.device)
    ws = _v35_workspace.get(device_key)
    if ws is None or ws[0].shape[0] < batch_size or ws[0].shape[1] != parts:
        cap = triton.cdiv(batch_size, 64) * 64
        ws = _v35_workspace[device_key] = (
            torch.empty((cap, parts), dtype=torch.float32, device=raw.device),
            torch.empty((cap, parts), dtype=torch.float32, device=raw.device),
        )

    out = torch.empty((batch_size, 2), dtype=torch.float32, device=raw.device)
    block_p = max(16, triton.next_power_of_2(parts))
    _v35_partials[(batch_size, parts)](
        raw,
        raw.stride(0),
        ws[0],
        ws[1],
        vocab_size,
        parts,
        _V35_BLOCK,
        num_warps=4,
    )
    _v35_finish[(batch_size,)](
        ws[0],
        ws[1],
        raw,
        raw.stride(0),
        ids,
        out,
        vocab_size,
        parts,
        block_p,
        num_warps=1,
    )
    if packed:
        return out
    return out[:, 0], out[:, 1]


@triton.jit(do_not_specialize=["batch_size", "rows_per_core"])
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    batch_size,
    rows_per_core,
    BLOCK_SIZE: tl.constexpr,
):
    core_id = tl.program_id(0)

    start_row = core_id * rows_per_core
    end_row = start_row + rows_per_core

    for req_idx in range(start_row, end_row):
        if req_idx < batch_size:
            row_ptr = logits_ptr + req_idx * logits_stride

            token_id = tl.load(token_ids_ptr + req_idx)
            x = tl.load(row_ptr + token_id)

            n_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for i in range(0, vocab_size, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
                n_vec += (logits > x).to(tl.int32)
            n = tl.sum(n_vec)
            tl.store(output_ptr + req_idx, n)


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids != 0:
        logger.warning_once("Custom logprob_token_ids is not supported yet. Falling back to default logprob_token_ids.")

    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
    logprobs = compute_token_logprobs(logits, logprob_token_ids)

    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )

    vec_core = get_vectorcore_num()
    NUM_CORES = min(batch_size, vec_core)

    rows_per_core = triton.cdiv(batch_size, NUM_CORES)
    BLOCK_SIZE = 8192
    grid = (NUM_CORES,)
    _ranks_kernel[grid](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        batch_size,
        rows_per_core,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )

    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )
