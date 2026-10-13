# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import NamedTuple

import torch
from vllm.triton_utils import tl, triton

_PREPARE_BLOCK_SIZE = 128


class SamplingInputs(NamedTuple):
    positions: torch.Tensor
    input_ids: torch.Tensor
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor


@triton.jit(do_not_specialize=["num_reqs"])
def _prepare_sampling_inputs_kernel(
    positions,
    input_ids,
    logits_indices,
    idx_mapping,
    expanded_idx_mapping,
    seq_lens,
    cu_num_logits,
    prefill_len,
    top_k,
    top_p,
    out_positions,
    out_input_ids,
    out_top_k,
    out_top_p,
    num_sampled,
    num_rejected,
    num_reqs,
    positions_stride: tl.constexpr,
    input_ids_stride: tl.constexpr,
    logits_indices_stride: tl.constexpr,
    idx_mapping_stride: tl.constexpr,
    expanded_idx_mapping_stride: tl.constexpr,
    seq_lens_stride: tl.constexpr,
    cu_num_logits_stride: tl.constexpr,
    prefill_len_stride: tl.constexpr,
    top_k_stride: tl.constexpr,
    top_p_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = row < num_reqs
    token = tl.load(logits_indices + row * logits_indices_stride, valid, 0)
    position = tl.load(positions + token * positions_stride, valid, 0)
    input_id = tl.load(input_ids + token * input_ids_stride, valid, 0)
    tl.store(out_positions + row, position, valid)
    tl.store(out_input_ids + row, input_id, valid)
    sample_idx = tl.load(expanded_idx_mapping + row * expanded_idx_mapping_stride, valid, 0)
    if top_k is not None:
        k = tl.load(top_k + sample_idx * top_k_stride, valid, 0)
        tl.store(out_top_k + row, k, valid)
    if top_p is not None:
        p = tl.load(top_p + sample_idx * top_p_stride, valid, 1.0)
        tl.store(out_top_p + row, p, valid)
    req_idx = tl.load(idx_mapping + row * idx_mapping_stride, valid, 0)
    length = tl.load(seq_lens + row * seq_lens_stride, valid, 0)
    prefill = tl.load(prefill_len + req_idx * prefill_len_stride, valid, 0)
    start = tl.load(cu_num_logits + row * cu_num_logits_stride, valid, 0)
    end = tl.load(cu_num_logits + (row + 1) * cu_num_logits_stride, valid, 0)
    done = length >= prefill
    sampled = tl.where(done, 1, 0)
    rejected = tl.where(done, end - start - sampled, 0)
    tl.store(num_sampled + row, sampled, valid)
    tl.store(num_rejected + row, rejected, valid)


def prepare_sampling_inputs(
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    prefill_len: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> SamplingInputs:
    """Gather one logit row per request and preserve chunked-prefill counts.

    Request mappings must be nonnegative. Speculative and padded request
    batches keep the upstream preparation path.
    """
    num_reqs = idx_mapping.numel()
    output = SamplingInputs(
        positions.new_empty(num_reqs),
        input_ids.new_empty(num_reqs),
        top_k.new_empty(num_reqs) if top_k is not None else None,
        top_p.new_empty(num_reqs) if top_p is not None else None,
        seq_lens.new_empty(num_reqs),
        seq_lens.new_empty(num_reqs),
    )
    _prepare_sampling_inputs_kernel[(triton.cdiv(num_reqs, _PREPARE_BLOCK_SIZE),)](
        positions,
        input_ids,
        logits_indices,
        idx_mapping,
        expanded_idx_mapping,
        seq_lens,
        cu_num_logits,
        prefill_len,
        top_k,
        top_p,
        *output,
        num_reqs,
        positions.stride(0),
        input_ids.stride(0),
        logits_indices.stride(0),
        idx_mapping.stride(0),
        expanded_idx_mapping.stride(0),
        seq_lens.stride(0),
        cu_num_logits.stride(0),
        prefill_len.stride(0),
        top_k.stride(0) if top_k is not None else 1,
        top_p.stride(0) if top_p is not None else 1,
        BLOCK_SIZE=_PREPARE_BLOCK_SIZE,
        multibuffer=False,
    )
    return output
