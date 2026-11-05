# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# Adapted from vLLM #64916.
"""Fused DFlash2 candidate-pair scoring for Ascend."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _dflash2_score_edges_kernel(
    predecessor_ptr,
    successor_ptr,
    candidate_ptr,
    unary_ptr,
    hidden_ptr,
    anchor_ptr,
    output_ptr,
    anchor_stride,
    vocab_size,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    position = tl.program_id(0)
    req_idx = position // NUM_STEPS
    step = position % NUM_STEPS
    k = tl.arange(0, BLOCK_K)
    r = tl.arange(0, BLOCK_R)
    k_mask = k < TOP_K
    r_mask = r < RANK
    candidate_ids = tl.load(candidate_ptr + position * TOP_K + k, mask=k_mask, other=0).to(tl.int64)
    if step == 0:
        # The predecessor for the first step is the anchor token.
        anchor = tl.load(anchor_ptr + req_idx * anchor_stride).to(tl.int64)
        anchor = tl.minimum(tl.maximum(0, anchor), vocab_size - 1)
        predecessor_ids = tl.zeros_like(candidate_ids) + anchor
    else:
        predecessor_ids = tl.load(candidate_ptr + (position - 1) * TOP_K + k, mask=k_mask, other=0).to(tl.int64)

    unary = tl.load(unary_ptr + position * TOP_K + k, mask=k_mask, other=0.0)
    hidden = tl.load(hidden_ptr + position * RANK + r, mask=r_mask, other=0.0)
    mask = k_mask[:, None] & r_mask[None, :]
    predecessors = tl.load(
        predecessor_ptr + predecessor_ids[:, None] * RANK + r[None, :],
        mask=mask,
        other=0.0,
    )
    successors = tl.load(
        successor_ptr + candidate_ids[:, None] * RANK + r[None, :],
        mask=mask,
        other=0.0,
    )
    weighted = predecessors.to(tl.float32) * hidden[None, :].to(tl.float32)
    edges = tl.dot(weighted.to(successors.dtype), tl.trans(successors))
    tl.store(
        output_ptr + (position * TOP_K + k[:, None]) * TOP_K + k[None, :],
        unary[None, :].to(tl.float32) + edges.to(tl.float32),
        mask=k_mask[:, None] & k_mask[None, :],
    )


def _score_edges_output(candidate_ids: torch.Tensor) -> torch.Tensor:
    batch, steps, top_k = candidate_ids.shape
    return candidate_ids.new_empty((batch, steps, top_k, top_k), dtype=torch.float32)


def dflash2_score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    output = _score_edges_output(candidate_ids)
    batch, steps, top_k = candidate_ids.shape
    if batch == 0 or steps == 0 or top_k == 0:
        return output
    vocab_size, rank = predecessor_table.shape
    _dflash2_score_edges_kernel[(batch * steps,)](
        predecessor_table,
        successor_table,
        candidate_ids.contiguous(),
        unary_logits.contiguous(),
        hidden.contiguous(),
        anchor_token_ids,
        output,
        anchor_token_ids.stride(0),
        vocab_size,
        NUM_STEPS=steps,
        TOP_K=top_k,
        RANK=rank,
        BLOCK_K=max(16, triton.next_power_of_2(top_k)),
        BLOCK_R=max(16, triton.next_power_of_2(rank)),
    )
    return output


def dflash2_score_edges_fake(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    return _score_edges_output(candidate_ids)


direct_register_custom_op(
    op_name="ascend_dflash2_score_edges",
    op_func=dflash2_score_edges,
    fake_impl=dflash2_score_edges_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
