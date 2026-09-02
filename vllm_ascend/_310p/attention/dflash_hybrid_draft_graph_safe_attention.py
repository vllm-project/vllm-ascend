# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Graph-safe Draft attention for 310P DFlash FULL_AND_PIECEWISE.

The public 310P SplitFuse operator consumes host qLens.  A FULL graph freezes
those host values at capture time, while Phase 4 replays one Draft model graph
for different speculative substeps and active request counts.  This private
entry keeps the graph's physical shapes fixed but derives the logical paged
attention rows entirely from persistent device tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu


@dataclass(frozen=True)
class DFlashHybridDraftAttentionInputs310:
    """Persistent device metadata owned by one Draft FULL descriptor."""

    valid_num_reqs: torch.Tensor
    valid_num_tokens: torch.Tensor
    query_lens: torch.Tensor
    query_starts: torch.Tensor
    query_ends: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    token_indices: torch.Tensor

    @property
    def capacity_reqs(self) -> int:
        return int(self.query_lens.shape[0])

    @property
    def capacity_tokens(self) -> int:
        return int(self.token_indices.shape[0])


@dataclass(frozen=True)
class DFlashHybridDraftPagedView310:
    request_ids: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    valid_token_mask: torch.Tensor


def create_dflash_hybrid_draft_attention_inputs_310(
    *,
    capacity_reqs: int,
    capacity_tokens: int,
    max_blocks: int,
    device: torch.device,
) -> DFlashHybridDraftAttentionInputs310:
    if capacity_reqs <= 0 or capacity_tokens <= 0 or max_blocks <= 0:
        raise ValueError("Draft FULL descriptor capacities must be positive")
    return DFlashHybridDraftAttentionInputs310(
        valid_num_reqs=torch.empty(1, dtype=torch.int32, device=device),
        valid_num_tokens=torch.empty(1, dtype=torch.int32, device=device),
        query_lens=torch.empty(capacity_reqs, dtype=torch.int32, device=device),
        query_starts=torch.empty(capacity_reqs, dtype=torch.int32, device=device),
        query_ends=torch.empty(capacity_reqs, dtype=torch.int32, device=device),
        seq_lens=torch.empty(capacity_reqs, dtype=torch.int32, device=device),
        block_table=torch.empty(
            capacity_reqs,
            max_blocks,
            dtype=torch.int32,
            device=device,
        ),
        token_indices=torch.arange(
            capacity_tokens,
            dtype=torch.int32,
            device=device,
        ),
    )


def _validate_inputs_310(inputs: DFlashHybridDraftAttentionInputs310) -> None:
    tensors = (
        inputs.valid_num_reqs,
        inputs.valid_num_tokens,
        inputs.query_lens,
        inputs.query_starts,
        inputs.query_ends,
        inputs.seq_lens,
        inputs.block_table,
        inputs.token_indices,
    )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("Draft FULL device metadata must use int32")
    device = inputs.query_lens.device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("Draft FULL device metadata must share one device")
    if inputs.valid_num_reqs.shape != (1,) or inputs.valid_num_tokens.shape != (1,):
        raise ValueError("Draft FULL valid counts must have shape [1]")
    request_shape = (inputs.capacity_reqs,)
    if any(
        tensor.shape != request_shape
        for tensor in (
            inputs.query_lens,
            inputs.query_starts,
            inputs.query_ends,
            inputs.seq_lens,
        )
    ):
        raise ValueError("Draft FULL request metadata shape mismatch")
    if inputs.block_table.ndim != 2 or inputs.block_table.shape[0] != inputs.capacity_reqs:
        raise ValueError("Draft FULL block table capacity mismatch")


def update_dflash_hybrid_draft_attention_inputs_310(
    inputs: DFlashHybridDraftAttentionInputs310,
    *,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    valid_num_reqs: int,
    valid_num_tokens: int,
) -> None:
    """Prepare replay metadata outside the graph without changing addresses."""
    _validate_inputs_310(inputs)
    if (
        valid_num_reqs <= 0
        or valid_num_tokens <= 0
        or valid_num_reqs > inputs.capacity_reqs
        or valid_num_tokens > inputs.capacity_tokens
    ):
        raise ValueError(
            "310P DFlash Hybrid Draft runtime count exceeds FULL descriptor capacity: "
            f"requests={valid_num_reqs}/{inputs.capacity_reqs}, "
            f"tokens={valid_num_tokens}/{inputs.capacity_tokens}"
        )
    if query_lens.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("Draft FULL query_lens and seq_lens must use int32")
    if block_table.dtype != torch.int32:
        raise TypeError("Draft FULL block_table must use int32")
    if any(
        tensor.device != inputs.query_lens.device
        for tensor in (query_lens, seq_lens, block_table)
    ):
        raise ValueError("Draft FULL runtime metadata device mismatch")
    if query_lens.ndim != 1 or query_lens.shape[0] < valid_num_reqs:
        raise ValueError("Draft FULL query_lens does not cover valid requests")
    if seq_lens.ndim != 1 or seq_lens.shape[0] < valid_num_reqs:
        raise ValueError("Draft FULL seq_lens does not cover valid requests")
    if (
        block_table.ndim != 2
        or block_table.shape[0] < valid_num_reqs
        or block_table.shape[1] != inputs.block_table.shape[1]
    ):
        raise ValueError("Draft FULL block_table does not cover valid requests")

    inputs.valid_num_reqs.fill_(valid_num_reqs)
    inputs.valid_num_tokens.fill_(valid_num_tokens)
    inputs.query_lens.zero_()
    inputs.query_lens[:valid_num_reqs].copy_(query_lens[:valid_num_reqs])
    inputs.query_starts.zero_()
    inputs.query_ends.zero_()
    torch.cumsum(
        inputs.query_lens[:valid_num_reqs],
        dim=0,
        out=inputs.query_ends[:valid_num_reqs],
    )
    if valid_num_reqs > 1:
        inputs.query_starts[1:valid_num_reqs].copy_(
            inputs.query_ends[: valid_num_reqs - 1]
        )
    inputs.seq_lens.zero_()
    inputs.seq_lens[:valid_num_reqs].copy_(seq_lens[:valid_num_reqs])
    inputs.block_table.zero_()
    inputs.block_table[:valid_num_reqs].copy_(block_table[:valid_num_reqs])


def copy_dflash_hybrid_draft_attention_inputs_310(
    destination: DFlashHybridDraftAttentionInputs310,
    source: DFlashHybridDraftAttentionInputs310,
) -> None:
    """D2D-refresh the one metadata set whose addresses were captured."""
    _validate_inputs_310(destination)
    _validate_inputs_310(source)
    destination_tensors = (
        destination.valid_num_reqs,
        destination.valid_num_tokens,
        destination.query_lens,
        destination.query_starts,
        destination.query_ends,
        destination.seq_lens,
        destination.block_table,
    )
    source_tensors = (
        source.valid_num_reqs,
        source.valid_num_tokens,
        source.query_lens,
        source.query_starts,
        source.query_ends,
        source.seq_lens,
        source.block_table,
    )
    for dst, src in zip(destination_tensors, source_tensors):
        if dst.shape != src.shape or dst.device != src.device:
            raise ValueError("Draft FULL captured/runtime metadata descriptors differ")
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src, non_blocking=True)


def build_dflash_hybrid_draft_paged_view_310(
    inputs: DFlashHybridDraftAttentionInputs310,
) -> DFlashHybridDraftPagedView310:
    """Build fixed-shape per-token metadata using only device tensor values."""
    _validate_inputs_310(inputs)
    token_indices = inputs.token_indices
    membership = torch.logical_and(
        token_indices[:, None] >= inputs.query_starts[None, :],
        token_indices[:, None] < inputs.query_ends[None, :],
    )
    membership_i32 = membership.to(torch.int32)
    request_ids = torch.argmax(membership_i32, dim=1)
    has_request = torch.sum(membership_i32, dim=1) > 0
    valid_token_mask = torch.logical_and(
        token_indices < inputs.valid_num_tokens[0],
        torch.logical_and(
            request_ids < inputs.valid_num_reqs[0],
            has_request,
        ),
    )
    safe_request_ids = torch.where(
        valid_token_mask,
        request_ids,
        torch.zeros_like(request_ids),
    )
    context_lens = torch.index_select(inputs.seq_lens, 0, safe_request_ids)
    context_lens = torch.where(
        valid_token_mask,
        context_lens,
        torch.ones_like(context_lens),
    )
    expanded_block_table = torch.index_select(
        inputs.block_table,
        0,
        safe_request_ids,
    )
    return DFlashHybridDraftPagedView310(
        request_ids=safe_request_ids,
        context_lens=context_lens,
        block_table=expanded_block_table,
        valid_token_mask=valid_token_mask,
    )


def dflash_hybrid_draft_graph_safe_attention_310(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    inputs: DFlashHybridDraftAttentionInputs310,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the exact Hybrid Draft FULL route without host SplitFuse tiling."""
    if query.shape[0] != inputs.capacity_tokens:
        raise ValueError(
            "Draft FULL query capacity does not match its descriptor: "
            f"query={query.shape[0]}, descriptor={inputs.capacity_tokens}"
        )
    if output.shape != query.shape:
        raise ValueError("Draft FULL output must retain the query physical shape")

    view = build_dflash_hybrid_draft_paged_view_310(inputs)
    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=view.block_table,
        context_lens=view.context_lens,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        out=output,
    )
    output.mul_(view.valid_token_mask[:, None, None].to(output.dtype))
    return output
