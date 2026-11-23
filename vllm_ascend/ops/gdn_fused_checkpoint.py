# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Preserve FP32 checkpoints with the A5 fused GDN prefill interface."""

from types import SimpleNamespace

import torch

from vllm_ascend.device.device_op import DeviceOperator


def fused_gdn_checkpoint_prefill(q, k, v, g, beta, initial_state, checkpoint_output, plan, fused_fwd):
    """Run packed prefix/tail segments without repeating the model forward.

    Each fused call returns an FP32 final state. That state is both the exact
    checkpoint and the tail's initial state; BF16 per-chunk H is never used as
    a cache snapshot. Inputs and outputs have sequence-major layout, and state
    has the cache's value-first layout.
    """
    if initial_state.dtype != torch.float32 or checkpoint_output.dtype != torch.float32:
        raise ValueError("Fused GDN checkpoint segments require FP32 recurrent state")
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    final_state = initial_state.clone()

    def select(tensor, segment, axis):
        region = segment.row_range if axis == 0 else segment.token_range
        if region is not None:
            return tensor.narrow(axis, *region)
        indices = segment.rows if axis == 0 else segment.tokens
        return tensor.index_select(axis, indices)

    def store(tensor, segment, values, axis):
        region = segment.row_range if axis == 0 else segment.token_range
        if region is not None:
            tensor.narrow(axis, *region).copy_(values)
        else:
            indices = segment.rows if axis == 0 else segment.tokens
            tensor.index_copy_(axis, indices, values)

    def run(segment):
        metadata = SimpleNamespace(
            cu_seqlens_host=segment.cu_seqlens,
            chunk_indices_chunk64_host=segment.chunk_indices,
            keep_meta=None,
        )
        segment_output, segment_final = DeviceOperator.fla_gdn_prefill(
            q=select(q, segment, 1),
            k=select(k, segment, 1),
            v=select(v, segment, 1),
            g=select(g, segment, 1),
            beta=select(beta, segment, 1),
            initial_state=select(final_state, segment, 0),
            scale=k.shape[-1] ** -0.5,
            prebuilt_meta=metadata,
            fused_fwd=fused_fwd,
        )
        store(output, segment, segment_output, 1)
        store(final_state, segment, segment_final, 0)

    if plan.prefix.rows.numel():
        run(plan.prefix)
    if plan.tail.rows.numel():
        store(checkpoint_output, plan.tail, select(final_state, plan.tail, 0), 0)
        run(plan.tail)
    return output, final_state
