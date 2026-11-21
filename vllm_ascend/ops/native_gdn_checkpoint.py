# SPDX-License-Identifier: Apache-2.0
"""Export a GDN checkpoint by splitting the native recurrence inside a forward."""

from dataclasses import dataclass
from itertools import accumulate

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)


@dataclass(frozen=True)
class NativeFlaPartition:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    keep_meta: None = None


@dataclass(frozen=True)
class NativeCheckpointPlan:
    prefix_tokens: torch.Tensor
    tail_tokens: torch.Tensor
    tail_rows: torch.Tensor
    restore_order: torch.Tensor
    prefix_lengths: torch.Tensor | NativeFlaPartition
    tail_lengths: torch.Tensor | NativeFlaPartition


def plan_native_checkpoint(offsets, lengths):
    prefix_tokens, tail_tokens, tail_rows = [], [], []
    prefix_lens, tail_lens = [], []
    cursor = 0
    for row, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        assert 0 <= offset < length
        prefix_len = offset or length
        prefix_tokens.extend(range(cursor, cursor + prefix_len))
        prefix_lens.append(prefix_len)
        if offset:
            tail_tokens.extend(range(cursor + prefix_len, cursor + length))
            tail_lens.append(length - prefix_len)
            tail_rows.append(row)
        cursor += length

    assert tail_rows
    restore = [0] * cursor
    for packed, original in enumerate(prefix_tokens + tail_tokens):
        restore[original] = packed
    return (
        prefix_tokens,
        tail_tokens,
        tail_rows,
        restore,
        list(accumulate(prefix_lens, initial=0)),
        list(accumulate(tail_lens, initial=0)),
    )


def _partition_metadata(cu_lens, device, fla):
    if not fla:
        return async_tensor_h2d(cu_lens, dtype=torch.int32, device=device)

    indices = []
    for row, (begin, end) in enumerate(zip(cu_lens, cu_lens[1:])):
        for chunk in range((end - begin + 63) // 64):
            indices.extend((row, chunk))
    return NativeFlaPartition(tuple(cu_lens), tuple(indices))


def _apply_recurrence(tensors, state, partition, scale, fused, fla_fwd):
    if fla_fwd is not None:
        from vllm_ascend.device.device_op import DeviceOperator

        return DeviceOperator.fla_gdn_prefill(*tensors, state, scale, prebuilt_meta=partition, fused_fwd=fla_fwd)
    return fused(*tensors, state, partition, scale)


def _native_checkpoint_plan(checkpoint, device, fla):
    plan = getattr(checkpoint, "_ascend_native_plan", None)
    if plan is None:
        arrays = plan_native_checkpoint(checkpoint.offsets, checkpoint.query_lens)
        plans = [async_tensor_h2d(values, dtype=torch.int64, device=device) for values in arrays[:4]]
        lengths = [_partition_metadata(values, device, fla) for values in arrays[4:]]
        plan = NativeCheckpointPlan(*plans, *lengths)
        checkpoint._ascend_native_plan = plan
    return plan


def _run_single_row(layer, metadata, raw_qkv, tensors, initial_state, scale, fused, fla_fwd):
    checkpoint = metadata.checkpoint
    offset = checkpoint.offsets[0]
    lengths = getattr(checkpoint, "_ascend_native_single_lengths", None)
    if lengths is None:
        device = tensors[0].device
        fla = fla_fwd is not None
        lengths = (
            _partition_metadata([0, offset], device, fla),
            _partition_metadata([0, checkpoint.query_lens[0] - offset], device, fla),
        )
        checkpoint._ascend_native_single_lengths = lengths
    prefix = tuple(tensor[:, :offset] for tensor in tensors)
    prefix_output, prefix_state = _apply_recurrence(prefix, initial_state, lengths[0], scale, fused, fla_fwd)
    _export_checkpoint(layer, metadata, raw_qkv, prefix_state)
    tail = tuple(tensor[:, offset:] for tensor in tensors)
    tail_output, tail_state = _apply_recurrence(tail, prefix_state, lengths[1], scale, fused, fla_fwd)
    return torch.cat((prefix_output, tail_output), dim=1), tail_state


def _export_checkpoint(layer, metadata, raw_qkv, state):
    layer._checkpoint_exporter.export(
        metadata.checkpoint,
        raw_qkv=raw_qkv,
        conv_state=layer.kv_cache[0].transpose(-1, -2),
        recurrent_checkpoint=state,
        recurrent_state=layer.kv_cache[1],
        cu_seqlens=metadata.prefill_query_start_loc,
    )
    logger.info_once("Native GDN prefills have launched the checkpoint export kernel")


def run_gdn_native_checkpoint_prefill(
    layer, metadata, raw_qkv, q, k, v, g, beta, initial_state, *, fused, fla_fwd=None
):
    checkpoint = metadata.checkpoint
    assert checkpoint is not None
    tensors = (q, k, v, g, beta)
    scale = k.shape[-1] ** -0.5
    if len(checkpoint.offsets) == 1:
        return _run_single_row(layer, metadata, raw_qkv, tensors, initial_state, scale, fused, fla_fwd)

    plan = _native_checkpoint_plan(checkpoint, q.device, fla_fwd is not None)

    prefix_inputs = tuple(tensor.index_select(1, plan.prefix_tokens) for tensor in tensors)
    prefix_output, prefix_state = _apply_recurrence(
        prefix_inputs, initial_state, plan.prefix_lengths, scale, fused, fla_fwd
    )
    _export_checkpoint(layer, metadata, raw_qkv, prefix_state)

    tail_inputs = tuple(tensor.index_select(1, plan.tail_tokens) for tensor in tensors)
    tail_state_in = prefix_state.index_select(0, plan.tail_rows)
    tail_output, tail_state = _apply_recurrence(tail_inputs, tail_state_in, plan.tail_lengths, scale, fused, fla_fwd)
    prefix_state.index_copy_(0, plan.tail_rows, tail_state)
    output = torch.cat((prefix_output, tail_output), dim=1).index_select(1, plan.restore_order)
    return output, prefix_state
