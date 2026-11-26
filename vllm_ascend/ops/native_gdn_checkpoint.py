# SPDX-License-Identifier: Apache-2.0
"""Export a GDN checkpoint by splitting the native recurrence inside a forward."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)

_FLA_CHUNK_SIZE = 64
_A5_SHARED_PREPARE_KEY_SHAPE = (16, 128)
_A5_SHARED_PREPARE_VALUE_SHAPE = (32, 128)


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
        for chunk in range((end - begin + _FLA_CHUNK_SIZE - 1) // _FLA_CHUNK_SIZE):
            indices.extend((row, chunk))
    return NativeFlaPartition(tuple(cu_lens), tuple(indices))


def _apply_recurrence(tensors, state, partition, scale, fused, fla_fwd, *, save_training_intermediates=True):
    if fla_fwd is not None:
        from vllm_ascend.device.device_op import DeviceOperator

        return DeviceOperator.fla_gdn_prefill(
            *tensors,
            state,
            scale,
            prebuilt_meta=partition,
            fused_fwd=fla_fwd,
            save_training_intermediates=save_training_intermediates,
            allow_chunk32=False,
        )
    return fused(*tensors, state, partition, scale)


@lru_cache(maxsize=1)
def _get_shared_fla_ops():
    try:
        from fla_npu.ops.ascendc import (  # type: ignore[import-not-found]
            npu_chunk_fwd_h,
            npu_chunk_fwd_o,
            npu_chunk_gated_delta_rule_fwd_prepare,
        )
    except ImportError:
        return None
    return npu_chunk_gated_delta_rule_fwd_prepare, npu_chunk_fwd_h, npu_chunk_fwd_o


def _run_shared_fla_prepare(tensors, initial_state, offset, scale, ops):
    prepare, recurrence, attention = ops
    q, k, v, g, beta = tensors
    length = q.shape[1]
    qkv = tuple(tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))

    def partition(size):
        meta = _partition_metadata((0, size), q.device, fla=True)
        return {"cu_seqlens": meta.cu_seqlens_host, "chunk_indices": meta.chunk_indices_chunk64_host}

    full = partition(length)
    # Preparation is local to each chunk. Both recurrences can use the same
    # prepared tensors when the checkpoint lies on a chunk boundary.
    q_hat, k_hat, _, _, _, cumsum, w, u, _ = prepare(
        *qkv,
        g.float().transpose(1, 2).contiguous(),
        beta.float().transpose(1, 2).contiguous(),
        chunk_size=_FLA_CHUNK_SIZE,
        use_qk_l2norm_in_kernel=True,
        use_exp2=True,
        output_a=False,
        **full,
    )

    def run_h(start, count, state):
        ks, ws, us, gs = (x.narrow(2, start, count).contiguous() for x in (k_hat, w, u, cumsum))
        return recurrence(
            ks,
            ws,
            us,
            g=gs,
            initial_state=state,
            output_final_state=True,
            chunk_size=_FLA_CHUNK_SIZE,
            use_exp2=True,
            state_v_first=True,
            **partition(count),
        )

    prefix_h, prefix_v, boundary = run_h(0, offset, initial_state.contiguous())
    tail_h, tail_v, final = run_h(offset, length - offset, boundary)
    output = attention(
        q_hat,
        k_hat,
        torch.cat((prefix_v, tail_v), dim=2),
        torch.cat((prefix_h, tail_h), dim=1),
        scale,
        g=cumsum,
        chunk_size=_FLA_CHUNK_SIZE,
        transpose_state_layout=True,
        use_exp2=True,
        output_layout="BSND",
        **full,
    )
    return output, boundary, final


def _native_checkpoint_plan(checkpoint, device, fla):
    plan = getattr(checkpoint, "_ascend_native_plan", None)
    if plan is None:
        arrays = plan_native_checkpoint(checkpoint.offsets, checkpoint.query_lens)
        plans = [async_tensor_h2d(values, dtype=torch.int64, device=device) for values in arrays[:4]]
        lengths = [_partition_metadata(values, device, fla) for values in arrays[4:]]
        plan = NativeCheckpointPlan(*plans, *lengths)
        checkpoint._ascend_native_plan = plan
    return plan


def _slice_inputs(tensors, tokens, head_first_qkv):
    q, k, v, g, beta = tensors
    if head_first_qkv is None:
        return tuple(tensor[:, tokens] for tensor in tensors)
    # The convolution produces all heads in one allocation. Pack the segment
    # once, then expose the same sequence-major Q/K/V views to the FLA backend.
    packed = head_first_qkv[:, tokens].contiguous()
    qkv = packed.split((q.shape[2], k.shape[2], v.shape[2]), dim=0)
    return (*(part.transpose(0, 1).unsqueeze(0) for part in qkv), g[:, tokens], beta[:, tokens])


def _run_contiguous_tail(layer, metadata, raw_qkv, tensors, initial_state, scale, fused, fla_fwd, head_first_qkv):
    checkpoint = metadata.checkpoint
    query_lens = checkpoint.query_lens
    offset = checkpoint.offsets[-1]
    split = sum(query_lens[:-1]) + offset
    lengths = getattr(checkpoint, "_ascend_native_contiguous_lengths", None)
    if lengths is None:
        device = tensors[0].device
        fla = fla_fwd is not None
        prefix_bounds = list(accumulate([*query_lens[:-1], offset], initial=0))
        lengths = (
            _partition_metadata(prefix_bounds, device, fla),
            _partition_metadata([0, query_lens[-1] - offset], device, fla),
        )
        checkpoint._ascend_native_contiguous_lengths = lengths
    save_intermediates = head_first_qkv is None
    prefix = _slice_inputs(tensors, slice(0, split), head_first_qkv)
    prefix_output, prefix_state = _apply_recurrence(
        prefix, initial_state, lengths[0], scale, fused, fla_fwd, save_training_intermediates=save_intermediates
    )
    _export_checkpoint(layer, metadata, raw_qkv, prefix_state)
    tail = _slice_inputs(tensors, slice(split, None), head_first_qkv)
    tail_output, tail_state = _apply_recurrence(
        tail, prefix_state[-1:], lengths[1], scale, fused, fla_fwd, save_training_intermediates=save_intermediates
    )
    if len(query_lens) == 1:
        return torch.cat((prefix_output, tail_output), dim=1), tail_state
    prefix_state[-1:].copy_(tail_state)
    return torch.cat((prefix_output, tail_output), dim=1), prefix_state


def _export_checkpoint(layer, metadata, raw_qkv, state):
    layer._checkpoint_exporter.export(
        metadata.checkpoint,
        raw_qkv=raw_qkv,
        conv_state=layer.kv_cache[0].transpose(-1, -2),
        recurrent_checkpoint=state,
        recurrent_state=layer.kv_cache[1],
        cu_seqlens=metadata.prefill_query_start_loc,
    )
    ack = getattr(metadata.checkpoint, "dedicated_ack", None)
    if ack is not None:
        ack.layers.add(layer.prefix)
    logger.info_once("Native GDN prefills have launched the checkpoint export kernel")


def run_gdn_native_checkpoint_prefill(
    layer, metadata, raw_qkv, q, k, v, g, beta, initial_state, *, fused, fla_fwd=None, head_first_qkv=None
):
    checkpoint = metadata.checkpoint
    assert checkpoint is not None
    tensors = (q, k, v, g, beta)
    scale = k.shape[-1] ** -0.5
    if (
        fla_fwd is not None
        and head_first_qkv is not None
        and len(checkpoint.query_lens) == 1
        and checkpoint.query_lens[0] == q.shape[1]
        and len(checkpoint.offsets) == 1
        and 0 < checkpoint.offsets[0] < q.shape[1]
        and checkpoint.offsets[0] % _FLA_CHUNK_SIZE == 0
        and q.shape[2:] == _A5_SHARED_PREPARE_KEY_SHAPE
        and v.shape[2:] == _A5_SHARED_PREPARE_VALUE_SHAPE
        and initial_state.shape == (1, *v.shape[2:], q.shape[-1])
        and initial_state.dtype == torch.float32
        and q.dtype == torch.bfloat16
    ):
        ops = _get_shared_fla_ops()
        if ops is not None:
            output, boundary, final = _run_shared_fla_prepare(tensors, initial_state, checkpoint.offsets[0], scale, ops)
            _export_checkpoint(layer, metadata, raw_qkv, boundary)
            logger.info_once("A5 stock FLA shared preparation exported a GDN checkpoint")
            return output, final

    if checkpoint.offsets[-1] and not any(checkpoint.offsets[:-1]):
        if len(checkpoint.offsets) > 1:
            logger.info_once("Native GDN checkpoints use contiguous batched tails")
        # Only the final row has a tail, so the packed token order is already contiguous.
        return _run_contiguous_tail(
            layer, metadata, raw_qkv, tensors, initial_state, scale, fused, fla_fwd, head_first_qkv
        )

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
