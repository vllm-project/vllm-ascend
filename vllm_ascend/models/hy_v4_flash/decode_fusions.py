# SPDX-License-Identifier: Apache-2.0
"""Native decode operators; prefill keeps its existing implementation."""

import torch
import torch_npu

from vllm_ascend.device.device_config import is_950

from .pd_role import uses_prefill_ops


def uses_decode_fusions(config):
    """Decode fusion is part of the 950 runtime, independent of configuration."""
    return not uses_prefill_ops(config) and is_950()


def is_decode_batch(batch):
    return batch is not None and bool(batch.prefill_reqs) and not any(batch.prefill_reqs)


def native_rms_norm(value, norm):
    """Retain nn.RMSNorm's FP32 arithmetic and final output cast."""
    dtype = value.dtype
    # torch.rms_norm uses the accumulation dtype for its implicit epsilon.
    # BF16/FP16 inputs accumulate in FP32 in this PyTorch runtime.
    eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps
    gamma = getattr(norm, "_hy4_decode_fp32_weight", None)
    if gamma is None:
        gamma = norm.weight.detach().float()
        norm._hy4_decode_fp32_weight = gamma
    return torch_npu.npu_rms_norm(value, gamma, eps)[0].to(dtype)


def native_mla_context(query, selected, kv, sink, scale):
    """FIA owns QK/masking/softmax/PV; retain the FP32 sink using its LSE."""
    if isinstance(kv, tuple):
        # The C8 reader has already materialized the exact FIA input layouts.
        nope, rope, mask = kv
    else:
        keys = kv.to(query.dtype).unsqueeze(2)
        nope = keys[..., :512].contiguous()
        rope = keys[..., 512:].contiguous()
        mask = (selected < 0)[:, None, None, :].contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score_v2(
        query[..., :512].unsqueeze(1).contiguous(),
        nope,
        nope,
        query_rope=query[..., 512:].unsqueeze(1).contiguous(),
        key_rope=rope,
        num_query_heads=query.shape[1],
        num_key_value_heads=1,
        input_layout="BSND",
        atten_mask=mask,
        softmax_scale=scale,
        return_softmax_lse=True,
    )
    from .attention_fusions import fused_sink_context

    return fused_sink_context(output.squeeze(1), lse, sink, selected)


def native_indexpool_rope(nope, even, odd, cos, sin, dtype):
    """Map HY4's even/odd input to CANN half layout and retain its output order."""
    packed = torch.cat((even, odd), -1)
    rows = packed.shape[0]
    heads = packed.shape[1] if packed.ndim == 3 else 1
    width = packed.shape[-1]
    rotated = torch_npu.npu_rotary_mul(
        packed.reshape(rows, 1, heads, width),
        torch.cat((cos, cos), -1).reshape(rows, 1, 1, width),
        torch.cat((sin, sin), -1).reshape(rows, 1, 1, width),
        rotary_mode="half",
    ).reshape_as(packed)
    return torch.cat((nope, rotated.to(dtype)), -1)


def prepare_indexpool_selection(visible, query_rows, pool_rows):
    """Prepare layer-independent selection metadata once per request/forward."""
    import cann_ops_transformer  # noqa: F401 - registers the CANN 9.3 operators

    ratio, budget = 4, 511
    cutoff = visible // ratio
    used = cutoff[-1:].to(torch.int32).clamp_min(budget + 1)
    residual = (visible[-1:] % ratio).to(torch.int32)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        16,
        1,
        128,
        budget,
        seqused_k=used,
        cmp_residual_k=residual,
        batch_size=1,
        max_seqlen_q=query_rows,
        max_seqlen_k=pool_rows,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    offsets = torch.arange(ratio, device=visible.device)
    return used, residual, metadata, cutoff, offsets


def native_indexpool_select(query, weights, pool, visible, selection_state=None, *, fuse_postprocess=False):
    """Score layer-specific keys using the shared selection geometry."""
    ratio, budget = 4, 511
    if selection_state is None:
        selection_state = prepare_indexpool_selection(visible, query.shape[0], pool.shape[0])
    used, residual, metadata, cutoff, offsets = selection_state
    groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query.to(torch.bfloat16).unsqueeze(0),
        pool.to(torch.bfloat16).view(1, -1, 1, 128),
        weights.unsqueeze(0),
        budget,
        seqused_k=used,
        cmp_residual_k=residual,
        metadata=metadata,
        max_seqlen_q=query.shape[0],
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=ratio,
        return_value=0,
    )
    groups = groups.reshape(query.shape[0], budget).long().sort(-1).values
    if fuse_postprocess:
        from .decode_data_fusions import expand_decode_candidates

        return expand_decode_candidates(groups, visible)
    history = (
        (groups[..., None] * ratio + offsets)
        .masked_fill(((groups < 0) | (groups >= cutoff[:, None]))[..., None], -1)
        .flatten(1)
    )
    tail = cutoff[:, None] * ratio + offsets
    return torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), -1)
