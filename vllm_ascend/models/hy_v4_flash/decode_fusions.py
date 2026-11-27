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
    if value.device.type != "npu" or not is_950():
        return norm(value)
    dtype = value.dtype
    # torch.rms_norm uses the accumulation dtype for its implicit epsilon.
    # BF16/FP16 inputs accumulate in FP32 in this PyTorch runtime.
    eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps
    gamma = getattr(norm, "_hy4_decode_fp32_weight", None)
    if gamma is None:
        gamma = norm.weight.detach().float()
        norm._hy4_decode_fp32_weight = gamma
    return torch_npu.npu_rms_norm(value, gamma, eps)[0].to(dtype)


def native_mla_context(query, selected, kv, sink, scale, *, scores=None, query_fp32=None):
    """Keep FP32 KV and sink normalization in the fused attention calculation.

    The CANN MLA FIA ABI accepts only low-precision KV/output and cannot consume
    a sink for this head dimension. Use the FP32 fusion instead of rounding KV
    or correcting an already-rounded context. Reader fusion remains enabled.
    """
    from .kv_compaction import CompactKV
    from .precision_attention import fused_fp32_mla_context

    invalid_mask = None
    if isinstance(kv, tuple):
        nope, rope, mask = kv
        values = torch.cat((nope[:, :, 0].float(), rope[:, :, 0].float()), dim=-1)
        invalid_mask = mask
        if isinstance(kv, CompactKV) and scores is not None:
            order = torch.argsort((selected < 0).to(torch.int32), dim=-1, stable=True)
            scores = scores.gather(-1, order[:, None].expand(-1, scores.shape[1], -1))
    else:
        values = kv.float()
    attention_query = query if query_fp32 is None else query_fp32
    if query.device.type == "npu":
        return fused_fp32_mla_context(
            attention_query,
            selected,
            values,
            sink,
            scale,
            invalid_mask=invalid_mask,
            scores=scores,
            output_dtype=query.dtype,
        )
    # CPU execution is for reference tests; no CANN/Triton launch is possible.
    if scores is None:
        scores = torch.bmm(attention_query.float(), values.transpose(1, 2))
    invalid = selected < 0 if invalid_mask is None else invalid_mask.reshape_as(selected)
    logits = (scores * scale).masked_fill(invalid[:, None], -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(query.shape[0], -1, 1)), -1)
    probabilities = logits.softmax(-1)[..., :-1]
    return torch.bmm(probabilities, values[..., :512]).to(query.dtype)


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
