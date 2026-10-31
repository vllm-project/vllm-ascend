# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt DSA arguments to the installed MixedQuantSparseFlashMla operators.

Two paths share the MixedQuantSparseFlashMla (cann_ops_transformer) operator:

* TurboQuant (quant_mode=3): ``mixed_quant_sparse_flash_mla_turboquant`` /
  ``mixed_quant_sparse_flash_mla_metadata_turboquant`` (see
  vllm_ascend/quantization/methods/kv_cache/turboquant.py).
* DSV4 FP8 (quant_mode=1): ``mixed_quant_sparse_flash_mla_fp8`` /
  ``mixed_quant_sparse_flash_mla_metadata_fp8``, adapted from
  the former ``npu_kv_quant_sparse_attn_sharedkv`` path.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _add_compressed_kv_lengths,
    _drop_paged_kv_cu_seqlens,
    _ensure_sinks,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import COMPRESS_RATIO


@lru_cache
def _get_mixed_quant_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.mixed_quant_sparse_flash_mla, namespace.mixed_quant_sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek V4 TurboQuant requires MixedQuantSparseFlashMla from a matching ops-transformer package."
        ) from exc


def _adapt(kwargs):
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    # This adapter is selected only for DeepSeek V4 C4 TurboQuant pages.
    # Keep the operator's compressed-length arithmetic in that mode even when
    # a lightweight caller omits the generic DSA ``cmp_ratio`` argument.
    kwargs.update(quant_mode=3, rope_head_dim=64, cmp_ratio=COMPRESS_RATIO, layout_kv="PA_BBND")
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _drop_paged_kv_cu_seqlens(kwargs)


def mixed_quant_sparse_flash_mla_metadata_turboquant(**kwargs):
    _adapt(kwargs)
    # Metadata requires explicit compressed lengths; fused TQ attention derives them.
    _add_compressed_kv_lengths(kwargs)
    _, metadata_op = _get_mixed_quant_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def mixed_quant_sparse_flash_mla_turboquant(q, **kwargs):
    _adapt(kwargs)
    _ensure_sinks(kwargs)
    attention_op, _ = _get_mixed_quant_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)


# ---------------------------------------------------------------------------
# DSV4 FP8 quantMode=1 adapter
# ---------------------------------------------------------------------------


def _has_cmp(kw):
    return bool(kw.get("has_cmp_kv", int(kw.get("cmp_ratio", 1)) > 1))


def _split_seqused(seqused, ratio, has_cmp):
    if not has_cmp:
        return None, None
    return seqused // ratio, seqused % ratio


def _fullcov_sparse_indices(si, max_used):
    """True when the sparse indices cover the whole KV range.

    ``max_used`` is the per-sequence KV length limit as a host int
    (``max(seqused_kv)``). Callers already hold it as ``max_seqlen_kv``
    derived from the CPU sequence lengths, so no NPU-to-CPU tensor
    migration is needed for the coverage decision.
    """
    if si is None:
        return False
    valid = si[si >= 0]
    if valid.numel() == 0:
        return False
    return valid.numel() >= max_used and int(valid.max().item()) >= max_used - 1


# ---------------------------------------------------------------------------
# metadata op
# ---------------------------------------------------------------------------


def mq_meta_adapter(kw):
    # quantMode=1 (FP8): keep the operator's per-mode layout; the generic DSA
    # ``layout_kv`` argument is rewritten to the paged PA_BBND contract.
    kw.pop("device", None)
    kw.pop("kv_quant_mode", None)
    kw.pop("tile_size", None)
    kw.update(quant_mode=1, layout_kv="PA_BBND")
    # ``has_ori_kv``/``has_cmp_kv`` are host booleans, not tensors; the op
    # rejects tensors for them.
    kw["has_ori_kv"] = bool(kw.get("has_ori_kv", True))
    kw["has_cmp_kv"] = _has_cmp(kw)
    ratio = max(int(kw.get("cmp_ratio", 1)), 1)
    cmp_len, residual = _split_seqused(kw["seqused_kv"], ratio, kw["has_cmp_kv"])
    # max(cmp_len) == max(seqused) // ratio for floor division, and
    # max_seqlen_kv is the host int max(seqused_kv) supplied by the caller
    # (derived from the CPU sequence lengths), so no NPU-to-CPU sync is
    # needed here. Graph-capture warmup may omit the key; fall back to the
    # old device-free 0, keeping the previous .item() path's behavior.
    max_cmp = max(int(kw.get("max_seqlen_kv", 0)) // ratio, 1) if kw["has_cmp_kv"] else 0
    kw.update(
        num_heads_kv=int(kw.get("num_heads_kv", 1)),
        seqused_ori_kv=kw["seqused_kv"],
        seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,
        max_seqlen_ori_kv=int(kw["max_seqlen_kv"]),
        max_seqlen_cmp_kv=max_cmp,
        ori_topk=0,
        cmp_topk=int(kw.get("cmp_topk", 0)),
        rope_head_dim=int(kw.get("rope_head_dim", 64)),
        cmp_ratio=ratio,
        ori_mask_mode=int(kw.get("ori_mask_mode", 4)),
        cmp_mask_mode=3 if kw["has_cmp_kv"] else 0,
        ori_win_left=int(kw.get("ori_win_left", 0)),
        ori_win_right=int(kw.get("ori_win_right", 0)),
        layout_q=kw.get("layout_q", "TND"),
    )
    # ``seqused_kv`` is the caller-facing DSA name; the op reads the split
    # ``*_ori_kv``/``*_cmp_kv`` names instead.
    kw.pop("seqused_kv", None)
    kw.pop("max_seqlen_kv", None)
    # Explicit per-branch topk lengths are unused in this mode.
    kw.update(ori_topk_length=None, cmp_topk_length=None)
    # Keep the op-facing optionals explicit (None), matching the operator's
    # schema defaults and the prior full-dict contract.
    kw.update(seqused_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None)
    # cu_seqlens_q stays for the fused attention kernel.
    kw["cu_seqlens_q"] = kw.get("cu_seqlens_q")


def mixed_quant_sparse_flash_mla_metadata_fp8(**kw):
    mq_meta_adapter(kw)
    _, meta_op = _get_mixed_quant_sparse_flash_mla_ops()
    return meta_op(**kw)


# ---------------------------------------------------------------------------
# attention op
# ---------------------------------------------------------------------------


def mq_attn_adapter(q, kw):
    kw.pop("device", None)
    kw.pop("kv_quant_mode", None)
    kw.pop("tile_size", None)
    kw.update(quant_mode=1, layout_kv="PA_BBND")
    ratio = max(int(kw.get("cmp_ratio", 1)), 1)
    has_cmp = ratio > 1 and kw.get("cmp_kv") is not None
    cmp_len, residual = _split_seqused(kw["seqused_kv"], ratio, has_cmp)
    if not has_cmp:
        kw.pop("cmp_block_table", None)
        kw.pop("cmp_kv", None)
        kw.pop("cmp_sparse_indices", None)
        kw.pop("cmp_mask_mode", None)
    win_r = int(kw.get("ori_win_right", 0))
    osi = kw.get("ori_sparse_indices")
    if osi is not None:
        # Full coverage: the new op derives attention from the window instead.
        # The coverage check needs the batch total on the host, so it only
        # runs outside graph capture (vision prefill, executed eagerly).
        # Inside CUDA-graph capture the DSpark draft window from kwargs is
        # equivalent: the sliding window clips at the sequence start, so a
        # short sequence gets full coverage either way.
        # max_seqlen_kv is the host int max(seqused_kv); the window right
        # edge at max_seqlen_kv - 1 covers every sequence's KV (per-sequence
        # KV is independently addressed), so no NPU-to-CPU sync is needed
        # here either. Graph-capture warmup may not supply max_seqlen_kv, so
        # keep the lookup inside the non-capturing branch.
        if not torch.npu.is_current_stream_capturing():
            max_kv = kw.get("max_seqlen_kv")
            if max_kv is not None and _fullcov_sparse_indices(osi, int(max_kv)):
                win_r = int(q.shape[0]) - 1
        # Original indices are never passed: the new op derives attention
        # from the window (ori_mask_mode) instead.
        kw.pop("ori_sparse_indices", None)
    kw.update(
        seqused_ori_kv=kw["seqused_kv"],
        seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,
        ori_topk_length=None,
        cmp_topk_length=None,
        quant_mode=1,
        rope_head_dim=int(kw.get("rope_head_dim", 64)),
        cmp_ratio=ratio,
        ori_mask_mode=int(kw.get("ori_mask_mode", 4)),
        cmp_mask_mode=3 if has_cmp else 0,
        ori_win_left=int(kw.get("ori_win_left", 0)),
        ori_win_right=win_r,
        layout_q=kw.get("layout_q", "TND"),
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    # ``seqused_kv`` is the caller-facing DSA name; the op reads the split
    # ``*_ori_kv``/``*_cmp_kv`` names instead.
    kw.pop("seqused_kv", None)
    kw.pop("max_seqlen_kv", None)
    # Keep the op-facing optionals explicit (None), matching the operator's
    # schema defaults and the prior full-dict contract.
    kw.update(
        seqused_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        ori_sparse_indices=None,
        metadata=kw.get("metadata"),
    )
    # cu_seqlens_q stays for the fused attention kernel.
    kw["cu_seqlens_q"] = kw.get("cu_seqlens_q")


def mixed_quant_sparse_flash_mla_fp8(q, **kw):
    mq_attn_adapter(q, kw)
    attention_op, _ = _get_mixed_quant_sparse_flash_mla_ops()
    out, lse = attention_op(q, **kw)
    return out, lse
