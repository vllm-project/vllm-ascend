# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.sparse_flash_mla, namespace.sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires SparseFlashMla from a matching CANN 9.2 toolkit."
        ) from exc


def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    if cmp_ratio <= 1 or seqused_ori_kv is None:
        return
    kwargs.setdefault("seqused_cmp_kv", seqused_ori_kv // cmp_ratio)
    kwargs.setdefault("cmp_residual_kv", seqused_ori_kv % cmp_ratio)
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def _drop_paged_kv_cu_seqlens(kwargs: dict[str, Any]) -> None:
    """Drop KV cu_seqlens; SparseFlashMla only accepts them when layout_kv is TND.

    This adapter always uses PA_BBND paged cache. Passing cu_seqlens_ori_kv or
    cu_seqlens_cmp_kv raises EZ0037 from aclnnSparseFlashMla.
    """
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)


def sparse_flash_mla_metadata(**kwargs):
    """Adapt existing DSA metadata kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    # This adapter is only selected for the BF16 paged-KV path. SparseFlashMla
    # accepts PA_BBND for this cache; PA_ND belongs to the FP8 quantized op.
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def _ensure_sinks(kwargs: dict[str, Any]) -> None:
    """Reject calls that omit the required per-head sinks tensor.

    SparseFlashMla needs a per-head float32 sinks tensor and the caller must
    own it. A tensor allocated here would only be referenced by these kwargs,
    so ACL Graph capture bakes in an address that the allocator reuses as soon
    as the call returns; replaying that graph then reads freed memory and traps
    inside the kernel (507011). Device errors are reported asynchronously, so
    the failure surfaces at an unrelated synchronize() rather than at the
    operator that caused it, which makes it very hard to attribute.
    """
    if kwargs.get("sinks") is None:
        raise ValueError(
            "SparseFlashMla requires a caller-owned per-head sinks tensor; "
            "allocating one here would not survive ACL Graph capture."
        )


def _fullcov_sparse_indices(si: torch.Tensor, max_used: int) -> bool:
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


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt existing DSA attention kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs["layout_kv"] = "PA_BBND"
    # Window assignment mirrors the FP8 mixed-quant adapter of #18808: the
    # caller's values pass through unchanged; only a full-coverage index set
    # widens the right edge (the coverage decision needs a host sync, which
    # only eager mode allows — the BF16 DSA path runs enforce_eager). Vision
    # indices are still passed to this op, so their span semantics never rely
    # on the widened window.
    win_r = int(kwargs.get("ori_win_right", 0))
    # DSpark draft (decode, borrow mode) passes physical indices that this op
    # cannot execute on A5 (kernel trap 507015). The draft metadata is already
    # built with non-causal tiling and the caller supplies the DSpark window,
    # so the indices are redundant: drop them and derive attention from the
    # window, exactly like the FP8 mixed-quant adapter. Vision prefill also
    # passes indices, but always as a prefill call (cu_seqlens_ori_kv set by
    # the prefill extra kwargs), and its span-level bidirectionality genuinely
    # needs them, so those pass through unchanged.
    if kwargs.get("ori_sparse_indices") is not None and kwargs.get("cu_seqlens_ori_kv") is None:
        draft_indices = kwargs.pop("ori_sparse_indices")
        max_kv = kwargs.get("max_seqlen_kv")
        if max_kv is not None and _fullcov_sparse_indices(draft_indices, int(max_kv)):
            win_r = int(q.shape[0]) - 1
    kwargs["ori_win_left"] = int(kwargs.get("ori_win_left", 0))
    kwargs["ori_win_right"] = win_r
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    _ensure_sinks(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)


# ---------------------------------------------------------------------------
# A3 compiled-binding adapters (DSpark draft only)
# ---------------------------------------------------------------------------


def _a3_draft_topk_from_kwargs(q: torch.Tensor, kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build logical indices and per-query lengths for a DSpark draft call.

    The caller (dsa_v1, borrow contract) passes physical indices with a
    causal window; the compiled binding needs logical positions with
    per-query topk lengths and mask_mode=0. Every query in a draft block
    sees the same trailing window + draft block, so both are uniform per
    request.
    """
    cu_q = kwargs["cu_seqlens_q"]
    seqused = kwargs["seqused_kv"]
    win_left = int(kwargs.get("ori_win_left", 0))
    device = q.device

    num_reqs = seqused.shape[0]
    total_q = int(cu_q[-1].item())
    query_lens = cu_q[1:] - cu_q[:-1]

    # visible set per request: trailing window + draft block (uniform)
    vis = torch.minimum(seqused, query_lens + win_left)
    max_vis = int(vis.max().item()) if num_reqs > 0 else 1

    # per-query lengths: repeat each request's visible count
    lengths = torch.repeat_interleave(vis, query_lens).unsqueeze(1).to(torch.int32)

    # logical indices: trailing positions per request, padded to max_vis
    indices = torch.full((total_q, 1, max_vis), -1, dtype=torch.int32, device=device)
    for b in range(num_reqs):
        start_q = int(cu_q[b].item())
        end_q = int(cu_q[b + 1].item())
        kv_len = int(seqused[b].item())
        v = int(vis[b].item())
        if v > 0 and end_q > start_q:
            positions = torch.arange(kv_len - v, kv_len, dtype=torch.int32, device=device)
            indices[start_q:end_q, 0, :v] = positions

    return indices, lengths


def a3_sparse_flash_mla_metadata(**kwargs):
    """Adapt DSA draft metadata kwargs to the compiled SparseFlashMla binding.

    The compiled binding (``torch.ops._C_ascend``) reaches the same
    aclnnSparseFlashMla kernel as the CANN toolkit namespace on A5, but
    takes explicit ``seqused_ori_kv`` / ``max_seqlen_ori_kv`` names and
    always addresses the paged cache. DSA names are translated here.
    """
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = int(kwargs.pop("max_seqlen_kv"))
    kwargs.pop("seqused_q", None)
    kwargs.pop("layout_q", None)
    kwargs.pop("layout_kv", None)
    _add_compressed_kv_lengths(kwargs)
    num_heads_q = kwargs.pop("num_heads_q")
    num_heads_kv = kwargs.pop("num_heads_kv")
    head_dim = kwargs.pop("head_dim")
    call_kwargs = dict(
        cu_seqlens_q=kwargs.pop("cu_seqlens_q").int(),
        seqused_ori_kv=kwargs.pop("seqused_ori_kv", None),
        seqused_cmp_kv=kwargs.pop("seqused_cmp_kv", None),
        cmp_residual_kv=kwargs.pop("cmp_residual_kv", None),
        ori_topk_length=kwargs.pop("ori_topk_length"),
        cmp_topk_length=kwargs.pop("cmp_topk_length", None),
        batch_size=kwargs.pop("batch_size"),
        max_seqlen_q=int(kwargs.pop("max_seqlen_q")),
        max_seqlen_ori_kv=int(kwargs.pop("max_seqlen_ori_kv", 0)),
        max_seqlen_cmp_kv=int(kwargs.pop("max_seqlen_cmp_kv", 0) or 0),
        ori_topk=kwargs.pop("ori_topk"),
        cmp_topk=int(kwargs.pop("cmp_topk", 0)),
        cmp_ratio=int(kwargs.pop("cmp_ratio", 1)),
        ori_mask_mode=int(kwargs.pop("ori_mask_mode")),
        cmp_mask_mode=int(kwargs.pop("cmp_mask_mode")),
        ori_win_left=int(kwargs.pop("ori_win_left")),
        ori_win_right=int(kwargs.pop("ori_win_right")),
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=bool(kwargs.pop("has_ori_kv", True)),
        has_cmp_kv=bool(kwargs.pop("has_cmp_kv", False)),
    )
    # reject silently-dropped names so new DSA kwargs fail loudly
    if kwargs:
        raise ValueError(f"Unexpected a3 metadata kwargs: {sorted(kwargs)}")
    if call_kwargs["seqused_ori_kv"] is not None and torch.is_tensor(call_kwargs["seqused_ori_kv"]):
        # zero-length sequences (cudagraph padding) crash the AICPU tiler
        call_kwargs["seqused_ori_kv"] = call_kwargs["seqused_ori_kv"].clamp(min=1)
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(num_heads_q, num_heads_kv, head_dim, **call_kwargs)


def a3_sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt DSA draft attention kwargs to the compiled SparseFlashMla binding.

    A DSpark draft call (indices present, prefill-only cu_seqlens_ori_kv
    absent) arrives in borrow mode: physical indices with a causal window.
    The compiled binding needs logical indices with per-query topk lengths
    and mask_mode=0, so the adapter builds them from the caller's kwargs.
    Vision prefill keeps its indices for span-level bidirectional masking.
    """
    if kwargs.get("ori_sparse_indices") is not None and kwargs.get("cu_seqlens_ori_kv") is None:
        indices, lengths = _a3_draft_topk_from_kwargs(q, kwargs)
        kwargs["ori_sparse_indices"] = indices
        kwargs["ori_topk_length"] = lengths
        kwargs["ori_mask_mode"] = 0
        kwargs["cmp_mask_mode"] = 0
        kwargs["ori_win_left"] = 0
        kwargs["ori_win_right"] = 0
        kwargs["cmp_ratio"] = 0
        kwargs["topk_value_mode"] = 1

    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    kwargs.pop("layout_q", None)
    kwargs.pop("layout_kv", None)
    _add_compressed_kv_lengths(kwargs)
    _ensure_sinks(kwargs)
    call_kwargs = dict(
        ori_kv=kwargs.pop("ori_kv", None),
        cmp_kv=kwargs.pop("cmp_kv", None),
        ori_sparse_indices=kwargs.pop("ori_sparse_indices"),
        cmp_sparse_indices=kwargs.pop("cmp_sparse_indices", None),
        ori_block_table=kwargs.pop("ori_block_table"),
        cmp_block_table=kwargs.pop("cmp_block_table", None),
        cu_seqlens_q=kwargs.pop("cu_seqlens_q").int(),
        seqused_q=kwargs.pop("seqused_q", None),
        seqused_ori_kv=kwargs.pop("seqused_ori_kv", None),
        seqused_cmp_kv=kwargs.pop("seqused_cmp_kv", None),
        cmp_residual_kv=kwargs.pop("cmp_residual_kv", None),
        ori_topk_length=kwargs.pop("ori_topk_length"),
        cmp_topk_length=kwargs.pop("cmp_topk_length", None),
        sinks=kwargs.pop("sinks"),
        metadata=kwargs.pop("metadata"),
        softmax_scale=float(kwargs.pop("softmax_scale", 1.0)),
        cmp_ratio=int(kwargs.pop("cmp_ratio", 1)),
        ori_mask_mode=int(kwargs.pop("ori_mask_mode")),
        cmp_mask_mode=int(kwargs.pop("cmp_mask_mode")),
        ori_win_left=int(kwargs.pop("ori_win_left")),
        ori_win_right=int(kwargs.pop("ori_win_right")),
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=int(kwargs.pop("topk_value_mode", 1)),
        return_softmax_lse=False,
    )
    # reject silently-dropped names so new DSA kwargs fail loudly
    if kwargs:
        raise ValueError(f"Unexpected a3 attention kwargs: {sorted(kwargs)}")
    if call_kwargs["seqused_ori_kv"] is not None and torch.is_tensor(call_kwargs["seqused_ori_kv"]):
        # zero-length sequences (cudagraph padding) crash the AICPU tiler
        call_kwargs["seqused_ori_kv"] = call_kwargs["seqused_ori_kv"].clamp(min=1)
    return torch.ops._C_ascend.npu_sparse_flash_mla(q, **call_kwargs)
