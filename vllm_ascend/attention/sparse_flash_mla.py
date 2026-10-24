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
    # Paged KV uses block tables and seqused lengths, not TND cumulative offsets.
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
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


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt existing DSA attention kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs["layout_kv"] = "PA_BBND"
    # Paged KV uses block tables and seqused lengths, not TND cumulative offsets.
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    _ensure_sinks(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)


def _check_a3_adapter_leftovers(kwargs: dict[str, Any], op_name: str) -> None:
    if kwargs:
        raise ValueError(f"Unexpected {op_name} kwargs for the compiled SparseFlashMla binding: {sorted(kwargs)}")


def _clamp_a3_zero_length_seqused(seqused):
    """Clamp zero-length KV sequences to one token for the compiled binding.

    Cudagraph padding adds dummy requests with no query tokens and no cached
    KV, i.e. ``seqused == 0``. The CANN toolkit SparseFlashMla op tolerates
    such entries, but the compiled binding's AICPU tiler computes an invalid
    index for a zero-length sequence (aicpu exception in
    SparseFlashMlaMetadata, IndexCheck). Dummy requests carry no queries, so
    the extra tile from the clamp is never consumed; real requests always
    have at least the current token cached, so the clamp is inert for them.
    """
    if torch.is_tensor(seqused):
        return seqused.clamp(min=1)
    return seqused


def a3_sparse_flash_mla_metadata(**kwargs):
    """Adapt DSA draft metadata kwargs to the compiled SparseFlashMla binding.

    A3-class hardware reaches the same aclnnSparseFlashMla kernel through the
    vllm-ascend compiled binding (the v41 integration) instead of the CANN
    toolkit namespace. The compiled op takes explicit ``seqused_ori_kv`` /
    ``max_seqlen_ori_kv`` names, no ``device`` pin, and always addresses the
    paged cache, so the DSA names are translated like the toolkit adapter
    above and ``seqused_q`` is dropped (the v41 call does not pass it either).
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
    _check_a3_adapter_leftovers(kwargs, "metadata")
    call_kwargs["seqused_ori_kv"] = _clamp_a3_zero_length_seqused(call_kwargs["seqused_ori_kv"])
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(num_heads_q, num_heads_kv, head_dim, **call_kwargs)


def a3_sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt DSA draft attention kwargs to the compiled SparseFlashMla binding.

    Returns the op's ``(output, softmax_lse)`` tuple so the shared
    ``attn_op(q, **kwargs)[0]`` call site keeps working.
    """
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
    _check_a3_adapter_leftovers(kwargs, "attention")
    call_kwargs["seqused_ori_kv"] = _clamp_a3_zero_length_seqused(call_kwargs["seqused_ori_kv"])
    return torch.ops._C_ascend.npu_sparse_flash_mla(q, **call_kwargs)
