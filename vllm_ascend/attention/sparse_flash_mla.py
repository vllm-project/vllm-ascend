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


def _apply_dspark_draft_kwargs(kwargs: dict[str, Any]) -> None:
    """Rewrite the DSpark draft (per-query top-k) attention contract.

    DSpark supplies visibility through ``ori_sparse_indices`` with per-query
    ``ori_topk_length``: both window bounds are disabled and the inactive
    compressed branch takes the toolkit's legal ratio of 1.
    """
    kwargs.update(cmp_ratio=1, ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1)


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
    if kwargs.get("ori_topk_length") is not None:
        # DSpark draft call (per-query top-k visibility): the adapter owns the
        # draft contract, so callers pass plain DSA kwargs.
        _apply_dspark_draft_kwargs(kwargs)
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
    if kwargs.get("ori_topk_length") is not None:
        # DSpark draft call (per-query top-k visibility): the adapter owns the
        # draft contract, so callers pass plain DSA kwargs.
        _apply_dspark_draft_kwargs(kwargs)
        kwargs["topk_value_mode"] = 1
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    _ensure_sinks(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)


def build_dspark_sparse_flash_mla_metadata(
    num_heads_q: int,
    head_dim: int,
    seqused_q: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen_q: int | torch.Tensor,
    max_seqlen_kv: int | torch.Tensor,
    ori_topk_length: torch.Tensor,
    ori_topk: int,
) -> torch.Tensor:
    """Build the DSpark draft metadata call for SparseFlashMla.

    The DSpark draft only runs on the BF16 SparseFlashMla plan, whose
    metadata op is this module's ``sparse_flash_mla_metadata``; routing the
    call through that adapter applies the draft contract
    (``_apply_dspark_draft_kwargs``). The plan's metadata kwargs are only a
    device pin the adapter drops, and the adapter forces ``layout_kv``, so
    neither is passed here.
    """
    return sparse_flash_mla_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        head_dim=head_dim,
        cu_seqlens_q=query_start_loc,
        seqused_q=seqused_q,
        seqused_kv=seq_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=len(seq_lens),
        ori_topk_length=ori_topk_length,
        ori_topk=ori_topk,
        layout_q="TND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )


def write_dspark_topk_lengths(per_token_lens: torch.Tensor, lengths_output: torch.Tensor) -> torch.Tensor:
    """Adapt DSpark per-token lengths to the SparseFlashMla top-k interface.

    The op takes per-query lengths as [num_tokens, 1]; writing through the
    caller-owned persistent view keeps the tensor address stable for ACL
    graph capture. Returns ``lengths_output``.
    """
    lens = per_token_lens.unsqueeze(1)
    if lengths_output.shape != lens.shape:
        raise ValueError("DSpark topk lengths output must have shape [num_tokens, 1]")
    lengths_output.copy_(lens)
    return lengths_output
