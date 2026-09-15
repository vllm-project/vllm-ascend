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


def _adapt_compressed_kv(kwargs: dict[str, Any], *, has_cmp_kv: bool) -> None:
    # SparseFlashMla validates the inactive branch too; its default mask is 3.
    # Use the same inactive-branch attributes for metadata and execution.
    if not has_cmp_kv:
        kwargs["cmp_ratio"] = 1
        kwargs["cmp_mask_mode"] = 0
        return
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    if cmp_ratio <= 1 or seqused_ori_kv is None:
        return
    kwargs.setdefault("seqused_cmp_kv", seqused_ori_kv // cmp_ratio)
    kwargs.setdefault("cmp_residual_kv", seqused_ori_kv % cmp_ratio)
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


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
    _adapt_compressed_kv(kwargs, has_cmp_kv=kwargs.get("has_cmp_kv", True))
    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


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
    _adapt_compressed_kv(kwargs, has_cmp_kv=kwargs.get("cmp_kv") is not None)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
