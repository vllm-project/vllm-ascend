# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
import torch_npu


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla operators exposed by torch_npu/op-plugin."""
    try:
        attention_op = torch_npu.npu_sparse_flash_mla
        metadata_op = torch_npu.npu_sparse_flash_mla_metadata
    except AttributeError as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires torch_npu APIs "
            "npu_sparse_flash_mla and npu_sparse_flash_mla_metadata from a "
            "matching CANN 9.2 toolkit and torch_npu/op-plugin build."
        ) from exc
    return attention_op, metadata_op


def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    if cmp_ratio <= 1 or seqused_ori_kv is None:
        return

    if kwargs.get("seqused_cmp_kv") is None:
        kwargs["seqused_cmp_kv"] = seqused_ori_kv // cmp_ratio
    if kwargs.get("cmp_residual_kv") is None:
        kwargs["cmp_residual_kv"] = seqused_ori_kv % cmp_ratio
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def sparse_flash_mla_metadata(**kwargs):
    """Adapt the existing DSA metadata convention to SparseFlashMla."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _add_compressed_kv_lengths(kwargs)

    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt the existing DSA attention convention to SparseFlashMla."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _add_compressed_kv_lengths(kwargs)

    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
