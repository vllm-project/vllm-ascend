# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt DSA arguments to the installed MixedQuantSparseFlashMla operators."""

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


def mixed_quant_sparse_flash_mla_metadata(**kwargs):
    _adapt(kwargs)
    # Metadata requires explicit compressed lengths; fused TQ attention derives them.
    _add_compressed_kv_lengths(kwargs)
    _, metadata_op = _get_mixed_quant_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def mixed_quant_sparse_flash_mla(q, **kwargs):
    _adapt(kwargs)
    _ensure_sinks(kwargs)
    attention_op, _ = _get_mixed_quant_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
