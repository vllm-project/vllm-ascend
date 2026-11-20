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
