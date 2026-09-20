# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


# FLASHMLA[ABI]: keep every scalar/layout argument shared by the two-stage
# invocation in one object. The external operator cannot validate that the
# metadata and main calls agree, so spelling these independently is unsafe.
@dataclass(frozen=True)
class FlashMLAContract:
    """Shared ABI contract for metadata generation and FlashMLA execution."""

    num_heads_q: int
    num_heads_kv: int = 1
    head_dim_qk: int = 576
    head_dim_v: int = 512
    max_seqlen_q: int = -1
    max_seqlen_kv: int = -1
    mask_mode: int = 0
    layout_q: str = "TND"
    layout_kv: str = "PA_BBND"
    layout_out: str = "NTD"
    softmax_scale: float = 1.0
    return_softmax_lse: bool = False

    def __post_init__(self) -> None:
        if self.num_heads_q <= 0 or self.num_heads_kv != 1:
            raise ValueError("FlashMLA requires positive Q heads and one KV head")
        if (self.head_dim_qk, self.head_dim_v) != (576, 512):
            raise ValueError("FlashMLA requires head_dim_qk=576 and head_dim_v=512")
        if self.mask_mode not in (0, 3):
            raise ValueError("FlashMLA mask_mode must be 0 (none) or 3 (causal)")
        if self.layout_q != "TND" or self.layout_kv != "PA_BBND" or self.layout_out != "NTD":
            raise ValueError("This MLA integration uses TND/PA_BBND/NTD layouts")

    def with_softmax_scale(self, softmax_scale: float) -> "FlashMLAContract":
        return replace(self, softmax_scale=float(softmax_scale))

    def metadata_kwargs(self, cu_seqlens_q: torch.Tensor, seqused_q: torch.Tensor) -> dict[str, Any]:
        return {
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_v": self.head_dim_v,
            "mask_mode": self.mask_mode,
            "layout_q": self.layout_q,
        }

    def attention_kwargs(
        self,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor,
        attn_mask: torch.Tensor | None,
        metadata: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "attn_mask": attn_mask,
            "metadata": metadata,
            "head_dim_v": self.head_dim_v,
            "softmax_scale": self.softmax_scale,
            "mask_mode": self.mask_mode,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "layout_q": self.layout_q,
            "layout_kv": self.layout_kv,
            "layout_out": self.layout_out,
            "return_softmax_lse": self.return_softmax_lse,
        }


def validate_flash_mla_kv_cache(
    k_cache: torch.Tensor,
    *,
    expected_dtype: torch.dtype | None = None,
    expected_device: torch.device | None = None,
) -> None:
    """Validate #16456's strided PA_BBND cache without materializing a copy."""

    # FLASHMLA[ABI]: this is deliberately a read-only check. Do not repair a
    # bad view with reshape/contiguous: the first stride is physical page
    # spacing that the PA kernel must consume.
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("FlashMLA k_cache must be a Tensor")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.shape[-1] != 576:
        raise ValueError(
            "FlashMLA PA_BBND k_cache must have shape [P, S, 1, 576], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.stride(3) != 1 or k_cache.stride(2) != 576 or k_cache.stride(1) != 576:
        raise ValueError(
            "FlashMLA PA_BBND k_cache must keep token strides [576, 576, 1], "
            f"got {k_cache.stride()}"
        )
    if k_cache.stride(0) < k_cache.shape[1] * 576:
        raise ValueError("FlashMLA k_cache pages must not overlap")
    if expected_dtype is not None and k_cache.dtype != expected_dtype:
        raise ValueError(f"FlashMLA k_cache dtype must be {expected_dtype}, got {k_cache.dtype}")
    if expected_device is not None and k_cache.device != expected_device:
        raise ValueError(f"FlashMLA k_cache must be on {expected_device}, got {k_cache.device}")


# FLASHMLA[EXTERNAL]: the Python integration follows #16468, but operator
# registration comes from the installed wheel rather than _C_ascend. The
# matching CANN/custom runtime environment must already be active in the worker.
@lru_cache
def _get_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load packaged CANN 9.2 FlashMLA operators after device selection.

    The custom ``.run`` installs the operator binaries, while the
    ``cann_ops_transformer`` wheel registers their PyTorch dispatchers. Keep
    the dependency lazy so configurations without FlashMLA remain unaffected.
    """
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return (
            namespace.flash_mla_with_kvcache,
            namespace.flash_mla_with_kvcache_metadata,
        )
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "A5 FlashMLA requires flash_mla_with_kvcache and "
            "flash_mla_with_kvcache_metadata from a matching CANN 9.2 "
            "cann_ops_transformer custom package and Python wheel."
        ) from exc


def ensure_flash_mla_ops_loaded() -> None:
    """Fail during FlashMLA initialization when the external ABI is absent."""
    _get_flash_mla_ops()


def flash_mla_with_kvcache_metadata(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Dispatch metadata generation through the external CANN package."""
    _, metadata_op = _get_flash_mla_ops()
    return metadata_op(*args, **kwargs)


def flash_mla_with_kvcache(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch FlashMLA through the external CANN package."""
    attention_op, _ = _get_flash_mla_ops()
    return attention_op(*args, **kwargs)
