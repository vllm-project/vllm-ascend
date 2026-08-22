"""Fix EZ1001 dtype mismatch in AscendRMSNorm on Ascend NPU.

LongCat MLA/MoE kernels may produce float32 while RMSNorm weights are
bfloat16, causing ACLNN dtype mismatch.

Migrated from EasyInfer easyinfer/plugins/vllm_ascend/fix_layernorm_dtype.py.
Renamed torch custom ops: easyinfer:: → vllm_ascend::longcat.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


@torch.library.custom_op("vllm_ascend::rms_norm_guard_x", mutates_args=())
def _rms_norm_guard_x(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Opaque runtime dtype guard for compiled graphs."""
    if x.dtype != weight.dtype:
        return x.to(dtype=weight.dtype)
    return x.clone()


@_rms_norm_guard_x.register_fake
def _rms_norm_guard_x_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("vllm_ascend::rms_norm_guard_residual", mutates_args=())
def _rms_norm_guard_residual(
    residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if residual.dtype != weight.dtype:
        return residual.to(dtype=weight.dtype)
    return residual.clone()


@_rms_norm_guard_residual.register_fake
def _rms_norm_guard_residual_fake(
    residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return residual.clone()


def patch() -> None:
    """Wrap AscendRMSNorm.forward_oot to cast inputs to weight dtype."""
    import vllm_ascend.ops.layernorm as _ln

    _RMS = getattr(_ln, "AscendRMSNorm", None)
    if _RMS is None:
        return
    if getattr(_RMS, "_ez_lndtype_patched", False):
        return
    _RMS._ez_lndtype_patched = True  # type: ignore[attr-defined]

    _original_oot = _RMS.forward_oot

    def _dtype_safe_forward_oot(
        self: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.weight.dtype
        if torch.compiler.is_compiling():
            x = torch.ops.vllm_ascend.rms_norm_guard_x(x, self.weight)
            if residual is not None:
                residual = torch.ops.vllm_ascend.rms_norm_guard_residual(
                    residual, self.weight
                )
        else:
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)
            if residual is not None and residual.dtype != target_dtype:
                residual = residual.to(dtype=target_dtype)
        return _original_oot(self, x, residual)

    _RMS.forward_oot = _dtype_safe_forward_oot
    logger.info("[fix_layernorm_dtype] AscendRMSNorm.forward_oot dtype guard applied")
