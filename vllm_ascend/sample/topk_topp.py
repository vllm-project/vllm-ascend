# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache

import torch
import vllm.envs as envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import is_310p, is_950
from vllm_ascend.ops.triton.v2.sample.topk_topp import apply_top_k_top_p_triton

_MaskFunction = Callable[[torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor]


@lru_cache(maxsize=1)
def _compilation_error_types() -> tuple[type[Exception], ...]:
    # Import only when actually using Triton. MLIRCompilationError is
    # specific to Triton Ascend and is absent from some Triton versions.
    from triton.compiler import errors

    return tuple(
        error_type
        for name in ("CompilationError", "MLIRCompilationError")
        if (error_type := getattr(errors, name, None)) is not None
    )


class _TopKTopPDispatcher:
    def __init__(self) -> None:
        self.compilation_failed = False

    def __call__(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        fallback: _MaskFunction,
    ) -> torch.Tensor:
        if not self.compilation_failed:
            try:
                return apply_top_k_top_p_triton(logits, k, p)
            except _compilation_error_types() as error:
                # These exceptions occur before the kernel is launched, so
                # logits are still intact. Never swallow runtime/device errors
                # or assertions, which may leave partially modified logits.
                self.compilation_failed = True
                logger.warning_once(
                    "[sample/topk_topp] Qrita compilation failed for shape=%s, "
                    "dtype=%s, device=%s, top_k=%s, top_p=%s (%s). "
                    "Using sort-based masking for this specialization.",
                    tuple(logits.shape),
                    logits.dtype,
                    logits.device,
                    k is not None,
                    p is not None,
                    type(error).__name__,
                    exc_info=True,
                )
        return fallback(logits, k, p)


@lru_cache(maxsize=128)
def _get_dispatcher(
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    k_signature: tuple | None,
    p_signature: tuple | None,
) -> _TopKTopPDispatcher:
    # Keep failure state per device/layout/filter specialization rather than
    # disabling Triton globally. The bounded cache avoids retrying a broken
    # compilation on every token while allowing other shapes to use Qrita.
    return _TopKTopPDispatcher()


def apply_top_k_top_p_with_fallback(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    fallback: _MaskFunction,
) -> torch.Tensor:
    if k is None and p is None:
        # A no-op must not allocate scratch buffers or compile an unused
        # specialization during MRV2 dummy/warmup calls.
        return logits
    if not HAS_TRITON or envs.VLLM_BATCH_INVARIANT or is_950() or is_310p():
        return fallback(logits, k, p)
    dispatcher = _get_dispatcher(
        logits.device,
        logits.dtype,
        tuple(logits.shape),
        tuple(logits.stride()),
        (k.dtype, tuple(k.shape), tuple(k.stride())) if k is not None else None,
        (p.dtype, tuple(p.shape), tuple(p.stride())) if p is not None else None,
    )
    return dispatcher(logits, k, p, fallback)
