# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend base class for the shared vLLM JIT-warmup kernel framework.

``VllmTritonJitKernel`` (vllm/model_executor/warmup) gates parallel warmup
compilation behind ``current_platform.is_cuda()``. On Ascend the same
``compile_triton_warmup`` pool (one ThreadPoolExecutor plus
``triton.AsyncCompileMode``) drives the CANN compiler instead; whether a
CANN build tolerates concurrent compiles is a deployment property, so the
gate here mirrors upstream minus the CUDA check. If a CANN build turns out
to be unsafe under concurrency, ``VLLM_TRITON_JIT_WARMUP_NUM_THREADS=1``
falls back to serial warmup with no code change.
"""

from __future__ import annotations

from typing import TypeVar

from vllm import envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import VllmTritonJitKernel
from vllm.triton_utils import triton

CompileKeyT = TypeVar("CompileKeyT")


class AscendTritonJitKernel(VllmTritonJitKernel[CompileKeyT]):
    """``VllmTritonJitKernel`` with the parallel-compile gate opened for CANN."""

    @property
    def supports_parallel_compile(self) -> bool:
        return (
            not self._run_autotune
            and getattr(triton, "AsyncCompileMode", None) is not None
            and envs.VLLM_TRITON_JIT_WARMUP_NUM_THREADS > 1
        )
