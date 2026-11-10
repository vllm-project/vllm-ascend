# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-only compiled launchers, imported after the hardware gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.ops.kda_state_copy_plan import StateCopyCompileKey
from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

if TYPE_CHECKING:
    from vllm_ascend.ops.triton.kda_state_copy_kernel import _CompiledKernel


class StateCopyJitKernel(AscendTritonJitKernel[StateCopyCompileKey]):
    """Worker-owned compiled launchers scheduled by JitWarmupRegistry.

    Metadata-only warmup compiles without allocating or launching device data.
    The cached executor includes its ABI constants, resolved once at compile
    time. With JIT warmup disabled, the framework cache compiles on first use.
    """

    def __init__(self, kernel: Any, constexpr_names: tuple[str, ...]) -> None:
        super().__init__()
        self.kernel = kernel
        self._constexpr_names = constexpr_names

    @property
    def supports_parallel_compile(self) -> bool:
        # Signature inspection needs a completed CompiledKernel rather than
        # the futures returned inside Triton's asynchronous compile pool.
        return False

    def get_warmup_keys(self, key: StateCopyCompileKey) -> list[StateCopyCompileKey]:
        return [key]

    def warmup_inputs(self, compile_key: StateCopyCompileKey) -> dict[str, Any]:
        return dict(compile_key.arguments)

    def compile(self, compile_key: StateCopyCompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        inputs = self.warmup_inputs(compile_key)
        with torch.npu.device(compile_key.device):
            compiled = self.kernel.warmup(grid=(1,), **inputs)
        constants = (
            tuple(inputs[name] for name in self._constexpr_names)
            if "constexpr" in compiled.src.signature.values()
            else ()
        )
        self._compiled_cache[compile_key] = compiled, constants

    def executor(self, compile_key: StateCopyCompileKey) -> tuple[_CompiledKernel, tuple[Any, ...]]:
        return self._get_or_compile(compile_key)
