# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, TypeVar

import torch
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

CompileKeyT = TypeVar("CompileKeyT")


class AscendTileLangJitKernel(VllmJitKernel[CompileKeyT]):
    """Compile TileLang PrimFuncs through the runner's JIT warmup registry.

    Subclasses own dispatch and warmup keys, including every shape, dtype,
    stride and compile option that changes generated code. The returned
    TileLang executor launches on the current PyTorch NPU stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device_index: int | None = None

    @abstractmethod
    def get_prim_func(self, compile_key: CompileKeyT) -> Any:
        """Construct a PrimFunc, importing TileLang inside this method."""
        raise NotImplementedError

    def get_compile_options(self, compile_key: CompileKeyT) -> Mapping[str, Any]:
        return {}

    def _check_device(self) -> None:
        device_index = torch.npu.current_device()
        if self._device_index is None:
            self._device_index = device_index
        elif self._device_index != device_index:
            raise RuntimeError("Use a separate TileLang kernel wrapper for each NPU worker device.")

    def get_kernel(self, compile_key: CompileKeyT) -> Any:
        self._check_device()
        return self._get_or_compile(compile_key)

    def compile(self, compile_key: CompileKeyT) -> None:
        self._check_device()
        if compile_key in self._compiled_cache:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("TileLang specialization was not warmed before ACL Graph capture.")
        import tilelang

        self._compiled_cache[compile_key] = tilelang.compile(
            self.get_prim_func(compile_key),
            target="ascend",
            execution_backend="tvm_ffi",
            **self.get_compile_options(compile_key),
        )
