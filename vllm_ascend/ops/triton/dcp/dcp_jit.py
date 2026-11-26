# SPDX-License-Identifier: Apache-2.0
"""Compile-only owners for the unchanged DCP Triton launch functions."""

from dataclasses import dataclass
from typing import Any

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel


def _launch_kernel(kernel, grid, *args, **kwargs):
    kernel[grid](*args, **kwargs)


class DcpJitKernel(AscendTritonJitKernel["DcpJitKernel.CompileKey"]):
    """Warm the same JITFunction used by runtime, without allocating payloads.

    Instances belong to the module's fixed kernel inventory. Only the shared
    runner registry retains registrations; no requests register new variants.
    """

    @dataclass(frozen=True)
    class CompileKey:
        inputs: tuple[tuple[str, Any], ...]

    def __init__(self, kernel, dynamic_scalars=()):
        self.kernel = kernel
        self.dynamic_scalars = dynamic_scalars
        super().__init__()

    def key_from_launch(self, *args, **kwargs):
        inputs = dict(zip(self.kernel.arg_names, args))
        inputs.update(kwargs)
        for name, value in inputs.items():
            if isinstance(value, TritonWarmupTensor):
                # Pointer dtype/alignment specialize Triton; tensor shape is
                # expressed by the explicit scalar launch arguments instead.
                inputs[name] = TritonWarmupTensor(value.dtype, aligned=value.aligned)
            # All eight DCP kernels spell their constexpr parameters in
            # uppercase; launch options such as multibuffer are booleans.
            elif type(value) is int and not name.isupper():
                representative = triton_scalar_specialization_rep(value)
                if name in self.dynamic_scalars:
                    # do_not_specialize keeps the integer type, not exact-one
                    # or divisibility. These kernels receive nonnegative i32
                    # or i64 sizes/strides.
                    representative = 2 if representative < (1 << 31) else (1 << 31) + 1
                inputs[name] = representative
        return self.CompileKey(tuple(inputs.items()))

    def get_warmup_keys(self, keys):
        return list(keys)

    def warmup_inputs(self, compile_key):
        return {"launch_kwargs": dict(compile_key.inputs)}

    def __call__(self, launch_kwargs):
        # AscendTritonJitKernel.compile sets _warming, so launch calls native
        # kernel.warmup(grid=(1,)) and never executes the kernel or HCCL.
        return self.launch(((1,), launch_kwargs), {})
