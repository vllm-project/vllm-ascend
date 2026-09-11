"""Temporary quantization shim for the reduced dummy DSV4 model.

The shim is opt-in and exists only to let the reduced dummy model exercise
the FXRT graph.  Real model execution keeps the torch_npu implementation
unchanged.
"""

import os

import torch
import torch_npu


_INSTALLED = False


def install_dummy_quant_matmul_shim() -> None:
    global _INSTALLED
    if _INSTALLED or os.getenv("VLLM_ASCEND_FXRT_DUMMY_QUANT") != "1":
        return

    original = torch_npu.npu_quant_matmul

    def quant_matmul(x1, x2, scale, *args, **kwargs):
        if x1.dtype == torch.bfloat16 and x2.dtype == torch.int8:
            x1, per_token_scale = torch_npu.npu_dynamic_quant(x1)
            kwargs.setdefault("pertoken_scale", per_token_scale)
        return original(x1, x2, scale, *args, **kwargs)

    torch_npu.npu_quant_matmul = quant_matmul
    _INSTALLED = True
