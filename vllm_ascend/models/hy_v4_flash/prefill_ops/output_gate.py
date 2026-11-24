# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu

from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg


def gate_dynamic_mx_quant(value: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BF16 sigmoid and multiplication before CANN's MXFP8 quantizer."""
    product = value * gate_logits.sigmoid()
    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        product, dst_type=torch.float8_e4m3fn, scale_alg=DynamicMXQuantScaleAlg.MAX_EXPONENT
    )
    return quantized, scale.view(torch.uint8)
