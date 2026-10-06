# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic Flash Q/W fusion; static weight packing, unchanged KV layout."""

import math

import torch
import torch_npu
from torch import nn


class IndexerQWFusion(nn.Module):
    @staticmethod
    def supports(wq_b, weights_proj, weights_scale: float) -> bool:
        """Keep other checkpoints on their original projection implementation."""
        weight = wq_b.weight
        scales = getattr(wq_b, "weight_scale", None)
        projection = weights_proj.weight
        scheme = getattr(getattr(wq_b, "quant_method", None), "quant_method", None)
        return (
            tuple(weight.shape) == (1280, 4096)
            and weight.dtype == torch.float8_e4m3fn
            and scales is not None
            and tuple(scales.shape) == (20, 4096, 2)
            and scales.dtype == torch.uint8
            and tuple(projection.shape) == (32, 5120)
            and projection.dtype == torch.bfloat16
            and math.isclose(weights_scale, 1 / 64, rel_tol=1e-12, abs_tol=0)
            and hasattr(scheme, "dynamic_mx_quant_scale_alg")
        )

    def __init__(self, wq_b, weights_proj, weights_scale: float):
        super().__init__()
        # Compiler import/registration and weight casts happen before warmup,
        # never at graph replay. Keep this lazy for non-A5 installations.
        from vllm_ascend.ops.pythondsl.indexer_prologue_qw_interleaved import indexer_prologue_qw, to_nz

        if tuple(wq_b.weight.shape) != (1280, 4096) or wq_b.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Q/W fusion requires postprocessed Flash MXFP8 weight [1280,4096]")
        if tuple(wq_b.weight_scale.shape) != (20, 4096, 2) or wq_b.weight_scale.dtype != torch.uint8:
            raise ValueError("Q/W fusion requires paired E8M0 scales [20,4096,2]")
        if tuple(weights_proj.weight.shape) != (32, 5120) or weights_proj.weight.dtype != torch.bfloat16:
            raise ValueError("Q/W fusion requires Flash BF16 weights_proj [32,5120]")
        if not math.isclose(weights_scale, 1 / 64, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Q/W fusion rounding is qualified only for Flash scale 1/64")
        scheme = wq_b.quant_method.quant_method
        self.scale_alg = scheme.dynamic_mx_quant_scale_alg
        self.weights_scale = 1 / 64
        self.kernel = indexer_prologue_qw
        self.register_buffer("wqb_nz", to_nz(wq_b.weight.T.contiguous()).view(torch.uint8), persistent=False)
        self.register_buffer("wqb_scale", wq_b.weight_scale.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("ww_nz", to_nz(weights_proj.weight.contiguous()), persistent=False)

    def forward(self, hidden_states, qr, cos, sin):
        tokens = qr.shape[0]
        q8, q8_scale = torch_npu.npu_dynamic_mx_quant(qr, dst_type=torch.float8_e4m3fn, scale_alg=self.scale_alg)
        q, scale, weights = self.kernel(
            hidden_states,
            q8.view(torch.uint8),
            self.wqb_nz,
            self.ww_nz,
            q8_scale.view(tokens, 20, 2),
            self.wqb_scale,
            sin.reshape(tokens, 64).float().contiguous(),
            cos.reshape(tokens, 64).float().contiguous(),
            softmax_scale=self.weights_scale,
        )
        # Current weights_proj materializes BF16 before multiplying by 1/64.
        # Power-of-two scaling commutes with BF16 rounding for normal values.
        return q, scale.reshape(tokens, 32, 4), weights.bfloat16().float()
