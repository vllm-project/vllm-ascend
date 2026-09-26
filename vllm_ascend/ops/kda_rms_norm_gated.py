# SPDX-License-Identifier: Apache-2.0
"""Compiled D128 BF16 RMS normalization with a strided sigmoid/SiLU gate."""

import torch


def supports_kda_rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    out: torch.Tensor | None,
) -> bool:
    def layout(tensor: torch.Tensor) -> bool:
        return (
            (tensor.ndim == 3 or (tensor.ndim == 4 and tensor.shape[0] == 1))
            and tensor.shape[-1] == 128
            and tensor.stride(-1) == 1
            and tensor.stride(-2) == 128
            and tensor.stride(-3) >= tensor.shape[-2] * 128
            and tensor.stride(-3) - tensor.shape[-2] * 128 <= (2**32 - 1) // 2
        )

    return (
        x.device.type == "npu"
        and x.dtype == gate.dtype == torch.bfloat16
        and layout(x)
        and layout(gate)
        and x.shape[-3:-1] == gate.shape[-3:-1]
        and 0 < x.shape[-2] <= 128
        and weight is not None
        and weight.shape == (128,)
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.device == gate.device == x.device
        and weight.is_contiguous()
        and (
            out is None
            or (out.shape == x.shape and out.dtype == x.dtype and out.device == x.device and out.is_contiguous())
        )
    )


def kda_rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    sigmoid_only: bool = False,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return torch.ops._C_ascend.kda_rms_norm_gated(x, gate, weight, out, eps, sigmoid_only)
