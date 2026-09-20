"""Torch implementations of the Qwen4Exp HyperConnection glue operations.

The CUDA implementation uses PDL-enabled Triton kernels.  Keeping these
small elementwise operations in PyTorch lets torch-npu compile them into the
surrounding graph without importing CUDA-only ``tl.extra.cuda`` primitives.
"""

import torch
import torch.nn.functional as F


def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    rows, dim = x.shape
    if dim % num_groups:
        raise ValueError(f"hidden dimension {dim} is not divisible by {num_groups}")
    group_dim = dim // num_groups
    grouped = x.float().view(rows, num_groups, group_dim)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    if weight.numel() == group_dim:
        affine = weight.float().view(1, 1, group_dim)
    elif weight.numel() == dim:
        affine = weight.float().view(1, num_groups, group_dim)
    else:
        raise ValueError(f"expected {group_dim} or {dim} norm weights, got {weight.numel()}")
    return (normalized * (1 + affine)).to(x.dtype).view_as(x)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return F.silu(x.float() / hc_count).to(x.dtype)


def hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if x.shape != gate.shape:
        raise ValueError("HyperConnection input and gate shapes must match")
    group_dim = x.shape[-1] // hc_count
    mixed = torch.sigmoid(gate.float()) * x.float()
    return mixed.view(-1, hc_count, group_dim).mean(1).to(x.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    group_dim = residual.shape[-1] // hc_count
    injection = 2 * torch.sigmoid(injection_logits.float() / hc_count)
    combined = residual.float().view(-1, hc_count, group_dim)
    combined = combined + block_output.float().unsqueeze(1) * injection.unsqueeze(-1)
    return combined.to(residual.dtype).view_as(residual)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = hc_combine(residual, block_output, injection_logits, hc_count)
    normalized = grouped_gemma_rmsnorm(combined, norm_weight, eps, hc_count)
    return combined, normalized


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
