# SPDX-License-Identifier: Apache-2.0
"""PyTorch HyperConnection operations for the NPU bootstrap path."""

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    if x.shape[-1] % num_groups:
        raise ValueError("Grouped RMSNorm input is not divisible by num_groups")
    group_dim = x.shape[-1] // num_groups
    if weight.numel() not in (group_dim, x.shape[-1]):
        raise ValueError("Grouped RMSNorm weight has an invalid size")
    grouped = x.reshape(-1, num_groups, group_dim).float()
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    affine = weight.float().reshape(1, -1, group_dim)
    if weight.numel() == group_dim:
        affine = affine[:, :1]
    return (normalized * (1.0 + affine)).reshape_as(x).to(x.dtype)


def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if hc_count <= 0:
        raise ValueError("hc_count must be positive")
    return F.silu(x.float() / hc_count).to(x.dtype)


def _hc_gate_mix(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    if x.shape != gate.shape or x.shape[-1] % hc_count:
        raise ValueError("HC gate and input shapes are incompatible")
    hidden_size = x.shape[-1] // hc_count
    mixed = torch.sigmoid(gate.float()) * x.float()
    return mixed.reshape(-1, hc_count, hidden_size).mean(1).to(x.dtype)


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if residual.shape[-1] % hc_count:
        raise ValueError("HC residual is not divisible by hc_count")
    hidden_size = residual.shape[-1] // hc_count
    if block_output.shape[-1] != hidden_size:
        raise ValueError("HC block output has an invalid size")
    injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    combined = residual.float().reshape(-1, hc_count, hidden_size)
    combined = combined + block_output.float().unsqueeze(1) * injection.unsqueeze(-1)
    return combined.reshape_as(residual).to(residual.dtype)


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = _hc_combine(
        residual, block_output, injection_logits, hc_count
    )
    normalized = _grouped_gemma_rmsnorm(
        combined, norm_weight, eps, hc_count
    )
    return combined, normalized


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    del args
    return x.new_empty(x.shape)


def _hc_gate_mix_fake(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    del gate
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return residual.new_empty(residual.shape)


def _hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_grouped_gemma_rmsnorm",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_silu",
    op_func=_hc_silu,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_gate_mix",
    op_func=_hc_gate_mix,
    fake_impl=_hc_gate_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm",
    op_func=_hc_combine_norm,
    fake_impl=_hc_combine_norm_fake,
)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm(
        x, weight, eps, num_groups
    )


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_silu(x, hc_count)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_combine(
        residual, block_output, injection_logits, hc_count
    )


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_hc_combine_norm(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
