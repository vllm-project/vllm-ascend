# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash-Next QKV preparation with an external gate buffer."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import (
    _triton_split_qkv_rmsnorm_mrope,
)


def _qwen4_exp_split_qkv_rmsnorm_mrope_out(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    gate_output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V and write the packed sigmoid gate to its final buffer."""
    q_output, k_output, v_output, _ = _triton_split_qkv_rmsnorm_mrope(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        has_gate=True,
        positions=positions,
        # Qwen3.8 Flash-Next uses Gemma RMSNorm: the checkpoint stores
        # ``weight`` while the effective scale is ``1 + weight``.
        weight_offset=1.0,
        gate_output=gate_output,
    )
    return q_output, k_output, v_output


def _qwen4_exp_split_qkv_rmsnorm_mrope_out_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    gate_output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos_sin, gate_output, eps, mrope_section
    del is_interleaved, rope_dim, positions
    num_tokens = qkv.shape[0]
    q_output = torch.empty(
        num_tokens,
        num_q_heads * head_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )
    k_output = torch.empty(
        num_tokens,
        num_kv_heads * head_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )
    return q_output, k_output, torch.empty_like(k_output)


direct_register_custom_op(
    op_name="qwen4_exp_split_qkv_rmsnorm_mrope_out",
    op_func=_qwen4_exp_split_qkv_rmsnorm_mrope_out,
    fake_impl=_qwen4_exp_split_qkv_rmsnorm_mrope_out_fake,
    mutates_args=["gate_output"],
    dispatch_key="PrivateUse1",
)


def qwen4_exp_split_qkv_rmsnorm_mrope_out(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    gate_output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_split_qkv_rmsnorm_mrope_out(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        gate_output,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        positions,
    )


__all__ = ["qwen4_exp_split_qkv_rmsnorm_mrope_out"]
