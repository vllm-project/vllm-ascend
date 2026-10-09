# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch

from vllm_ascend.attention.utils import trans_rope_weight, transdata

MLAPO_WEIGHT_BLOCK_SIZE = (16, 32)


class StaticMLAPOWeights(NamedTuple):
    wd_qkv: torch.Tensor
    deq_scale_qkv: torch.Tensor
    quant_bias_qkv: torch.Tensor
    wu_q: torch.Tensor
    qb_deq_scl: torch.Tensor
    qb_qt_bias: torch.Tensor


def prepare_static_mlapo_weights(
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_weight: torch.Tensor,
    q_deq_scale: torch.Tensor,
    q_quant_bias: torch.Tensor,
    *,
    q_lora_rank: int,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> StaticMLAPOWeights:
    """Pack BF16 static W8A8 MLAPO parameters without changing their values.

    Inputs are logical ND weights in [input, output] order. The kernel consumes
    [KV, Q] down-projection channels and deinterleaved RoPE channels. Apply the
    same permutation to the INT8 weights, FP32 dequantization scales and raw
    INT32 accumulator biases; the biases must not be converted to output units.
    The caller owns the final NPU format cast and retains the source parameters
    for native preprocessing.
    """
    if q_lora_rank <= 0 or kv_lora_rank <= 0 or num_heads <= 0:
        raise ValueError("Static MLAPO ranks and num_heads must be positive")
    if qk_nope_head_dim < 0 or qk_rope_head_dim < 0 or qk_rope_head_dim % 2:
        raise ValueError("Static MLAPO head dimensions must be nonnegative and RoPE dimension must be even")
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    if head_dim == 0:
        raise ValueError("Static MLAPO query head dimension must be positive")
    qkv_output_size = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_output_size = num_heads * head_dim
    if qkv_weight.ndim != 2 or qkv_weight.shape[0] == 0 or qkv_weight.shape[1] != qkv_output_size:
        raise ValueError(f"qkv_weight must have shape [input, {qkv_output_size}]")
    if q_weight.shape != (q_lora_rank, q_output_size):
        raise ValueError(f"q_weight must have shape [{q_lora_rank}, {q_output_size}]")
    output_alignment, input_alignment = MLAPO_WEIGHT_BLOCK_SIZE
    if (
        qkv_output_size % output_alignment
        or q_output_size % output_alignment
        or qkv_weight.shape[0] % input_alignment
        or q_lora_rank % input_alignment
    ):
        raise ValueError(
            f"Static MLAPO weight output dimensions must be divisible by {output_alignment} "
            f"and input dimensions by {input_alignment}"
        )
    for name, parameter, output_size, dtype in (
        ("qkv_deq_scale", qkv_deq_scale, qkv_output_size, torch.float32),
        ("qkv_quant_bias", qkv_quant_bias, qkv_output_size, torch.int32),
        ("q_deq_scale", q_deq_scale, q_output_size, torch.float32),
        ("q_quant_bias", q_quant_bias, q_output_size, torch.int32),
    ):
        if parameter.shape != (output_size,):
            raise ValueError(f"{name} must have shape [{output_size}]")
        if parameter.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
    if qkv_weight.dtype != torch.int8 or q_weight.dtype != torch.int8:
        raise ValueError("Static MLAPO weights must have dtype torch.int8")
    if any(
        parameter.device != qkv_weight.device
        for parameter in (qkv_deq_scale, qkv_quant_bias, q_weight, q_deq_scale, q_quant_bias)
    ):
        raise ValueError("Static MLAPO parameters must be on the same device")

    q_down, kv_down = qkv_weight.split((q_lora_rank, kv_lora_rank + qk_rope_head_dim), dim=-1)
    kv_down = trans_rope_weight(kv_down.t(), qk_rope_head_dim)
    wd_qkv = torch.cat((kv_down, q_down.t()), dim=0).contiguous()
    wd_qkv = transdata(wd_qkv, block_size=MLAPO_WEIGHT_BLOCK_SIZE).unsqueeze(0).contiguous()

    q_down_scale, kv_down_scale = qkv_deq_scale.split((q_lora_rank, kv_lora_rank + qk_rope_head_dim))
    kv_down_scale = trans_rope_weight(kv_down_scale.unsqueeze(-1), qk_rope_head_dim).flatten()
    deq_scale_qkv = torch.cat((kv_down_scale, q_down_scale)).contiguous()
    q_down_bias, kv_down_bias = qkv_quant_bias.split((q_lora_rank, kv_lora_rank + qk_rope_head_dim))
    kv_down_bias = trans_rope_weight(kv_down_bias.unsqueeze(-1), qk_rope_head_dim).flatten()
    quant_bias_qkv = torch.cat((kv_down_bias, q_down_bias)).contiguous()

    wu_q = q_weight.t().reshape(num_heads, head_dim, q_lora_rank)
    wu_q = trans_rope_weight(wu_q, qk_rope_head_dim).reshape(q_output_size, q_lora_rank)
    wu_q = transdata(wu_q, block_size=MLAPO_WEIGHT_BLOCK_SIZE).unsqueeze(0).contiguous()
    qb_deq_scl = trans_rope_weight(q_deq_scale.reshape(num_heads, head_dim, 1), qk_rope_head_dim)
    qb_qt_bias = trans_rope_weight(q_quant_bias.reshape(num_heads, head_dim, 1), qk_rope_head_dim)

    return StaticMLAPOWeights(
        wd_qkv,
        deq_scale_qkv,
        quant_bias_qkv,
        wu_q,
        qb_deq_scl.flatten().contiguous(),
        qb_qt_bias.flatten().contiguous(),
    )
