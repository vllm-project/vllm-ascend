# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused NPU implementation of the DeepSeek V4.1 Engram gate.

The kernel consumes the WKV split tensors directly. It keeps the rotated
residual, both RMS reductions, and the weighted dot product in FP32 registers,
then writes only the BF16 residual injection result. The WKV GEMM and the
Hyper-Connection operators intentionally remain outside this boundary.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs

from .common import engram_gate

ROTATION_BLOCK_SIZE = 32
# Annotated as constexpr: Triton kernels cannot read plain module globals.
MIN_GATE_MAGNITUDE: tl.constexpr = 1e-6


@triton.jit
def _engram_gate_kernel(
    hidden_ptr,
    key_ptr,
    value_ptr,
    channel_weight_ptr,
    rotation_block_ptr,
    token_mask_ptr,
    output_ptr,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    ROTATION_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC_MULT
    hc_index = row % HC_MULT
    block_offsets = tl.arange(0, ROTATION_BLOCK)
    hidden_rms_sum = 0.0
    key_rms_sum = 0.0
    weighted_dot = 0.0

    # ``rotation_block`` stores R[out, in]. Loading it as [in, out] computes
    # hidden @ R.T, exactly matching the eager reference implementation.
    for block_start in range(0, HIDDEN_SIZE, ROTATION_BLOCK):
        offsets = block_start + block_offsets
        hidden = tl.load(hidden_ptr + row * HIDDEN_SIZE + offsets).to(tl.float32)
        key = tl.load(key_ptr + row * HIDDEN_SIZE + offsets).to(tl.float32)
        channel_weight = tl.load(channel_weight_ptr + hc_index * HIDDEN_SIZE + offsets).to(tl.float32)
        rotation = tl.load(
            rotation_block_ptr + block_offsets[:, None] + block_offsets[None, :] * ROTATION_BLOCK
        ).to(tl.float32)
        original = tl.sum(hidden[:, None] * rotation, axis=0)
        hidden_rms_sum += tl.sum(original * original, axis=0)
        key_rms_sum += tl.sum(key * key, axis=0)
        weighted_dot += tl.sum(original * channel_weight * key, axis=0)

    rstd = tl.rsqrt(hidden_rms_sum / HIDDEN_SIZE + eps)
    rstd *= tl.rsqrt(key_rms_sum / HIDDEN_SIZE + eps)
    dot = weighted_dot * rstd * HIDDEN_SIZE**-0.5
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), MIN_GATE_MAGNITUDE))
    signed_magnitude = tl.where(dot < 0.0, -magnitude, magnitude)
    gate = 1.0 / (1.0 + tl.exp(-signed_magnitude))
    active = tl.load(token_mask_ptr + token).to(tl.int1)
    gate = tl.where(active, gate, 0.0)

    for block_start in range(0, HIDDEN_SIZE, ROTATION_BLOCK):
        offsets = block_start + block_offsets
        hidden = tl.load(hidden_ptr + row * HIDDEN_SIZE + offsets).to(tl.float32)
        value = tl.load(value_ptr + token * HIDDEN_SIZE + offsets).to(tl.float32)
        tl.store(output_ptr + row * HIDDEN_SIZE + offsets, hidden + gate * value)


def npu_engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the single-kernel Engram gate for contiguous supported NPU inputs."""
    output = torch.empty_like(hidden)
    _engram_gate_kernel[(hidden.shape[0] * hidden.shape[1],)](
        hidden,
        key,
        value,
        channel_weight,
        rotation_block,
        token_mask,
        output,
        eps,
        HIDDEN_SIZE=hidden.shape[-1],
        HC_MULT=hidden.shape[1],
        ROTATION_BLOCK=ROTATION_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def npu_engram_gate_fake(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Describe the output during torch.compile and ACL graph capture."""
    del key, value, channel_weight, rotation_block, token_mask, eps
    return torch.empty_like(hidden)


direct_register_custom_op(
    op_name="npu_engram_gate",
    op_func=npu_engram_gate,
    mutates_args=[],
    fake_impl=npu_engram_gate_fake,
    dispatch_key="PrivateUse1",
)


def _supports_fused_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
) -> bool:
    """Return whether inputs satisfy the fixed-layout first-release contract."""
    return (
        envs.VLLM_ASCEND_ENABLE_ENGRAM_GATE_FUSION
        and hidden.device.type == "npu"
        and hidden.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and hidden.ndim == 3
        and key.shape == hidden.shape
        and value.shape == (hidden.shape[0], hidden.shape[-1])
        and channel_weight.shape == hidden.shape[1:]
        and channel_weight.dtype in (torch.bfloat16, torch.float32)
        and rotation_block.shape == (ROTATION_BLOCK_SIZE, ROTATION_BLOCK_SIZE)
        and rotation_block.dtype in (torch.bfloat16, torch.float32)
        and token_mask.shape == (hidden.shape[0],)
        and token_mask.dtype == torch.bool
        and hidden.shape[-1] % ROTATION_BLOCK_SIZE == 0
        and hidden.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and channel_weight.is_contiguous()
        and rotation_block.is_contiguous()
        and token_mask.is_contiguous()
    )


def engram_gate_fused(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Use the fused NPU kernel when enabled, otherwise preserve eager semantics."""
    if _supports_fused_gate(hidden, key, value, channel_weight, rotation_block, token_mask):
        return torch.ops.vllm.npu_engram_gate(
            hidden, key, value, channel_weight, rotation_block, token_mask, eps
        )
    return engram_gate(hidden, key, value, channel_weight, rotation_block, token_mask, eps)
