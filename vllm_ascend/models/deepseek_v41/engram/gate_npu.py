# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-stage NPU implementation of the DeepSeek V4.1 Engram gate.

P28 keeps the efficient batched FP32 rotation matmul as a Cube stage and
fuses all following statistics, gate, mask, and residual work into one Triton
Vector kernel.  The FP32 rotated residual is the stage boundary; this avoids
the 160 small Cube/Vector hand-offs of the single MIX kernel on 910_9382.
"""

import torch
import triton.runtime.driver as driver
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs

from .common import engram_gate

ROTATION_BLOCK_SIZE = 32
MIN_GATE_MAGNITUDE: tl.constexpr = 1e-6
ROWS_PER_TILE = 16
FEATURES_PER_STEP = 256


@triton.jit
def _engram_gate_vector_tile(
    hidden_ptr,
    original_ptr,
    key_ptr,
    value_ptr,
    channel_weight_ptr,
    token_mask_ptr,
    output_ptr,
    row,
    NUM_ROWS,
    eps: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    ROWS: tl.constexpr,
    FEATURES: tl.constexpr,
):
    row_valid = row < NUM_ROWS
    token = row // HC_MULT
    hc_index = row - token * HC_MULT
    feature_base = tl.arange(0, FEATURES)
    hidden_rms_sum = tl.zeros((ROWS,), dtype=tl.float32)
    key_rms_sum = tl.zeros((ROWS,), dtype=tl.float32)
    weighted_dot = tl.zeros((ROWS,), dtype=tl.float32)

    for step in range(0, HIDDEN_SIZE, FEATURES):
        feature = step + feature_base
        feature_valid = feature < HIDDEN_SIZE
        mask = row_valid[:, None] & feature_valid[None, :]
        offset = row[:, None] * HIDDEN_SIZE + feature[None, :]
        original = tl.load(original_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        key = tl.load(key_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        weight_offset = hc_index[:, None] * HIDDEN_SIZE + feature[None, :]
        channel_weight = tl.load(channel_weight_ptr + weight_offset, mask=mask, other=0.0).to(tl.float32)
        hidden_rms_sum += tl.sum(original * original, axis=1)
        key_rms_sum += tl.sum(key * key, axis=1)
        weighted_dot += tl.sum(original * channel_weight * key, axis=1)

    rstd = tl.rsqrt(hidden_rms_sum / HIDDEN_SIZE + eps)
    rstd *= tl.rsqrt(key_rms_sum / HIDDEN_SIZE + eps)
    dot = weighted_dot * rstd * HIDDEN_SIZE**-0.5
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), MIN_GATE_MAGNITUDE))
    signed_magnitude = tl.where(dot < 0.0, -magnitude, magnitude)
    gate = 1.0 / (1.0 + tl.exp(-signed_magnitude))
    active = tl.load(token_mask_ptr + token, mask=row_valid, other=False).to(tl.int1)
    gate = tl.where(active, gate, 0.0)

    for step in range(0, HIDDEN_SIZE, FEATURES):
        feature = step + feature_base
        feature_valid = feature < HIDDEN_SIZE
        mask = row_valid[:, None] & feature_valid[None, :]
        offset = row[:, None] * HIDDEN_SIZE + feature[None, :]
        hidden = tl.load(hidden_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        value_offset = token[:, None] * HIDDEN_SIZE + feature[None, :]
        value = tl.load(value_ptr + value_offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offset, hidden + gate[:, None] * value, mask=mask)


@triton.jit
def _engram_gate_vector_kernel(
    hidden_ptr,
    original_ptr,
    key_ptr,
    value_ptr,
    channel_weight_ptr,
    token_mask_ptr,
    output_ptr,
    NUM_ROWS,
    TILES_PER_PROGRAM,
    eps: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    ROWS: tl.constexpr,
    FEATURES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_in_tile = tl.arange(0, ROWS)
    for tile_index in range(0, TILES_PER_PROGRAM):
        row = (pid * TILES_PER_PROGRAM + tile_index) * ROWS + row_in_tile
        _engram_gate_vector_tile(
            hidden_ptr,
            original_ptr,
            key_ptr,
            value_ptr,
            channel_weight_ptr,
            token_mask_ptr,
            output_ptr,
            row,
            NUM_ROWS,
            eps,
            HIDDEN_SIZE,
            HC_MULT,
            ROWS,
            FEATURES,
        )


def npu_engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the P28 Cube-stage plus fused-Vector-stage candidate."""
    hidden_blocks = hidden.float().unflatten(-1, (-1, ROTATION_BLOCK_SIZE))
    original = torch.matmul(hidden_blocks, rotation_block.float().T).flatten(-2)
    output = torch.empty_like(hidden)
    num_rows = hidden.shape[0] * hidden.shape[1]
    num_tiles = triton.cdiv(num_rows, ROWS_PER_TILE)
    device = torch.npu.current_device()
    num_vectorcores = driver.active.utils.get_device_properties(device)["num_vectorcore"]
    grid_size = min(num_tiles, num_vectorcores)
    tiles_per_program = triton.cdiv(num_tiles, grid_size)
    _engram_gate_vector_kernel[(grid_size,)](
        hidden,
        original,
        key,
        value,
        channel_weight,
        token_mask,
        output,
        num_rows,
        tiles_per_program,
        eps,
        HIDDEN_SIZE=hidden.shape[-1],
        HC_MULT=hidden.shape[1],
        ROWS=ROWS_PER_TILE,
        FEATURES=FEATURES_PER_STEP,
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
    if _supports_fused_gate(hidden, key, value, channel_weight, rotation_block, token_mask):
        return torch.ops.vllm.npu_engram_gate(
            hidden,
            key,
            value,
            channel_weight,
            rotation_block,
            token_mask,
            eps,
        )
    return engram_gate(hidden, key, value, channel_weight, rotation_block, token_mask, eps)
