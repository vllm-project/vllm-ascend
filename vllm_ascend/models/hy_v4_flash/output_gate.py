# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

_MX_GROUP_SIZE = 32
_MX_SCALE_PAIR = 2
_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
_E4M3_MAX_EXPONENT = 8


@triton.jit(do_not_specialize=["num_groups"])
def _gate_dynamic_mx_quant(
    value,
    gate_logits,
    quantized,
    scales,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    E4M3_MAX: tl.constexpr,
    E4M3_MAX_EXPONENT: tl.constexpr,
):
    group = tl.program_id(0) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    valid = group < num_groups
    offset = group[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    value_f32 = tl.load(value + offset, valid[:, None], 0).to(tl.float32)
    logits = tl.load(gate_logits + offset, valid[:, None], 0).to(tl.float32)
    # Preserve the two BF16 outputs of the unfused sigmoid and multiplication.
    gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16)
    product = (value_f32 * gate.to(tl.float32)).to(tl.bfloat16)

    bits = product.to(tl.int16, bitcast=True).to(tl.int32)
    exponent = (bits & 0x7F80) >> 7
    group_exponent = tl.max(exponent, axis=1)
    nonfinite = group_exponent == 255
    scale = tl.maximum(group_exponent - E4M3_MAX_EXPONENT, 0)
    reciprocal = ((254 - scale) << 23).to(tl.float32, bitcast=True)
    scaled = product.to(tl.float32) * reciprocal[:, None]
    clipped = tl.minimum(tl.maximum(scaled, -E4M3_MAX), E4M3_MAX)
    fp8 = tl.cast(clipped, tl.float8e4nv, fp_downcast_rounding="rtne")
    # CANN preserves signed zero for tiny groups, and emits all-zero payload
    # with an E8M0 NaN scale when a group contains any nonfinite value.
    signed_zero = (bits >> 8) & 0x80
    codes = tl.where(scale[:, None] == 0, signed_zero, fp8.to(tl.uint8, bitcast=True))
    tl.store(quantized + offset, tl.where(nonfinite[:, None], 0, codes), valid[:, None])
    tl.store(scales + group, tl.where(nonfinite, 255, scale), valid)


def gate_dynamic_mx_quant(value: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return OCP MXFP8 activation bytes for contiguous BF16 attention outputs."""
    rows, width = value.shape
    num_groups = rows * width // _MX_GROUP_SIZE
    groups_per_program = 16 if num_groups <= 512 else 32
    quantized = torch.empty_like(value, dtype=torch.uint8)
    scales = torch.empty(
        (rows, width // (_MX_GROUP_SIZE * _MX_SCALE_PAIR), _MX_SCALE_PAIR), dtype=torch.uint8, device=value.device
    )
    _gate_dynamic_mx_quant[(triton.cdiv(num_groups, groups_per_program),)](
        value,
        gate_logits,
        quantized,
        scales,
        num_groups,
        _MX_GROUP_SIZE,
        groups_per_program,
        _E4M3_MAX,
        _E4M3_MAX_EXPONENT,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return quantized.view(torch.float8_e4m3fn), scales
