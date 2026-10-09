# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical regression for ModelSlim M4 compensation in static MLAPO.

The reference uses native static INT8 matmuls and RMSNorm, including its BF16
cast before adding the Q norm bias. The fused kernel adds that bias in FP32.
Sparse, deterministic projections avoid amplification of this rounding
difference, while retaining independent, nonzero affine and zero-point terms.
"""

from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.attention.mla_static import prepare_static_mlapo_weights
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

enable_custom_op()

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Static MLAPO numerical regression requires A2 or A3 hardware.",
)

HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
ROPE_DIM = 64
NOPE_DIM = 128
NUM_HEADS = 2
BLOCK_SIZE = 128
BLOCK_NUM = 2
EPSILON = 1e-6
BF16_ATOL = 1 / 32
BF16_RTOL = 1 / 64
# These fixed channels stay away from static quantization rounding boundaries
# for the deterministic input, despite the native/fused norm cast difference.
Q_INPUT_CHANNELS = (0, 3, 10, 12, 13, 15, 18, 20)


@dataclass
class StaticMLACase:
    hidden: torch.Tensor
    qkv_weight: torch.Tensor
    qkv_scale: torch.Tensor
    qkv_bias: torch.Tensor
    qkv_zero_bias: torch.Tensor
    q_weight: torch.Tensor
    q_scale: torch.Tensor
    q_bias: torch.Tensor
    q_zero_bias: torch.Tensor
    input_scale: torch.Tensor
    input_offset: torch.Tensor
    q_input_scale: torch.Tensor
    q_input_offset: torch.Tensor
    q_gamma: torch.Tensor
    q_beta: torch.Tensor
    kv_gamma: torch.Tensor
    wuk: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def _make_case(token_num: int) -> StaticMLACase:
    """Build static quant_bias = zero-point correction + M4 affine bias."""
    qkv_out = Q_LORA_RANK + KV_LORA_RANK + ROPE_DIM
    columns = torch.arange(qkv_out)
    qkv_weight = torch.zeros((HIDDEN_SIZE, qkv_out), dtype=torch.int8)
    qkv_weight[columns, columns] = (columns % 3 + 1).to(torch.int8)
    qkv_weight[columns + 37, columns] = -(columns % 2 + 1).to(torch.int8)
    qkv_scale = (columns % 4 + 1).float() / 128
    input_offset = -3
    qkv_zero_bias = -input_offset * qkv_weight.sum(dim=0, dtype=torch.int32)
    qkv_affine_bias = ((columns % 7 - 3) * 64 + 32).to(torch.int32)

    q_columns = torch.arange(NUM_HEADS * (NOPE_DIM + ROPE_DIM))
    q_weight = torch.zeros((Q_LORA_RANK, q_columns.numel()), dtype=torch.int8)
    q_inputs = torch.tensor(Q_INPUT_CHANNELS)[q_columns % len(Q_INPUT_CHANNELS)]
    q_weight[q_inputs, q_columns] = ((q_columns % 3 + 1) * 2).to(torch.int8)
    q_scale = (q_columns % 4 + 1).float() / 128
    q_input_offset = 2
    q_zero_bias = -q_input_offset * q_weight.sum(dim=0, dtype=torch.int32)
    q_affine_bias = ((q_columns % 5 - 2) * 64 + 32).to(torch.int32)

    hidden = (torch.arange(token_num * HIDDEN_SIZE).reshape(token_num, HIDDEN_SIZE) % 23) - 11
    hidden = hidden.to(torch.bfloat16) / 16
    q_channels = torch.arange(Q_LORA_RANK)
    q_gamma = (q_channels % 3 + 2).to(torch.bfloat16) / 4
    q_beta = (q_channels % 5 - 2).to(torch.bfloat16) / 4
    kv_gamma = (torch.arange(KV_LORA_RANK) % 3 + 2).to(torch.bfloat16) / 4
    wuk = torch.zeros((NUM_HEADS, NOPE_DIM, KV_LORA_RANK), dtype=torch.bfloat16)
    kv_columns = torch.arange(KV_LORA_RANK)
    for head in range(NUM_HEADS):
        wuk[head, kv_columns % NOPE_DIM, kv_columns] = (kv_columns % 3 + 1).to(torch.bfloat16) / 4
    angles = torch.arange(token_num * (ROPE_DIM // 2)).reshape(token_num, -1).float() / 127
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16)

    return StaticMLACase(
        hidden=hidden.npu(),
        qkv_weight=qkv_weight.npu(),
        qkv_scale=qkv_scale.npu(),
        qkv_bias=(qkv_zero_bias + qkv_affine_bias).npu(),
        qkv_zero_bias=qkv_zero_bias.npu(),
        q_weight=q_weight.npu(),
        q_scale=q_scale.npu(),
        q_bias=(q_zero_bias + q_affine_bias).npu(),
        q_zero_bias=q_zero_bias.npu(),
        input_scale=torch.tensor([0.125], dtype=torch.bfloat16, device="npu"),
        input_offset=torch.tensor([input_offset], dtype=torch.int8, device="npu"),
        q_input_scale=torch.tensor([0.125], dtype=torch.bfloat16, device="npu"),
        q_input_offset=torch.tensor([q_input_offset], dtype=torch.int8, device="npu"),
        q_gamma=q_gamma.npu(),
        q_beta=q_beta.npu(),
        kv_gamma=kv_gamma.npu(),
        wuk=wuk.npu(),
        cos=cos.npu(),
        sin=sin.npu(),
    )


def _quantize(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    # This is the same native primitive used by torch.ops.vllm.quantize.
    reciprocal = scale.reciprocal().expand(x.shape[-1]).contiguous()
    offsets = offset.to(x.dtype).expand(x.shape[-1]).contiguous()
    return torch_npu.npu_quantize(x, reciprocal, offsets, torch.qint8, -1, False)


def _rope(x: torch.Tensor, case: StaticMLACase) -> torch.Tensor:
    token_num = x.shape[0]
    return torch_npu.npu_interleave_rope(
        x.unsqueeze(2),
        case.cos.reshape(token_num, 1, 1, ROPE_DIM),
        case.sin.reshape(token_num, 1, 1, ROPE_DIM),
    ).reshape_as(x)


def _native_reference(
    case: StaticMLACase,
    *,
    keep_down_affine: bool = True,
    keep_q_norm_bias: bool = True,
    keep_up_affine: bool = True,
) -> tuple[torch.Tensor, ...]:
    qkv = torch_npu.npu_quant_matmul(
        _quantize(case.hidden, case.input_scale, case.input_offset),
        case.qkv_weight,
        case.qkv_scale,
        bias=case.qkv_bias if keep_down_affine else case.qkv_zero_bias,
        output_dtype=torch.bfloat16,
    )
    q_c, kv, k_rope = qkv.split((Q_LORA_RANK, KV_LORA_RANK, ROPE_DIM), dim=-1)
    q_c = torch_npu.npu_rms_norm(q_c.contiguous(), case.q_gamma, EPSILON)[0]
    if keep_q_norm_bias:
        q_c.add_(case.q_beta)
    kv = torch_npu.npu_rms_norm(kv.contiguous(), case.kv_gamma, EPSILON)[0]
    query = torch_npu.npu_quant_matmul(
        _quantize(q_c, case.q_input_scale, case.q_input_offset),
        case.q_weight,
        case.q_scale,
        bias=case.q_bias if keep_up_affine else case.q_zero_bias,
        output_dtype=torch.bfloat16,
    ).reshape(-1, NUM_HEADS, NOPE_DIM + ROPE_DIM)
    q_nope, q_rope = query.split((NOPE_DIM, ROPE_DIM), dim=-1)
    q_nope = torch.bmm(q_nope.transpose(0, 1), case.wuk).transpose(0, 1).contiguous()
    return q_nope, _rope(q_rope, case), kv, _rope(k_rope.unsqueeze(1), case).squeeze(1), q_c


def _make_cache(width: int, stride_factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    backing = torch.full(
        (BLOCK_NUM * stride_factor, BLOCK_SIZE, width),
        float("nan"),
        dtype=torch.bfloat16,
        device="npu",
    )
    return backing[::stride_factor], backing


@pytest.mark.parametrize("token_num", [1, 8])
@pytest.mark.parametrize("stride_factor", [1, 2])
@pytest.mark.parametrize("enable_inner_out", [False, True])
@torch.inference_mode()
def test_static_mlapo_preserves_affine_bias(token_num: int, stride_factor: int, enable_inner_out: bool):
    """Check numeric outputs and all cache slots, including padding/stride holes."""
    case = _make_case(token_num)
    weights = prepare_static_mlapo_weights(
        case.qkv_weight,
        case.qkv_scale,
        case.qkv_bias,
        case.q_weight,
        case.q_scale,
        case.q_bias,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        num_heads=NUM_HEADS,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
    )
    weights = weights._replace(
        wd_qkv=torch_npu.npu_format_cast(weights.wd_qkv, ACL_FORMAT_FRACTAL_NZ),
        wu_q=torch_npu.npu_format_cast(weights.wu_q, ACL_FORMAT_FRACTAL_NZ),
    )
    expected = _native_reference(case)
    # Prove that this fixed input detects each missing M4 term independently.
    # Removing only affine compensation preserves the activation zero-point
    # correction, so this cannot pass merely by checking asymmetric quantization.
    down_missing = _native_reference(case, keep_down_affine=False)
    norm_bias_missing = _native_reference(case, keep_q_norm_bias=False)
    up_missing = _native_reference(case, keep_up_affine=False)
    for actual, reference in ((down_missing[4], expected[4]), (norm_bias_missing[4], expected[4])):
        assert (actual - reference).abs().max() > 4 * BF16_ATOL
    assert (up_missing[0] - expected[0]).abs().max() > 4 * BF16_ATOL

    kv_cache, kv_backing = _make_cache(KV_LORA_RANK, stride_factor)
    rope_cache, rope_backing = _make_cache(ROPE_DIM, stride_factor)
    # Alternate blocks to exercise the real first-axis stride. The final row
    # in the multi-token case is graph-style padding and must not write a slot.
    slots = [(row % BLOCK_NUM) * BLOCK_SIZE + 2 * row + 1 for row in range(token_num)]
    if token_num > 1:
        slots[-1] = -1
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device="npu")
    q_nope = torch.full_like(expected[0], float("nan"))
    q_rope = torch.full_like(expected[1], float("nan"))
    q_down = torch.full_like(expected[4], float("nan"))
    unit_scale = torch.ones(1, dtype=torch.bfloat16, device="npu")
    torch.ops._C_ascend.mla_preprocess(
        case.hidden,
        weights.wd_qkv,
        weights.deq_scale_qkv,
        case.q_gamma,
        case.q_beta,
        weights.wu_q,
        weights.qb_deq_scl,
        case.kv_gamma,
        case.cos,
        case.sin,
        case.wuk,
        kv_cache,
        rope_cache,
        slot_mapping,
        quant_scale0=case.input_scale,
        quant_offset0=case.input_offset,
        bias0=weights.quant_bias_qkv,
        quant_scale1=case.q_input_scale,
        quant_offset1=case.q_input_offset,
        bias1=weights.qb_qt_bias,
        ctkv_scale=unit_scale,
        q_nope_scale=unit_scale.expand(NUM_HEADS).contiguous(),
        cache_mode="krope_ctkv",
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=enable_inner_out,
        q_out0=q_nope,
        kv_cache_out0=kv_cache,
        q_out1=q_rope,
        kv_cache_out1=rope_cache,
        inner_out=q_down,
    )
    torch.npu.synchronize()
    for name, actual, reference in (
        ("q_nope", q_nope, expected[0]),
        ("q_rope", q_rope, expected[1]),
    ):
        torch.testing.assert_close(actual, reference, atol=BF16_ATOL, rtol=BF16_RTOL, msg=name)
    if enable_inner_out:
        torch.testing.assert_close(q_down, expected[4], atol=BF16_ATOL, rtol=BF16_RTOL, msg="q_down")

    expected_kv = torch.full_like(kv_cache, float("nan"))
    expected_rope = torch.full_like(rope_cache, float("nan"))
    for row, slot in enumerate(slots):
        if slot >= 0:
            block, offset = divmod(slot, BLOCK_SIZE)
            expected_kv[block, offset] = expected[2][row]
            expected_rope[block, offset] = expected[3][row]
    for name, actual, reference in (
        ("kv_cache", kv_cache, expected_kv),
        ("rope_cache", rope_cache, expected_rope),
    ):
        torch.testing.assert_close(actual, reference, atol=BF16_ATOL, rtol=BF16_RTOL, equal_nan=True, msg=name)
    for backing in (kv_backing, rope_backing):
        for hole in range(1, stride_factor):
            assert torch.isnan(backing[hole::stride_factor]).all(), "MLAPO wrote into a first-axis stride hole"
