#!/usr/bin/python
# ruff: noqa
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None
import ast
import copy
import math
import random

import numpy as np
from qliv2_parameter_normalization import normalize_qliv2_params

try:
    import cann_ops_transformer
except ImportError:
    cann_ops_transformer = None

DISCONTINUOUS_KEYS = True  # key非连续
DEFAULT_SPLIT_S1 = False  # golden切分S1Flag
DEFAULT_S1SIZE = 2048  # s1切分基本块大小

FP32_FRACTION_BITS = 23  # fp32尾数位数

HIF8_EXP_ZERO_THRESHOLD = -23  # 边界值
HIF8_EXP_DML_MIN = -22  # DML最小指数
HIF8_EXP_DML_MAX = -15  # DML最大指数
HIF8_EXP_D0 = 0  # D0指数值
HIF8_EXP_D1_BOUNDARY = 1  # D1指数值
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3  # D2指数范围
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7  # D3指数范围
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15  # D4指数范围

HIF8_DOT_DML = 0  # DML: Denormal Low, 指数范围 -22 ~ -16, 0位尾数
HIF8_DOT_D0 = 1  # D0: 指数为0，3位尾数（最高精度）
HIF8_DOT_D1 = 2  # D1: 指数为±1，3位尾数
HIF8_DOT_D2 = 4  # D2: 指数为±2 ~ ±3，3位尾数
HIF8_DOT_D3 = 8  # D3: 指数为±4 ~ ±7，2位尾数
HIF8_DOT_D4 = 12  # D4: 指数为±8 ~ ±15，1位尾数（最低精度）
HIF8_DOT_INVALID = -1  # 无效状态

HIF8_FRAC_BITS_DML = 0  # DML档位尾数位数
HIF8_FRAC_BITS_D0 = 3  # D0档位尾数位数
HIF8_FRAC_BITS_D1 = 3  # D1档位尾数位数
HIF8_FRAC_BITS_D2 = 3  # D2档位尾数位数
HIF8_FRAC_BITS_D3 = 2  # D3档位尾数位数
HIF8_FRAC_BITS_D4 = 1  # D4档位尾数位数

HIF8_EXP_BITS_DML = 3  # DML档位指数位数
HIF8_EXP_BITS_D0 = 0  # D0档位指数位数
HIF8_EXP_BITS_D1 = 1  # D1档位指数位数
HIF8_EXP_BITS_D2 = 2  # D2档位指数位数
HIF8_EXP_BITS_D3 = 3  # D3档位指数位数
HIF8_EXP_BITS_D4 = 4  # D4档位指数位数

HIF8_ZERO = 0
HIF8_NAN = 128  # 0b10000000, NaN
HIF8_NEG_INF = 239  # 0b11101111, -inf
HIF8_NEG_MAX = 238  # 0b11101110, 负极大值
HIF8_POS_INF = 111  # 0b01101111, +inf
HIF8_POS_MAX = 110  # 0b01101110, 正极大值

HIF8_SIGN_MASK = 128  # 0b10000000, 符号位掩码
HIF8_DOT_MASK = 120  # 0b01110000, dot值掩码
HIF8_FRAC_MASK_3BIT = 7  # 0b00000111, 3位尾数掩码（D0/D1/D2）
HIF8_FRAC_MASK_2BIT = 3  # 0b00000011, 2位尾数掩码（D3）
HIF8_FRAC_MASK_1BIT = 1  # 0b00000001, 1位尾数掩码（D4）
HIF8_EXP_MASK_DML = 7  # 0b00000111, DML指数掩码（bit0-2）
HIF8_EXP_MASK_D4 = 30  # 0b00011110, D4指数掩码（bit1-4）
HIF8_EXP_MASK_D3 = 28  # 0b00011100, D3指数掩码（bit2-4）
HIF8_EXP_MASK_D2 = 24  # 0b00011000, D2指数掩码（bit3-4）
HIF8_EXP_SIGN_MASK_D1 = 8  # 0b00001000, D1指数掩码（bit3）

HIF8_DOT_BIT_SHIFT = 3  # Dot值在HiF8中的起始位置(bit3)
HIF8_DML_EXP_OFFSET = 23  # DML指数偏移值
HIF8_OVERFLOW_SCALE = 1.25  # 溢出阈值缩放因子
HIF8_MAX_FINITE_VALUE = 32768  # 最大有限值（非饱和模式下的边界值, 2^15

SSR_T14_MASK = 16383  # 0b0011 1111 1111 1111, 14位低位掩码
SSR_F14_OFFSET = 8192  # 0b0010 0000 0000 0000, F14偏移值
SSR_DML_SHIFT = 10  # SSR舍入移位值
SSR_RESERVED_BITS = 14  # SSR舍入保留位数
HYBRID_ROUND_EXP_THRESHOLD = 4  # 混合舍入的指数分界点

QUANT_MODE_MXFP8 = 3
QUANT_MODE_HIFLOAT8 = 4
QUANT_MODE_MXFP4 = 5
MX_SCALE_SHAPE_ALIGN = 64
MX_SCALE_PACK_NUM = 2
MX_SCALE_GROUP_SIZE = MX_SCALE_SHAPE_ALIGN // MX_SCALE_PACK_NUM
FP4_PACK_NUM = 2
E8M0_ONE_VALUE = 127
BF16_SIGNIFICAND_BITS = 8
BF16_MIN_NORMAL = 2.0**-126
BF16_MIN_SUBNORMAL = 2.0**-133
FP4_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
MXFP4_TORCH_DTYPE = torch.float4_e2m1fn_x2


def is_mx_quant_mode(quant_mode):
    return quant_mode in (QUANT_MODE_MXFP8, QUANT_MODE_MXFP4)


def is_mxfp4_quant_mode(quant_mode):
    return quant_mode == QUANT_MODE_MXFP4


def round_fp64_to_bf16_rne(value):
    if value.dtype != torch.float64:
        raise TypeError(f"round_fp64_to_bf16_rne expects float64 input, got {value.dtype}")

    mantissa, exponent = torch.frexp(value)
    rounded_normal = torch.ldexp(
        torch.round(mantissa * (1 << BF16_SIGNIFICAND_BITS)),
        exponent - BF16_SIGNIFICAND_BITS,
    )
    rounded_subnormal = torch.round(value / BF16_MIN_SUBNORMAL) * BF16_MIN_SUBNORMAL
    rounded = torch.where(value.abs() < BF16_MIN_NORMAL, rounded_subnormal, rounded_normal)
    rounded = torch.where(torch.isfinite(value), rounded, value)

    # The value is already on the BF16 grid, so both casts below are exact.
    return rounded.to(torch.float32).to(torch.bfloat16)


def reduce_mxfp4_weighted_qk(weight_matrix, qk_matrix):
    output_shape = (weight_matrix.shape[0], weight_matrix.shape[1], qk_matrix.shape[2])
    acc_bf16 = torch.zeros(output_shape, dtype=torch.bfloat16, device=weight_matrix.device)

    for g_idx in range(weight_matrix.shape[2]):
        weight_fp64 = weight_matrix[:, :, g_idx : g_idx + 1].to(torch.float64)
        qk_fp64 = qk_matrix[:, g_idx : g_idx + 1, :].to(torch.float64)
        # Model a BF16-destination FMA with one BF16 rounding per G.
        fma_fp64 = acc_bf16.to(torch.float64) + weight_fp64 * qk_fp64
        acc_bf16 = round_fp64_to_bf16_rne(fma_fp64)
    return acc_bf16


def reduce_mxfp8_weighted_qk(weight_matrix, qk_matrix):
    output_shape = (weight_matrix.shape[0], weight_matrix.shape[1], qk_matrix.shape[2])
    acc_fp32 = torch.zeros(output_shape, dtype=torch.float32, device=weight_matrix.device)

    for g_idx in range(weight_matrix.shape[2]):
        weight_fp64 = weight_matrix[:, :, g_idx : g_idx + 1].to(torch.float64)
        qk_fp64 = qk_matrix[:, g_idx : g_idx + 1, :].to(torch.float64)
        # Model an FP32-destination MulAddDst: compute the fused multiply-add in
        # FP64, then round once to the FP32 destination after every G.
        fma_fp64 = acc_fp32.to(torch.float64) + weight_fp64 * qk_fp64
        acc_fp32 = fma_fp64.to(torch.float32)
    return acc_fp32


def get_qk_physical_head_dim(head_dim, quant_mode):
    return head_dim // FP4_PACK_NUM if is_mxfp4_quant_mode(quant_mode) else head_dim


def e8m0_raw_to_float(raw):
    raw_float = raw.to(torch.float32)
    scale = torch.pow(2.0, raw_float - E8M0_ONE_VALUE)
    scale = torch.where(raw == 255, torch.full_like(scale, float("nan")), scale)
    return scale


def validate_mx_scale_dtype(scale_dtype):
    if scale_dtype != torch.float8_e8m0fnu:
        raise TypeError(f"MXFP8/MXFP4 require dequant_dtype=torch.float8_e8m0fnu, but got {scale_dtype}")


def make_e8m0_tensor_from_raw(raw, scale_dtype, to_npu=True):
    validate_mx_scale_dtype(scale_dtype)
    scale = raw.contiguous().view(scale_dtype)
    return scale.npu() if to_npu else scale


def make_mx_e8m0_scale_pair(base_shape, tail_shape, scale_range, scale_dtype, to_npu=True):
    validate_mx_scale_dtype(scale_dtype)
    range_min = float(scale_range[0])
    range_max = float(scale_range[1])
    if not math.isfinite(range_min) or not math.isfinite(range_max):
        raise ValueError(f"E8M0 scale range must be finite, got {scale_range}")
    if range_min <= 0 or range_min > range_max:
        raise ValueError(f"E8M0 scale range must satisfy 0 < min <= max, got {scale_range}")

    # datarange表示真实scale值；在生成器内部转换为E8M0编码（编码e对应2^(e-127)）。
    range_min_code = max(0, math.ceil(math.log2(range_min)) + E8M0_ONE_VALUE)
    range_max_code = min(254, math.floor(math.log2(range_max)) + E8M0_ONE_VALUE)
    while range_min_code <= 254 and math.ldexp(1.0, range_min_code - E8M0_ONE_VALUE) < range_min:
        range_min_code += 1
    while range_max_code >= 0 and math.ldexp(1.0, range_max_code - E8M0_ONE_VALUE) > range_max:
        range_max_code -= 1
    if range_min_code > range_max_code:
        raise ValueError(f"E8M0 scale range contains no representable value: {scale_range}")

    # 沿连续存储顺序循环取值，保证不同token和不同D group使用可区分的scale。
    scale_shape = tuple(base_shape) + tuple(tail_shape)
    raw_count = range_max_code - range_min_code + 1
    scale_raw = torch.arange(math.prod(scale_shape), dtype=torch.int64)
    scale_raw = (scale_raw % raw_count + range_min_code).reshape(scale_shape).to(torch.uint8)

    cpu_scale = e8m0_raw_to_float(scale_raw)
    return make_e8m0_tensor_from_raw(scale_raw, scale_dtype, to_npu), cpu_scale


def validate_mxfp4_dtype(data_dtype):
    if data_dtype != MXFP4_TORCH_DTYPE:
        raise TypeError(f"MXFP4 requires qk_dtype=torch.float4_e2m1fn_x2, but got {data_dtype}")


def make_mxfp4_tensor_pair(logical_shape, data_range, data_dtype, to_npu=True):
    validate_mxfp4_dtype(data_dtype)
    range_min = float(data_range[0])
    range_max = float(data_range[1])
    if not math.isfinite(range_min) or not math.isfinite(range_max) or range_min > range_max:
        raise ValueError(f"invalid FP4 E2M1 numeric range: {data_range}")
    if logical_shape[-1] % FP4_PACK_NUM != 0:
        raise ValueError(f"MXFP4 head_dim must be even, got {logical_shape[-1]}")

    valid_codes = (
        torch.nonzero(
            (range_min <= FP4_E2M1_VALUES) & (range_max >= FP4_E2M1_VALUES),
            as_tuple=False,
        )
        .flatten()
        .to(torch.uint8)
    )
    if valid_codes.numel() == 0:
        raise ValueError(f"FP4 E2M1 range contains no representable value: {data_range}")

    raw_indices = torch.randint(valid_codes.numel(), tuple(logical_shape), dtype=torch.long)
    raw = valid_codes[raw_indices]
    raw_rows = raw.reshape(-1, logical_shape[-1])
    coverage_count = min(valid_codes.numel(), logical_shape[-1])
    coverage = torch.arange(coverage_count).unsqueeze(0) + torch.arange(raw_rows.shape[0]).unsqueeze(1)
    raw_rows[:, :coverage_count] = valid_codes[coverage % valid_codes.numel()]
    packed = raw[..., 0::2] | (raw[..., 1::2] << 4)
    packed_mxfp4 = packed.contiguous().view(data_dtype)
    cpu_ref = FP4_E2M1_VALUES[raw.to(torch.long)]
    return (packed_mxfp4.npu() if to_npu else packed_mxfp4), cpu_ref


def make_e8m0_zero(shape, scale_dtype):
    validate_mx_scale_dtype(scale_dtype)
    return torch.zeros(shape, dtype=torch.uint8).view(scale_dtype)


class GeneralizedQLIV2:
    def __init__(
        self,
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        dequant_dtype,
        actual_seq_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        act_seq_q,
        act_seq_k,
        cmp_residual_k,
        max_seqlen_q,
        quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
        cmp_ratio,
        return_value,
        split_s1=DEFAULT_SPLIT_S1,
        s1size=DEFAULT_S1SIZE,
    ):
        self.batch_size = batch_size
        self.q_seq = q_seq
        self.k_seq = k_seq
        self.q_t_size = q_t_size
        self.k_t_size = k_t_size
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        self.group_size = q_head_num // k_head_num
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_num = block_num
        self.qk_dtype = qk_dtype
        self.weight_dtype = weight_dtype
        self.dequant_dtype = dequant_dtype
        self.actual_seq_dtype = actual_seq_dtype
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.seqused_q = act_seq_q
        self.seqused_k = act_seq_k
        self.act_seq_q = act_seq_q
        self.act_seq_k = act_seq_k
        self.cmp_residual_k = cmp_residual_k
        self.max_seqlen_q = max_seqlen_q
        self.quant_mode = quant_mode
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode
        self.cmp_ratio = cmp_ratio
        self.w_dtype = weight_dtype
        self.return_value = return_value
        self.split_s1 = split_s1  # 是否切分S1轴 / Whether to split the S1 axis
        self.s1size = s1size  # S1轴切分块大小 / S1 axis chunk size

        if layout_query == "BSND":
            self.q_shape = [batch_size, q_seq, q_head_num, head_dim]
            self.w_shape = [batch_size, q_seq, q_head_num]
            self.q_tnd_flag = 0
        elif layout_query == "TND":
            self.q_shape = [q_t_size, q_head_num, head_dim]
            self.w_shape = [q_t_size, q_head_num]
            self.q_tnd_flag = 1

        if layout_key == "BSND":
            self.k_shape = [batch_size, k_seq, k_head_num, head_dim]
        elif layout_key == "TND":
            self.k_shape = [k_t_size, k_head_num, head_dim]

        if layout_query == "BSND":
            self.out_shape = [batch_size, q_seq, k_head_num, sparse_count]
            self.output_idx_offset_shape = [batch_size, q_seq, k_head_num]
        elif layout_query == "TND":
            self.out_shape = [q_t_size, k_head_num, sparse_count]
            self.output_idx_offset_shape = [q_t_size, k_head_num]

    def cal_atten_bnsd(self, output_idx_offset):
        batch_size = self.batch_size
        qs = self.q_seq
        ks = self.k_seq
        n1 = self.q_head_num
        n2 = self.k_head_num
        cu_seqlens_q = self.cu_seqlens_q
        cu_seqlens_k = self.cu_seqlens_k
        seqused_q = self.seqused_q
        seqused_k = self.seqused_k
        cmp_residual_k = self.cmp_residual_k
        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        wt_bnsd_tensor = self.wt_bnsd_tensor
        mask_tensor = self.m_tensor
        q_scale_bnsd_tensor = self.q_scale_bnsd_tensor
        k_scale_bnsd_tensor = self.k_scale_bnsd_tensor
        cmp_ratio = self.cmp_ratio

        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.sparse_count

        out_shape_bnss = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnss[1] = n2
        # out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k))
        out_shape_bnss[-1] = math.floor(max(seqused_k)) if seqused_k is not None else ks

        y = torch.full(out_shape_bnsd, -1, dtype=torch.int32)
        y_value = torch.full(out_shape_bnss, -float("inf"), dtype=torch.float32)
        y_value_np = np.full(out_shape_bnsd, -np.inf, dtype=np.float32)

        prefix = 0
        for b_idx in range(batch_size):
            if self.layout_query == "TND":
                if seqused_q is not None:
                    curr_actualSeq_q = seqused_q[b_idx]
                else:
                    # 已被处理为shape为(B,)的tensor
                    curr_actualSeq_q = cu_seqlens_q[b_idx]
            elif self.layout_query == "BSND":
                if seqused_q is not None:
                    curr_actualSeq_q = seqused_q[b_idx]
                else:
                    curr_actualSeq_q = qs

            if self.layout_key == "TND":
                if seqused_k is not None:
                    curr_actualSeq_k = seqused_k[b_idx]
                else:
                    curr_actualSeq_k = cu_seqlens_k[b_idx]
            elif self.layout_key == "PA_BBND":
                curr_actualSeq_k = seqused_k[b_idx]
            elif self.layout_key == "BSND":
                if seqused_k is not None:
                    curr_actualSeq_k = seqused_k[b_idx]
                else:
                    curr_actualSeq_k = ks
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k

            self.cur_b_idx = b_idx

            if self.split_s1:
                # 切分S1轴以减小中间结果内存占用
                # Split S1 axis to reduce intermediate result memory usage
                num_s1_chunks = math.ceil(curr_actualSeq_q / self.s1size) if curr_actualSeq_q > 0 else 1
                for s1_chunk_idx in range(num_s1_chunks):
                    s1_start = s1_chunk_idx * self.s1size
                    s1_end = min(s1_start + self.s1size, curr_actualSeq_q)
                    cur_chunk_s1 = s1_end - s1_start

                    self.cur_q = q_bnsd_tensor[b_idx : (b_idx + 1), :, s1_start:s1_end, :]
                    self.cur_k = k_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_k, :]
                    self.cur_wt = wt_bnsd_tensor[b_idx : (b_idx + 1), :, s1_start:s1_end, :]
                    self.cur_q_scale = q_scale_bnsd_tensor[b_idx : (b_idx + 1), :, s1_start:s1_end, :]
                    self.cur_k_scale = k_scale_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_k]
                    if self.sparse_mode != 0:
                        self.cur_m = mask_tensor[b_idx : (b_idx + 1), s1_start:s1_end, :curr_actualSeq_k]

                    if cur_chunk_s1 != 0:
                        actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                        if is_mx_quant_mode(self.quant_mode):
                            (
                                y[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :actual_selected_count,
                                ],
                                y_value[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :curr_actualSeq_k,
                                ],
                            ) = self.cal_atten_per_batch_mx(b_idx)
                        elif self.qk_dtype == torch.int8:
                            (
                                y[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :actual_selected_count,
                                ],
                                y_value[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :curr_actualSeq_k,
                                ],
                            ) = self.cal_atten_per_batch_int8(b_idx)
                        elif self.qk_dtype == torch.float8_e4m3fn:
                            (
                                y[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :actual_selected_count,
                                ],
                                y_value[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :curr_actualSeq_k,
                                ],
                            ) = self.cal_atten_per_batch_fp8(b_idx)
                        elif self.qk_dtype == torch.uint8:
                            (
                                y[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :actual_selected_count,
                                ],
                                y_value[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :curr_actualSeq_k,
                                ],
                            ) = self.cal_atten_per_batch_hifp8(b_idx)
                        if output_idx_offset is not None:
                            if self.layout_query == "TND":
                                offset = output_idx_offset.flatten()[prefix + s1_start : prefix + s1_end].reshape(
                                    1, -1, 1
                                )
                            else:
                                offset = output_idx_offset.flatten()[
                                    b_idx * qs + s1_start : b_idx * qs + s1_end
                                ].reshape(1, -1, 1)
                            offset_mask = (
                                y[
                                    b_idx : (b_idx + 1),
                                    :,
                                    s1_start:s1_end,
                                    :actual_selected_count,
                                ]
                                != -1
                            )
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                s1_start:s1_end,
                                :actual_selected_count,
                            ] += offset * offset_mask
                        y_value_np[
                            b_idx : (b_idx + 1),
                            :,
                            s1_start:s1_end,
                            :actual_selected_count,
                        ] = -np.sort(-y_value.numpy())[
                            b_idx : (b_idx + 1),
                            :,
                            s1_start:s1_end,
                            :actual_selected_count,
                        ]
                y[
                    b_idx : (b_idx + 1),
                    :,
                    curr_actualSeq_q:,
                    : min(curr_actualSeq_k, self.sparse_count),
                ] = -1
            else:
                self.cur_q = q_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_q, :]
                self.cur_k = k_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_k, :]
                self.cur_wt = wt_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_q, :]
                self.cur_q_scale = q_scale_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_q, :]
                self.cur_k_scale = k_scale_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_k]
                if self.sparse_mode != 0:
                    self.cur_m = mask_tensor[b_idx : (b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k]

                if curr_actualSeq_q != 0:
                    actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                    if is_mx_quant_mode(self.quant_mode):
                        (
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :actual_selected_count,
                            ],
                            y_value[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :curr_actualSeq_k,
                            ],
                        ) = self.cal_atten_per_batch_mx(b_idx)
                    elif self.qk_dtype == torch.int8:
                        (
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :actual_selected_count,
                            ],
                            y_value[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :curr_actualSeq_k,
                            ],
                        ) = self.cal_atten_per_batch_int8(b_idx)
                    elif self.qk_dtype == torch.float8_e4m3fn:
                        (
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :actual_selected_count,
                            ],
                            y_value[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :curr_actualSeq_k,
                            ],
                        ) = self.cal_atten_per_batch_fp8(b_idx)
                    elif self.qk_dtype == torch.uint8:
                        (
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :actual_selected_count,
                            ],
                            y_value[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :curr_actualSeq_k,
                            ],
                        ) = self.cal_atten_per_batch_hifp8(b_idx)
                    y[
                        b_idx : (b_idx + 1),
                        :,
                        curr_actualSeq_q:,
                        :actual_selected_count,
                    ] = -1
                    if output_idx_offset is not None:
                        if self.layout_query == "TND":
                            offset = output_idx_offset.flatten()[prefix : prefix + curr_actualSeq_q].reshape(1, -1, 1)
                        else:
                            offset = output_idx_offset.flatten()[b_idx * qs : b_idx * qs + curr_actualSeq_q].reshape(
                                1, -1, 1
                            )
                        offset_mask = (
                            y[
                                b_idx : (b_idx + 1),
                                :,
                                :curr_actualSeq_q,
                                :actual_selected_count,
                            ]
                            != -1
                        )
                        y[
                            b_idx : (b_idx + 1),
                            :,
                            :curr_actualSeq_q,
                            :actual_selected_count,
                        ] += offset * offset_mask
                    y_value_np[
                        b_idx : (b_idx + 1),
                        :,
                        :curr_actualSeq_q,
                        :actual_selected_count,
                    ] = -np.sort(-y_value.numpy())[
                        b_idx : (b_idx + 1),
                        :,
                        :curr_actualSeq_q,
                        :actual_selected_count,
                    ]
                else:
                    pass
            if self.layout_query == "TND":
                prefix += cu_seqlens_q[b_idx]
        return y, y_value, y_value_np

    def trans_shape_to_bnsd(
        self,
        tensor,
        shape,
        layout,
        headnums=None,
        act_seq=None,
        is_weights=False,
        tensor_name=None,
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout == "BSN":
            print("shape", shape)
            B = shape[0]
            S = shape[1]
            N = shape[2]
            if is_weights:
                D = 1
                tensor = torch.unsqueeze(tensor, dim=-1)  # 补D轴
                tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
                return tensor, [B, N, S, D]
            else:
                tensor = tensor.reshape(B, S, N).permute(0, 2, 1)
                return tensor, [B, N, S]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[2]
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
                t_start += act_s
            return new_tensor, [B, N, S, D]
        elif layout == "TN":
            T = shape[0]
            N = shape[1]
            D = 1
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s] = tensor[t_start:t_end, n_index]
                t_start += act_s
            return new_tensor, [B, N, S]
        else:
            return tensor, shape

    def trans_tnd_actseq(self, list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError("TND情况下 act_seq需要必传")
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f"TND情况下 act_seq_len 为非递减数列 act_seq_len={list}")
        return list_new

    def cal_atten_per_batch_mx(self, b_idx):
        cur_q = self.cur_q.to(dtype=torch.float32)
        cur_k = self.cur_k.to(dtype=torch.float32)
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        if is_mxfp4_quant_mode(self.quant_mode):
            head_dim = cur_q.shape[-1]
            if head_dim % MX_SCALE_GROUP_SIZE != 0:
                raise ValueError(f"MXFP4 head_dim must be divisible by {MX_SCALE_GROUP_SIZE}, but got {head_dim}")

            cur_q = cur_q * cur_q_scale
            cur_k = cur_k * cur_k_scale
            qk_bmm_res = torch.bmm(cur_q.squeeze(0), cur_k.permute(0, 1, 3, 2).squeeze(0)).unsqueeze(0)

            qk_relu_out = qk_bmm_res.clamp_min(0.0).to(torch.bfloat16)
            weight_matrix = cur_wt.to(torch.bfloat16).permute(0, 2, 3, 1).squeeze(0)
            qk_matrix = qk_relu_out.permute(0, 2, 1, 3).squeeze(0)
            brc_vmul_matrix = reduce_mxfp4_weighted_qk(weight_matrix, qk_matrix)
            brc_vmul = brc_vmul_matrix.unsqueeze(0)
        else:
            cur_q = cur_q * cur_q_scale
            cur_k = cur_k * cur_k_scale
            qk_bmm_res = torch.bmm(cur_q.squeeze(0), cur_k.permute(0, 1, 3, 2).squeeze(0)).unsqueeze(0)
            qk_relu_out = qk_bmm_res.to(dtype=torch.float32).clamp_min(0.0)
            weight_matrix = cur_wt.permute(0, 2, 3, 1).squeeze(0)
            qk_matrix = qk_relu_out.permute(0, 2, 1, 3).squeeze(0)
            brc_vmul = reduce_mxfp8_weighted_qk(weight_matrix, qk_matrix).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_n2 = self.k_head_num
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone().to(torch.bfloat16)
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype=torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_hifp8(self, b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        cur_q = trans_hifuint8_tensor_to_float(cur_q)
        cur_k = trans_hifuint8_tensor_to_float(cur_k)
        qk_bmm_res = torch.bmm(cur_q.squeeze(0), cur_k.permute(0, 1, 3, 2).squeeze(0)).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)
        brc_vmul = torch.bmm(
            cur_w.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0),
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16)
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype=torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device=row_ele.device)

                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_fp8(self, b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype=torch.float32).squeeze(0),
            cur_k.to(dtype=torch.float32).permute(0, 1, 3, 2).squeeze(0),
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)
        brc_vmul = torch.bmm(
            cur_w.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0),
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16)
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype=torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device=row_ele.device)

                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_int8(self, b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float16)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float16)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float16)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype=torch.int32).squeeze(0),
            cur_k.to(dtype=torch.int32).permute(0, 1, 3, 2).squeeze(0),
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
        brc_vmul = torch.bmm(
            cur_w.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0),
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf

        to_be_sort_ele = reduce_sum.clone()
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype=torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device=row_ele.device)

                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(torch.stack([-row_ele, indices], dim=1), dim=0, stable=True)
                b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def trans_bnsd_to_layout(self, tensor, shape, layout, act_q=None):
        # 此时的输出D轴是K轴
        if layout == "BSH":
            output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
            return output
        elif layout == "BSND":
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
            output = tensor.permute(1, 0, 2, 3).contiguous()
            return output
        elif layout in ["TND", "TND_NTD"]:
            T = sum(act_q)
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.full(size=(T, N, D), fill_value=-1, dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
                t_start += act_s
            if layout == "TND_NTD":
                output = output.permute(1, 0, 2).contiguous()
            return output
        else:
            return tensor

    def broadcast_n_axis(self, n1, n2, temp_tensor, input_dtype):
        g = n1 // n2
        temp_shape = temp_tensor.shape
        B = temp_shape[0]
        S = temp_shape[2]
        D = temp_shape[3]
        modify_tensor = torch.zeros([B, n1, S, D], dtype=temp_tensor.dtype)
        for i in range(n1):
            j = i // g
            modify_tensor[:, i : i + 1, :, :] = temp_tensor[:, j : j + 1, :, :]
        return modify_tensor, modify_tensor.shape

    def flatten_mx_scale_tail(self, tensor, shape):
        scale_head_dim = shape[-2] * shape[-1]
        flat_shape = list(shape[:-2]) + [scale_head_dim]
        return tensor.reshape(flat_shape), flat_shape

    def broadcast_mx_scale_d_axis(self, tensor):
        output = tensor.repeat_interleave(MX_SCALE_GROUP_SIZE, dim=-1)
        return output, list(output.shape)

    def create_mask(self, m_shape, act_k, S1):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        cmp_ratio = self.cmp_ratio
        tmp_pos_orig = act_k - S1

        for i in range(S1):
            if ((tmp_pos_orig + i + 1) / cmp_ratio) < 0:
                atten_masks[i, :] = 1
            else:
                atten_masks[i, math.floor((tmp_pos_orig + i + 1) / cmp_ratio) :] = 1
        return atten_masks

    def create_mask_right_down(self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, batch):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
        cmp_ratio = self.cmp_ratio
        cmp_residual_k = self.cmp_residual_k
        next_tokens_list = []
        re_mask_batch = []
        pre_tokens = 214748647
        for i in range(batch):
            if len(actualSeqLengthsQ) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengthsQ[i]

            if len(actualSeqLengthsK) == 0:
                S2 = mask_s_kv
            else:
                S2 = math.floor(actualSeqLengthsK[i])
            next_tokens = S2 - S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i] * cmp_ratio + cmp_residual_k[i]
            atten_masks = self.create_mask(m_shape, act_k, S1)
            re_mask_batch.append(np.array(atten_masks, dtype=np.bool_))
        re_mask_np = np.array(re_mask_batch, dtype=np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list

    def forward(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        block_table,
        output_idx_offset,
    ):
        print("cpu执行中...")

        # 参数的初始化
        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        out_shape = self.out_shape
        q_shape = self.q_shape
        head_dim = self.head_dim
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        q_t_size = self.q_t_size
        k_t_size = self.k_t_size
        block_size = self.block_size
        block_num = self.block_num
        q_dtype = self.qk_dtype
        k_dtype = self.qk_dtype
        w_shape = self.w_shape
        w_dtype = self.w_dtype
        actual_seq_dtype = self.actual_seq_dtype
        cmp_ratio = self.cmp_ratio
        return_value = self.return_value
        is_mx_mode = is_mx_quant_mode(self.quant_mode)

        if layout_query == "TND":
            q_scale_shape = [q_t_size, q_head_num]
            self.cu_seqlens_q = self.trans_tnd_actseq(cu_seqlens_q[1:])
            actualSeqLengths_q = self.cu_seqlens_q
            if seqused_q is not None:
                self.seqused_q = seqused_q
                self.has_seqused_q = True
                actualSeqLengths_q = self.seqused_q
        elif layout_query == "BSND":
            q_scale_shape = [batch_size, q_seq, q_head_num]
            if seqused_q is not None:
                self.seqused_q = seqused_q
                actual_seq_lengths_query = seqused_q
                self.has_seqused_q = True
            else:
                actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(torch.int32)
            actualSeqLengths_q = actual_seq_lengths_query

        if layout_key == "TND":
            layout_key_scale = "TN"
            k_scale_shape = [k_t_size, k_head_num]
            self.cu_seqlens_k = self.trans_tnd_actseq(cu_seqlens_k[1:])
            actualSeqLengths_k = self.cu_seqlens_k
            k_shape = self.k_shape
            if seqused_k is not None:
                self.seqused_k = seqused_k
                self.has_seqused_k = True
                actualSeqLengths_k = self.seqused_k
        elif layout_key == "BSND":
            layout_key_scale = "BSN"
            k_shape = self.k_shape
            k_scale_shape = [batch_size, k_seq, k_head_num]
            if seqused_k is not None:
                self.seqused_k = seqused_k
                actual_seq_lengths_key = seqused_k
                self.has_seqused_k = True
            else:
                actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq, k_seq, batch_size)).to(torch.int32)
            actualSeqLengths_k = actual_seq_lengths_key

        elif layout_key == "PA_BBND":
            self.actual_seq_lengths_key = seqused_k
            actualSeqLengths_k = self.actual_seq_lengths_key
            layout_key_scale = layout_key
            k_max_s2 = math.floor(max(actualSeqLengths_k))
            k_shape = [batch_size, k_head_num, k_max_s2, head_dim]
            k_scale_shape = [batch_size, k_head_num, k_max_s2]
        if is_mx_mode:
            mx_scale_tail_shape = [head_dim // MX_SCALE_SHAPE_ALIGN, MX_SCALE_PACK_NUM]
            q_scale_shape = q_scale_shape + mx_scale_tail_shape
            k_scale_shape = k_scale_shape + mx_scale_tail_shape
            if layout_key in ["BSND", "TND"]:
                layout_key_scale = layout_key
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()
        query_dequant_scale = query_dequant_scale.cpu()
        key_dequant_scale = key_dequant_scale.cpu()
        q_scale_is_weights = True
        if is_mx_mode:
            query_dequant_scale, q_scale_shape = self.flatten_mx_scale_tail(query_dequant_scale, q_scale_shape)
            key_dequant_scale, k_scale_shape = self.flatten_mx_scale_tail(key_dequant_scale, k_scale_shape)
            q_scale_is_weights = False
        if output_idx_offset is not None:
            output_idx_offset = output_idx_offset.cpu()

        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        if self.layout_query == "TND":
            q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(
                query, q_shape, layout_query, q_head_num, self.cu_seqlens_q
            )
        else:
            q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(
                query, q_shape, layout_query, q_head_num, actualSeqLengths_q
            )

        ## BSND/TND/ -> BNSD
        if self.layout_key == "TND":
            k_bnsd_tensor, k_bnsd_shape = self.trans_shape_to_bnsd(
                key, k_shape, layout_key, k_head_num, self.cu_seqlens_k
            )
            k_scale_bnsd_tensor, k_scale_bnsd_shape = self.trans_shape_to_bnsd(
                key_dequant_scale,
                k_scale_shape,
                layout_key_scale,
                k_head_num,
                self.cu_seqlens_k,
            )
        else:
            k_bnsd_tensor, k_bnsd_shape = self.trans_shape_to_bnsd(
                key,
                k_shape,
                layout_key,
                k_head_num,
                torch.floor(actualSeqLengths_k).to(actual_seq_dtype),
            )
            k_scale_bnsd_tensor, k_scale_bnsd_shape = self.trans_shape_to_bnsd(
                key_dequant_scale,
                k_scale_shape,
                layout_key_scale,
                k_head_num,
                torch.floor(actualSeqLengths_k).to(actual_seq_dtype),
            )

        ## BSN1 -> BNS1   TN1 -> BNS1
        is_weights = True
        if self.layout_query == "TND":
            wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(
                weights,
                w_shape,
                layout_query,
                q_head_num,
                self.cu_seqlens_q,
                is_weights,
            )
            q_scale_bnsd_tensor, q_scale_bnsd_shape = self.trans_shape_to_bnsd(
                query_dequant_scale,
                q_scale_shape,
                layout_query,
                q_head_num,
                self.cu_seqlens_q,
                q_scale_is_weights,
            )
        else:
            wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(
                weights,
                w_shape,
                layout_query,
                q_head_num,
                actualSeqLengths_q,
                is_weights,
            )
            # BSN1 -> BNS1
            q_scale_bnsd_tensor, q_scale_bnsd_shape = self.trans_shape_to_bnsd(
                query_dequant_scale,
                q_scale_shape,
                layout_query,
                q_head_num,
                actualSeqLengths_q,
                q_scale_is_weights,
            )
        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor, k_bnsd_shape = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor, k_dtype)
            if is_mx_mode:
                k_scale_bnsd_tensor, k_scale_bnsd_shape = self.broadcast_n_axis(
                    q_head_num,
                    k_head_num,
                    k_scale_bnsd_tensor,
                    k_scale_bnsd_tensor.dtype,
                )
        if is_mx_mode:
            q_scale_bnsd_tensor, q_scale_bnsd_shape = self.broadcast_mx_scale_d_axis(q_scale_bnsd_tensor)
            k_scale_bnsd_tensor, k_scale_bnsd_shape = self.broadcast_mx_scale_d_axis(k_scale_bnsd_tensor)
        self.q_bnsd_tensor = q_bnsd_tensor
        self.q_bnsd_shape = q_bnsd_shape
        self.k_bnsd_tensor = k_bnsd_tensor
        self.k_bnsd_shape = k_bnsd_shape
        self.wt_bnsd_tensor = wt_bnsd_tensor
        self.wt_bnsd_shape = wt_bnsd_shape
        self.q_scale_bnsd_tensor = q_scale_bnsd_tensor
        self.q_scale_bnsd_shape = q_scale_bnsd_shape
        self.k_scale_bnsd_tensor = k_scale_bnsd_tensor
        self.k_scale_bnsd_shape = k_scale_bnsd_shape
        # 生成mask, sparse_mode=3时使能
        m_shape_std = [q_bnsd_shape[2], k_bnsd_shape[2]]  # m_shape应该是[s1,s2]
        batch = q_bnsd_shape[0]
        m_tensor = []
        if sparse_mode == 3:
            m_tensor, next_tokens_list = self.create_mask_right_down(
                m_shape_std, actualSeqLengths_q, actualSeqLengths_k, batch
            )
        elif sparse_mode == 0:
            pass
        else:
            raise ValueError("unsupported sparse_mode!")
        self.m_tensor = m_tensor
        y, y_value, y_value_np = self.cal_atten_bnsd(output_idx_offset)
        sparse_value = torch.from_numpy(y_value_np)

        # TND & PA 需要传入out_shape为BNSD
        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = k_head_num
        out_shape_bnsd[-1] = sparse_count
        if self.layout_query == "TND":
            y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, self.cu_seqlens_q)  # TODO
            if return_value:
                sparse_value = self.trans_bnsd_to_layout(sparse_value, out_shape_bnsd, layout_query, self.cu_seqlens_q)
        else:
            y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, q_seq)
            if return_value:
                sparse_value = self.trans_bnsd_to_layout(sparse_value, out_shape_bnsd, layout_query, q_seq)
        return y, y_value, sparse_value


def trans_prefix_actseq(self, list):
    list_len = len(list)
    if list_len == 0:
        raise ValueError("PA场景下 act_seq需要必传")
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            raise ValueError(f"PA场景下act seq 为非递减数列 act_seq ={list}")
    return list_new


def qliv2_output_single(
    params,
    is_batch=False,
    split_s1=DEFAULT_SPLIT_S1,
    s1size=DEFAULT_S1SIZE,
    generate_golden=True,
):
    if is_batch:
        params = normalize_qliv2_params(params)
    (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        dequant_dtype,
        actual_seq_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        max_seqlen_q,
        quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
        cmp_ratio,
        return_value,
        output_idx_offset,
    ) = params

    if is_batch:
        dtype_map = {
            "INT8": torch.int8,
            "UINT8": torch.uint8,
            "HIFLOAT8": torch.uint8,
            "INT32": torch.int32,
            "INT64": torch.int64,
            "FP16": torch.float16,
            "FLOAT16": torch.float16,
            "FP32": torch.float32,
            "FLOAT": torch.float32,
            "FLOAT32": torch.float32,
            "BF16": torch.bfloat16,
            "FLOAT8_E4M3FN": torch.float8_e4m3fn,
            "FLOAT8_E8M0": torch.float8_e8m0fnu,
            "FLOAT8_E8M0FNU": torch.float8_e8m0fnu,
            "FLOAT4_E2M1": MXFP4_TORCH_DTYPE,
            "FLOAT4_E2M1FN_X2": MXFP4_TORCH_DTYPE,
        }
        qk_dtype = dtype_map.get(qk_dtype, qk_dtype)
        weight_dtype = dtype_map.get(weight_dtype, weight_dtype)
        dequant_dtype = dtype_map.get(dequant_dtype, dequant_dtype)
        actual_seq_dtype = dtype_map.get(actual_seq_dtype, actual_seq_dtype)

        if cu_seqlens_q is not None and isinstance(cu_seqlens_q, str):
            cu_seqlens_q = ast.literal_eval(cu_seqlens_q)
        if cu_seqlens_k is not None and isinstance(cu_seqlens_k, str):
            cu_seqlens_k = ast.literal_eval(cu_seqlens_k)
        if seqused_q is not None and isinstance(seqused_q, str):
            seqused_q = ast.literal_eval(seqused_q)
        if seqused_k is not None and isinstance(seqused_k, str):
            seqused_k = ast.literal_eval(seqused_k)
        if cmp_residual_k is not None and isinstance(cmp_residual_k, str):
            cmp_residual_k = ast.literal_eval(cmp_residual_k)
        if query_datarange is not None and isinstance(query_datarange, str):
            query_datarange = ast.literal_eval(query_datarange)
        if key_datarange is not None and isinstance(key_datarange, str):
            key_datarange = ast.literal_eval(key_datarange)
        if weights_datarange is not None and isinstance(weights_datarange, str):
            weights_datarange = ast.literal_eval(weights_datarange)
        if output_idx_offset is not None and isinstance(output_idx_offset, str):
            output_idx_offset = ast.literal_eval(output_idx_offset)
            output_idx_offset = [int(x) for x in output_idx_offset]
            if layout_query == "TND":
                output_idx_offset_size = q_t_size * 1
            else:
                output_idx_offset_size = batch_size * q_seq * 1
            output_idx_offset = [
                [random.randint(output_idx_offset[0], output_idx_offset[1]) for _ in range(output_idx_offset_size)]
                for _ in range(1)
            ]
        if isinstance(q_scale_datarange, str):
            q_scale_datarange = ast.literal_eval(q_scale_datarange)
        if isinstance(k_scale_datarange, str):
            k_scale_datarange = ast.literal_eval(k_scale_datarange)

    hifp8mode = 1 if quant_mode == QUANT_MODE_HIFLOAT8 else 0
    if is_mx_quant_mode(quant_mode):
        validate_mx_scale_dtype(dequant_dtype)

    # ======================== 核心推导：从 cu_seqlens / seqused 推导个体长度 ========================
    # 辅助函数：从前缀和 cu_seqlens [B+1] 推导个体长度 [B]
    def _cu_seqlens_to_lengths(cu_list):
        return [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]

    # Q 侧个体长度（CPU golden 用）
    if layout_query == "TND":
        # TND: 必传 cu_seqlens_q，从差分推导个体长度
        assert cu_seqlens_q is not None, "TND layout requires cu_seqlens_q"
        lengths_q_list = _cu_seqlens_to_lengths(cu_seqlens_q)
    else:
        # BSND: 从 seqused_q 获取，若 None 则用 q_seq 填满
        if seqused_q is not None:
            lengths_q_list = list(seqused_q)
        else:
            lengths_q_list = [q_seq] * batch_size

    # K 侧个体长度（CPU golden 用）
    if layout_key == "TND":
        # TND: 必传 cu_seqlens_k，从差分推导个体长度
        assert cu_seqlens_k is not None, "TND layout requires cu_seqlens_k"
        lengths_k_list = _cu_seqlens_to_lengths(cu_seqlens_k)
    elif layout_key == "PA_BBND":
        # PA_BBND: 从 seqused_k 获取
        assert seqused_k is not None, f"{layout_key} layout requires seqused_k"
        lengths_k_list = list(seqused_k)
    else:
        # BSND: 从 seqused_k 获取，若 None 则用 q_seq 填满
        if seqused_k is not None:
            lengths_k_list = list(seqused_k)
        else:
            lengths_k_list = [k_seq] * batch_size

    # ======================== 构造 NPU 输入 tensor ========================
    # cu_seqlens tensor（仅 TND 传入）
    if layout_query == "TND":
        cu_seqlens_query = torch.tensor(cu_seqlens_q).to(actual_seq_dtype)
    else:
        cu_seqlens_query = None

    if layout_key == "TND":
        cu_seqlens_key = torch.tensor(cu_seqlens_k).to(actual_seq_dtype)
    else:
        cu_seqlens_key = None

    # seqused tensor
    if seqused_q is not None:
        seqused_q_tensor = torch.tensor(seqused_q).to(actual_seq_dtype)
    else:
        seqused_q_tensor = None
    if seqused_k is not None:
        seqused_k_tensor = torch.tensor(seqused_k).to(actual_seq_dtype)
    else:
        seqused_k_tensor = None

    # ======================== CPU golden forward 用的 actual_seq ========================
    # TND:     actual_seq 是前缀和格式，即 cu_seqlens[1:]（去掉首位 0）
    #          golden.forward 内部会 trans_tnd_actseq 差分为个体长度
    # BSND/PA: actual_seq 是个体长度，即 seqused
    # （actual_seq始终传入，CPU golden 也需要）
    if layout_query == "TND":
        actual_seq_lengths_query = torch.tensor(cu_seqlens_q[1:]).to(actual_seq_dtype)
    else:
        actual_seq_lengths_query = torch.tensor(lengths_q_list).to(actual_seq_dtype)

    if layout_key == "TND":
        actual_seq_lengths_key = torch.tensor(cu_seqlens_k[1:]).to(actual_seq_dtype)
    else:
        actual_seq_lengths_key = torch.tensor(lengths_k_list).to(actual_seq_dtype)

    # PA_BBND key 构造用的 act_seq_k 列表
    act_seq_k = lengths_k_list

    # 检查 cmp_residual_k 参数
    if (sparse_mode == 0 or cmp_ratio == 1) and cmp_residual_k is not None:
        print(
            f"Warning: sparse_mode={sparse_mode} or cmp_ratio={cmp_ratio}, "
            f"cmp_residual_k={cmp_residual_k}, should be None"
        )
        print("Hint: set cmp_residual_k to None when sparse_mode==0 or cmp_ratio==1")

    # cmp_residual_k for CPU golden (always a list with zeros when cmp_ratio==1 or sparse_mode==0)
    if cmp_ratio == 1 or sparse_mode == 0:
        cmp_residual_k_for_cpu = [0] * batch_size
    else:
        cmp_residual_k_for_cpu = list(cmp_residual_k)

    # cmp_residual_k for NPU (None when cmp_ratio==1 or sparse_mode==0, tensor otherwise)
    if cmp_ratio == 1 or sparse_mode == 0:
        cmp_residual_k_for_npu = None
    else:
        cmp_residual_k_for_npu = torch.tensor(cmp_residual_k).to(actual_seq_dtype)

    if cu_seqlens_q is not None:
        cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    if cu_seqlens_k is not None:
        cu_seqlens_k = torch.tensor(cu_seqlens_k).to(torch.int32)
    if seqused_q is not None:
        seqused_q = torch.tensor(seqused_q).to(torch.int32)
    if seqused_k is not None:
        seqused_k = torch.tensor(seqused_k).to(torch.int32)
    # ======================== 构造 GeneralizedQLIV2 用于 CPU golden ========================
    # GeneralizedQLIV2 需要 act_seq 个体长度（用于 TND→BNSD 转换等）
    test_qliv2 = GeneralizedQLIV2(
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        dequant_dtype,
        actual_seq_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        lengths_q_list,
        lengths_k_list,
        cmp_residual_k_for_cpu,
        max_seqlen_q,
        quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
        cmp_ratio,
        return_value,
        split_s1=split_s1,
        s1size=s1size,
    )

    qk_physical_head_dim = get_qk_physical_head_dim(head_dim, quant_mode)
    mx_scale_tail_shape = (head_dim // MX_SCALE_SHAPE_ALIGN, MX_SCALE_PACK_NUM)
    if layout_query == "BSND":
        q_logical_shape = (batch_size, q_seq, q_head_num, head_dim)
        q_physical_shape = (batch_size, q_seq, q_head_num, qk_physical_head_dim)
        if is_mxfp4_quant_mode(quant_mode):
            query, query_cpu_ref = make_mxfp4_tensor_pair(q_logical_shape, query_datarange, qk_dtype, to_npu=False)
        else:
            query_base = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], q_logical_shape)).to(
                torch.float
            )
            if hifp8mode == 1:
                query = trans_float_tensor_to_hifuint8(query_base, round_mode="hybrid", over_mode=True)
            else:
                query = query_base.to(qk_dtype)
            query_cpu_ref = query

        q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
        if is_mx_quant_mode(quant_mode):
            query_dequant_scale, query_dequant_scale_cpu = make_mx_e8m0_scale_pair(
                (batch_size, q_seq, q_head_num),
                mx_scale_tail_shape,
                q_scale_datarange,
                dequant_dtype,
                to_npu=False,
            )
        elif quant_mode == QUANT_MODE_HIFLOAT8:
            query_dequant_scale = torch.tensor([q_scale]).to(dequant_dtype)
            query_dequant_scale_cpu = torch.tensor(
                np.random.uniform(q_scale, q_scale, (batch_size, q_seq, q_head_num))
            ).to(dequant_dtype)
        else:
            query_dequant_scale = torch.tensor(
                np.random.uniform(
                    q_scale_datarange[0],
                    q_scale_datarange[1],
                    (batch_size, q_seq, q_head_num),
                )
            ).to(dequant_dtype)
            query_dequant_scale_cpu = query_dequant_scale

        weights_cpu = torch.tensor(
            np.random.uniform(
                weights_datarange[0],
                weights_datarange[1],
                (batch_size, q_seq, q_head_num),
            )
        ).to(weight_dtype)
        weights = weights_cpu
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(batch_size, q_seq, 1).to(torch.int32)
    elif layout_query == "TND":
        q_logical_shape = (q_t_size, q_head_num, head_dim)
        q_physical_shape = (q_t_size, q_head_num, qk_physical_head_dim)
        if is_mxfp4_quant_mode(quant_mode):
            query, query_cpu_ref = make_mxfp4_tensor_pair(q_logical_shape, query_datarange, qk_dtype, to_npu=False)
        else:
            query_base = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], q_logical_shape)).to(
                torch.float
            )
            if hifp8mode == 1:
                query = trans_float_tensor_to_hifuint8(query_base, round_mode="hybrid", over_mode=True)
            else:
                query = query_base.to(qk_dtype)
            query_cpu_ref = query

        q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
        if is_mx_quant_mode(quant_mode):
            query_dequant_scale, query_dequant_scale_cpu = make_mx_e8m0_scale_pair(
                (q_t_size, q_head_num),
                mx_scale_tail_shape,
                q_scale_datarange,
                dequant_dtype,
                to_npu=False,
            )
        elif quant_mode == QUANT_MODE_HIFLOAT8:
            query_dequant_scale = torch.tensor([q_scale]).to(dequant_dtype)
            query_dequant_scale_cpu = torch.tensor(np.random.uniform(q_scale, q_scale, (q_t_size, q_head_num))).to(
                dequant_dtype
            )
        else:
            query_dequant_scale = torch.tensor(
                np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], (q_t_size, q_head_num))
            ).to(dequant_dtype)
            query_dequant_scale_cpu = query_dequant_scale

        weights_cpu = torch.tensor(
            np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))
        ).to(weight_dtype)
        weights = weights_cpu
        if output_idx_offset is not None:
            output_idx_offset = torch.tensor(output_idx_offset).reshape(q_t_size, 1).to(torch.int32)

    blockFusion = None
    if layout_key == "BSND":
        k_logical_shape = (batch_size, k_seq, k_head_num, head_dim)
        k_physical_shape = (batch_size, k_seq, k_head_num, qk_physical_head_dim)
        if is_mxfp4_quant_mode(quant_mode):
            key, key_cpu_ref = make_mxfp4_tensor_pair(k_logical_shape, key_datarange, qk_dtype, to_npu=False)
        else:
            key_base = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], k_logical_shape)).to(
                torch.float
            )
            if hifp8mode == 1:
                key = trans_float_tensor_to_hifuint8(key_base, round_mode="hybrid", over_mode=True)
            else:
                key = key_base.to(qk_dtype)
            key_cpu_ref = key

        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if is_mx_quant_mode(quant_mode):
            key_dequant_scale, key_dequant_scale_cpu = make_mx_e8m0_scale_pair(
                (batch_size, k_seq, k_head_num),
                mx_scale_tail_shape,
                k_scale_datarange,
                dequant_dtype,
                to_npu=False,
            )
        elif quant_mode == QUANT_MODE_HIFLOAT8:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype)
            key_dequant_scale_cpu = torch.tensor(
                np.random.uniform(k_scale, k_scale, (batch_size, k_seq, k_head_num))
            ).to(dequant_dtype)
        else:
            key_dequant_scale = torch.tensor(
                np.random.uniform(
                    k_scale_datarange[0],
                    k_scale_datarange[1],
                    (batch_size, k_seq, k_head_num),
                )
            ).to(dequant_dtype)
            key_dequant_scale_cpu = key_dequant_scale

        block_table = None
        if generate_golden:
            cpu_result, topk_value, cpu_topk_value = test_qliv2.forward(
                query_cpu_ref,
                key_cpu_ref,
                weights,
                query_dequant_scale_cpu,
                key_dequant_scale_cpu,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                block_table,
                output_idx_offset,
            )
        else:
            cpu_result, topk_value, cpu_topk_value = None, None, None

    elif layout_key == "TND":
        k_logical_shape = (k_t_size, k_head_num, head_dim)
        k_physical_shape = (k_t_size, k_head_num, qk_physical_head_dim)
        if is_mxfp4_quant_mode(quant_mode):
            key, key_cpu_ref = make_mxfp4_tensor_pair(k_logical_shape, key_datarange, qk_dtype, to_npu=False)
        else:
            key_base = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], k_logical_shape)).to(
                torch.float
            )
            if hifp8mode == 1:
                key = trans_float_tensor_to_hifuint8(key_base, round_mode="hybrid", over_mode=True)
            else:
                key = key_base.to(qk_dtype)
            key_cpu_ref = key

        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if is_mx_quant_mode(quant_mode):
            key_dequant_scale, key_dequant_scale_cpu = make_mx_e8m0_scale_pair(
                (k_t_size, k_head_num),
                mx_scale_tail_shape,
                k_scale_datarange,
                dequant_dtype,
                to_npu=False,
            )
        elif quant_mode == QUANT_MODE_HIFLOAT8:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype)
            key_dequant_scale_cpu = torch.tensor(np.random.uniform(k_scale, k_scale, (k_t_size, k_head_num))).to(
                dequant_dtype
            )
        else:
            key_dequant_scale = torch.tensor(
                np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (k_t_size, k_head_num))
            ).to(dequant_dtype)
            key_dequant_scale_cpu = key_dequant_scale

        block_table = None
        if generate_golden:
            cpu_result, topk_value, cpu_topk_value = test_qliv2.forward(
                query_cpu_ref,
                key_cpu_ref,
                weights,
                query_dequant_scale_cpu,
                key_dequant_scale_cpu,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                block_table,
                output_idx_offset,
            )
        else:
            cpu_result, topk_value, cpu_topk_value = None, None, None

    elif layout_key == "PA_BBND":
        k_max_s2 = math.floor(max(act_seq_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)
        k_logical_shape = (batch_size, k_head_num, k_max_s2, head_dim)
        k_physical_shape = (batch_size, k_head_num, k_max_s2, qk_physical_head_dim)
        if is_mxfp4_quant_mode(quant_mode):
            key_bnsd, key_bnsd_cpu_ref = make_mxfp4_tensor_pair(k_logical_shape, key_datarange, qk_dtype, to_npu=False)
        else:
            key_bnsd_base = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], k_logical_shape)).to(
                torch.float
            )
            if hifp8mode == 1:
                key_bnsd = trans_float_tensor_to_hifuint8(key_bnsd_base, round_mode="hybrid", over_mode=True)
            else:
                key_bnsd = key_bnsd_base.to(qk_dtype)
            key_bnsd_cpu_ref = key_bnsd

        k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
        if is_mx_quant_mode(quant_mode):
            key_dequant_scale_bns_mx, key_dequant_scale_bns = make_mx_e8m0_scale_pair(
                (batch_size, k_head_num, k_max_s2),
                mx_scale_tail_shape,
                k_scale_datarange,
                dequant_dtype,
                to_npu=False,
            )
        elif quant_mode == QUANT_MODE_HIFLOAT8:
            key_dequant_scale_bns = torch.tensor(
                np.random.uniform(k_scale, k_scale, (batch_size, k_head_num, k_max_s2))
            ).to(dequant_dtype)
        else:
            key_dequant_scale_bns = torch.tensor(
                np.random.uniform(
                    k_scale_datarange[0],
                    k_scale_datarange[1],
                    (batch_size, k_head_num, k_max_s2),
                )
            ).to(dequant_dtype)

        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError("key actual block num < needed block num")

        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full((batch_size, k_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        if is_mxfp4_quant_mode(quant_mode):
            # FP4 shell dtype仅用于接口语义；PA重排按其底层packed uint8字节完成。
            key_storage_dtype = torch.uint8
            key_bnsd_storage = key_bnsd.view(torch.uint8)
        else:
            key_storage_dtype = qk_dtype
            key_bnsd_storage = key_bnsd
        key_expand = torch.zeros(
            (
                batch_size,
                k_head_num,
                k_max_block_num_per_batch * block_size,
                qk_physical_head_dim,
            ),
            dtype=key_storage_dtype,
        )
        key_expand[:, :, :k_max_s2, :] = key_bnsd_storage
        key = torch.zeros(
            (block_num, block_size, k_head_num, qk_physical_head_dim),
            dtype=key_storage_dtype,
        )
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[
                            i_batch,
                            i_n,
                            block_start_pos : block_start_pos + block_size,
                            :,
                        ]

        if is_mx_quant_mode(quant_mode):
            key_dequant_scale_expand = make_e8m0_zero(
                (
                    batch_size,
                    k_head_num,
                    k_max_block_num_per_batch * block_size,
                    *mx_scale_tail_shape,
                ),
                dequant_dtype,
            )
            key_dequant_scale_expand[:, :, :k_max_s2, :, :] = key_dequant_scale_bns_mx
            key_dequant_scale_block = make_e8m0_zero(
                (block_num, block_size, k_head_num, *mx_scale_tail_shape), dequant_dtype
            )
            for i_batch in range(batch_size):
                for i_block, cur_block_id in enumerate(block_table[i_batch]):
                    block_start_pos = i_block * block_size
                    if cur_block_id == -1:
                        continue
                    else:
                        for i_n in range(k_head_num):
                            key_dequant_scale_block[cur_block_id, :, i_n, :, :] = key_dequant_scale_expand[
                                i_batch,
                                i_n,
                                block_start_pos : block_start_pos + block_size,
                                :,
                                :,
                            ]
        else:
            key_dequant_scale_expand = torch.zeros(
                (batch_size, k_head_num, k_max_block_num_per_batch * block_size),
                dtype=dequant_dtype,
            )
            key_dequant_scale_expand[:, :, :k_max_s2] = key_dequant_scale_bns
            key_dequant_scale_block = torch.zeros((block_num, block_size, k_head_num), dtype=dequant_dtype)
            for i_batch in range(batch_size):
                for i_block, cur_block_id in enumerate(block_table[i_batch]):
                    block_start_pos = i_block * block_size
                    if cur_block_id == -1:
                        continue
                    else:
                        for i_n in range(k_head_num):
                            key_dequant_scale_block[cur_block_id, :, i_n] = key_dequant_scale_expand[
                                i_batch,
                                i_n,
                                block_start_pos : block_start_pos + block_size,
                            ]

        if not is_mx_quant_mode(quant_mode) and quant_mode != QUANT_MODE_HIFLOAT8 and DISCONTINUOUS_KEYS:
            bytes_per_token = head_dim + key_dequant_scale_block.element_size() // key.element_size()
            blockFusion = torch.zeros((block_num, block_size * k_head_num * bytes_per_token), dtype=qk_dtype)
            key_flat = key.view(block_num, block_size * k_head_num * head_dim)
            scale_flat = key_dequant_scale_block.view(block_num, block_size * k_head_num).view(qk_dtype)
            blockFusion[:, : block_size * k_head_num * head_dim] = key_flat
            blockFusion[:, block_size * k_head_num * head_dim :] = scale_flat

        if is_mxfp4_quant_mode(quant_mode):
            key = key.view(qk_dtype)
        if quant_mode == QUANT_MODE_HIFLOAT8:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype)
        else:
            key_dequant_scale = key_dequant_scale_block
        if generate_golden:
            cpu_result, topk_value, cpu_topk_value = test_qliv2.forward(
                query_cpu_ref,
                key_bnsd_cpu_ref,
                weights,
                query_dequant_scale_cpu,
                key_dequant_scale_bns,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                block_table,
                output_idx_offset,
            )
        else:
            cpu_result, topk_value, cpu_topk_value = None, None, None
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32)
    # ======================== metadata 构造 ========================
    # max_seqlen 从个体长度中取
    max_seqlen_q_meta = actual_seq_lengths_query.max().item()
    max_seqlen_k_meta = actual_seq_lengths_key.max().item()

    if is_batch:
        if qk_dtype == torch.float8_e4m3fn:
            query = query.to(dtype=torch.float16)
            key = key.to(dtype=torch.float16)
            if blockFusion is not None:
                blockFusion = blockFusion.view(torch.uint8)

        golden_key = key_bnsd_cpu_ref if layout_key == "PA_BBND" else key_cpu_ref
        golden_key_scale = key_dequant_scale_bns if layout_key == "PA_BBND" else key_dequant_scale_cpu
        golden_block_table = block_table
        if torch.is_tensor(golden_block_table):
            golden_block_table = golden_block_table.detach().cpu()
        golden_state = {
            "model_args": (
                batch_size,
                q_seq,
                k_seq,
                q_t_size,
                k_t_size,
                q_head_num,
                k_head_num,
                head_dim,
                block_size,
                block_num,
                qk_dtype,
                weight_dtype,
                dequant_dtype,
                actual_seq_dtype,
                cu_seqlens_q,
                cu_seqlens_k,
                lengths_q_list,
                lengths_k_list,
                cmp_residual_k_for_cpu,
                max_seqlen_q,
                quant_mode,
                layout_query,
                layout_key,
                sparse_count,
                sparse_mode,
                query_datarange,
                key_datarange,
                weights_datarange,
                q_scale_datarange,
                k_scale_datarange,
                cmp_ratio,
                return_value,
            ),
            "split_s1": split_s1,
            "s1size": s1size,
            "forward_inputs": {
                "query": query_cpu_ref,
                "key": golden_key,
                "weights": weights,
                "query_dequant_scale": query_dequant_scale_cpu,
                "key_dequant_scale": golden_key_scale,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "block_table": golden_block_table,
                "output_idx_offset": output_idx_offset,
            },
        }

        output_tensors = {
            "params": params,
            "cpu_result": cpu_result,
            "topk_value": topk_value,
            "cpu_topk_value": cpu_topk_value,
            "query": query,
            "key": key,
            "weights": weights,
            "query_dequant_scale": query_dequant_scale,
            "key_dequant_scale": key_dequant_scale,
            "blockFusion": blockFusion,
            "cu_seqlens_query": cu_seqlens_query,
            "cu_seqlens_key": cu_seqlens_key,
            "seqused_q": seqused_q_tensor,
            "seqused_k": seqused_k_tensor,
            "output_idx_offset": output_idx_offset,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "cmp_residual_k_for_npu": cmp_residual_k_for_npu,
            "block_table": block_table,
            "max_seqlen_q_meta": max_seqlen_q_meta,
            "max_seqlen_k_meta": max_seqlen_k_meta,
            "quant_mode": quant_mode,
            "layout_query": layout_query,
            "layout_key": layout_key,
            "sparse_count": sparse_count,
            "sparse_mode": sparse_mode,
            "cmp_ratio": cmp_ratio,
            "golden_state": golden_state,
        }
        return output_tensors
    else:
        metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu_seqlens_query.npu() if cu_seqlens_query is not None else None,
            cu_seqlens_k=cu_seqlens_key.npu() if cu_seqlens_key is not None else None,
            seqused_q=seqused_q_tensor.npu() if seqused_q_tensor is not None else None,
            seqused_k=seqused_k_tensor.npu() if seqused_k_tensor is not None else None,
            cmp_residual_k=cmp_residual_k_for_npu.npu() if cmp_residual_k_for_npu is not None else None,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q_meta,
            max_seqlen_k=max_seqlen_k_meta,
            num_heads_q=q_head_num,
            num_heads_k=k_head_num,
            head_dim=head_dim,
            topk=sparse_count,
            quant_mode=quant_mode,
            mask_mode=sparse_mode,
            layout_q=layout_query,
            layout_k=layout_key,
            cmp_ratio=cmp_ratio,
        )
        metadata = metadata.npu()
        if blockFusion is not None:
            blockFusion = blockFusion.npu()
            key = blockFusion[:, : block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)
            key_dequant_scale_block = (
                blockFusion[:, block_size * k_head_num * head_dim :]
                .view(dequant_dtype)
                .view(block_num, block_size, k_head_num)
            )
        else:
            key = key.npu()
        if quant_mode == QUANT_MODE_HIFLOAT8:
            key_dequant_scale = torch.tensor([k_scale]).to(dequant_dtype).npu()
        else:
            if layout_key == "PA_BBND":
                key_dequant_scale = key_dequant_scale_block.npu()
            else:
                key_dequant_scale = key_dequant_scale.npu()

        npu_result, npu_topk_value = torch.ops.cann_ops_transformer.quant_lightning_indexer(
            query.npu(),
            key,
            weights.npu(),
            query_dequant_scale.npu(),
            key_dequant_scale,
            cu_seqlens_q=cu_seqlens_query.npu() if cu_seqlens_query is not None else None,
            cu_seqlens_k=cu_seqlens_key.npu() if cu_seqlens_key is not None else None,
            seqused_q=seqused_q_tensor.npu() if seqused_q_tensor is not None else None,
            seqused_k=seqused_k_tensor.npu() if seqused_k_tensor is not None else None,
            cmp_residual_k=cmp_residual_k_for_npu.npu() if cmp_residual_k_for_npu is not None else None,
            output_idx_offset=output_idx_offset.npu() if output_idx_offset is not None else None,
            max_seqlen_q=max_seqlen_q,
            block_table=block_table.npu() if block_table is not None else None,
            metadata=metadata,
            quant_mode=quant_mode,
            layout_q=layout_query,
            layout_k=layout_key,
            topk=sparse_count,
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
        )

        torch.npu.synchronize()
        if return_value:
            if npu_topk_value.shape != npu_result.shape:
                raise RuntimeError(
                    "sparse_values and sparse_indices must have the same shape when return_value=1, "
                    f"but got {tuple(npu_topk_value.shape)} and {tuple(npu_result.shape)}"
                )
            npu_topk_value, npu_sort_order = npu_topk_value.sort(dim=-1, descending=True)
            npu_result = torch.gather(npu_result, dim=-1, index=npu_sort_order)
        return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value


def build_qliv2_metadata_input(params, tensor_dict):
    """Return the canonical metadata arguments produced by the pytest case."""
    return {
        "num_heads_q": int(params[5]),
        "num_heads_k": int(params[6]),
        "head_dim": int(params[7]),
        "topk": int(tensor_dict["sparse_count"]),
        "quant_mode": int(tensor_dict["quant_mode"]),
        "cu_seqlens_q": tensor_dict["cu_seqlens_query"],
        "cu_seqlens_k": tensor_dict["cu_seqlens_key"],
        "seqused_q": tensor_dict["seqused_q"],
        "seqused_k": tensor_dict["seqused_k"],
        "cmp_residual_k": tensor_dict["cmp_residual_k_for_npu"],
        "batch_size": int(params[0]),
        "max_seqlen_q": int(tensor_dict["max_seqlen_q_meta"]),
        "max_seqlen_k": int(tensor_dict["max_seqlen_k_meta"]),
        "layout_q": tensor_dict["layout_query"],
        "layout_k": tensor_dict["layout_key"],
        "mask_mode": int(tensor_dict["sparse_mode"]),
        "cmp_ratio": int(tensor_dict["cmp_ratio"]),
    }


def generate_qliv2_test_data(
    params,
    split_s1=DEFAULT_SPLIT_S1,
    s1size=DEFAULT_S1SIZE,
    generate_golden=True,
):
    """Generate QLI_V2 inputs and optionally materialize the CPU Golden."""
    data = qliv2_output_single(
        params,
        is_batch=True,
        split_s1=split_s1,
        s1size=s1size,
        generate_golden=generate_golden,
    )
    data["metadata_input"] = build_qliv2_metadata_input(data["params"], data)
    return data


def generate_cpu_golden(input_data):
    """Calculate CPU Golden from input-stage data without rerunning random generation."""
    state = input_data["golden_state"]
    test_qliv2 = GeneralizedQLIV2(
        *state["model_args"],
        split_s1=state["split_s1"],
        s1size=state["s1size"],
    )
    values = state["forward_inputs"]
    block_table = values["block_table"]
    if torch.is_tensor(block_table):
        # The CPU reference indexes block tables as an ndarray; keep the saved
        # tensor intact for PT consumers and convert only this local argument.
        block_table = block_table.detach().cpu().numpy()
    cpu_result, topk_value, cpu_topk_value = test_qliv2.forward(
        values["query"],
        values["key"],
        values["weights"],
        values["query_dequant_scale"],
        values["key_dequant_scale"],
        values["cu_seqlens_q"],
        values["cu_seqlens_k"],
        values["seqused_q"],
        values["seqused_k"],
        block_table,
        values["output_idx_offset"],
    )
    state["cpu_result"] = cpu_result
    state["topk_value"] = topk_value
    state["cpu_topk_value"] = cpu_topk_value
    input_data["cpu_result"] = cpu_result
    input_data["topk_value"] = topk_value
    input_data["cpu_topk_value"] = cpu_topk_value
    return cpu_result, topk_value, cpu_topk_value


def fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        return True, 0
    hif8_value_tmp = fraction32_int >> (FP32_FRACTION_BITS - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = SSR_T14_MASK
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        f14_values = (fraction32_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (FP32_FRACTION_BITS - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (FP32_FRACTION_BITS - hif8_bits_num))
        f14_values = f14_t14 >> (FP32_FRACTION_BITS - hif8_bits_num - SSR_RESERVED_BITS)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def get_hif8_fraction_bits_number(exponent):
    if exponent < HIF8_EXP_DML_MIN:
        return HIF8_DOT_INVALID, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if HIF8_EXP_DML_MIN <= exponent < HIF8_EXP_DML_MAX:
        return HIF8_DOT_DML, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if exponent == HIF8_EXP_D0:
        return HIF8_DOT_D0, HIF8_EXP_BITS_D0, HIF8_FRAC_BITS_D0
    if abs(exponent) == HIF8_EXP_D1_BOUNDARY:
        return HIF8_DOT_D1, HIF8_EXP_BITS_D1, HIF8_FRAC_BITS_D1
    if HIF8_EXP_D2_MIN <= abs(exponent) <= HIF8_EXP_D2_MAX:
        return HIF8_DOT_D2, HIF8_EXP_BITS_D2, HIF8_FRAC_BITS_D2
    if HIF8_EXP_D3_MIN <= abs(exponent) <= HIF8_EXP_D3_MAX:
        return HIF8_DOT_D3, HIF8_EXP_BITS_D3, HIF8_FRAC_BITS_D3
    if HIF8_EXP_D4_MIN <= abs(exponent) <= HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_FRAC_BITS_D4
    if exponent > HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_DOT_INVALID


def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    ec = 0
    over_value = HIF8_OVERFLOW_SCALE * pow(2.0, HIF8_EXP_D4_MAX + ec)
    if x < 0.0:
        sign = True
        sign_int_value = HIF8_SIGN_MASK
    if torch.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return HIF8_NEG_INF
            else:
                return HIF8_NEG_MAX
        else:
            if over_mode:
                return HIF8_POS_INF
            else:
                return HIF8_POS_MAX
    if torch.isnan(x):
        if over_mode:
            return HIF8_NAN
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent) - pow(2, FP32_FRACTION_BITS))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print("unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < HIF8_EXP_ZERO_THRESHOLD:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= HIF8_EXP_ZERO_THRESHOLD:
            return 0
        else:
            return sign_int_value + exponent + HIF8_DML_EXP_OFFSET
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value


def trans_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    """
    通过向量操作，将 float32 Tensor 批量转换为 HiF8 编码的 uint8 Tensor
    """

    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.float32)

    # 先用int32作为输出类型，避免出现赋值错误
    out = torch.zeros_like(x, dtype=torch.int32)

    # 1. 符号位与绝对值提取
    sign_mask = x < 0.0
    sign_int_value = torch.where(sign_mask, HIF8_SIGN_MASK, 0)
    x_abs = torch.abs(x)

    # 2. 溢出与边界条件判断 (Masks)
    over_value = HIF8_OVERFLOW_SCALE * (2.0**HIF8_EXP_D4_MAX)
    mask_inf_or_over = torch.isinf(x) | (x_abs >= over_value)
    mask_nan = torch.isnan(x)
    mask_zero = x_abs == 0.0

    # 处理特殊边界填值
    if over_mode:
        out = torch.where(mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_INF, HIF8_POS_INF), out)
        out = torch.where(mask_nan, HIF8_NAN, out)
    else:
        out = torch.where(mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_MAX, HIF8_POS_MAX), out)
        out = torch.where(mask_nan, 0, out)
    out = torch.where(mask_zero, 0, out)

    # 提取正常数字的 Mask
    mask_normal = ~(mask_inf_or_over | mask_nan | mask_zero)
    if not mask_normal.any():
        return out.reshape(shape).to(torch.uint8)

    x_norm = x_abs[mask_normal]
    sign_norm = sign_int_value[mask_normal]

    # 计算基本指数
    exponent = torch.floor(torch.log2(x_norm)).to(torch.int32)

    # 确定截断模式 (TA / SSR)
    if round_mode == "hybrid":
        cut_bit_is_ta = torch.abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD
    elif round_mode == "round":
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)
    elif round_mode == "storound":
        cut_bit_is_ta = torch.zeros_like(exponent, dtype=torch.bool)
    else:
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)

    # 计算 fraction_int
    fraction_int = (
        x_norm * (2.0**FP32_FRACTION_BITS) * torch.pow(2.0, -exponent.float()) - (2.0**FP32_FRACTION_BITS)
    ).to(torch.int32)

    # 批量获取档位属性 (根据 exponent 映射)
    abs_exp = torch.abs(exponent)

    dot = torch.full_like(exponent, HIF8_DOT_INVALID)
    exp_bits = torch.zeros_like(exponent)
    frac_bits = torch.zeros_like(exponent)

    # 条件区间映射
    m1 = exponent < HIF8_EXP_DML_MIN
    dot = torch.where(m1, HIF8_DOT_INVALID, dot)
    exp_bits = torch.where(m1, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m1, HIF8_FRAC_BITS_DML, frac_bits)

    m2 = (~m1) & (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX)
    dot = torch.where(m2, HIF8_DOT_DML, dot)
    exp_bits = torch.where(m2, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m2, HIF8_FRAC_BITS_DML, frac_bits)

    m3 = exponent == HIF8_EXP_D0
    dot = torch.where(m3, HIF8_DOT_D0, dot)
    exp_bits = torch.where(m3, HIF8_EXP_BITS_D0, exp_bits)
    frac_bits = torch.where(m3, HIF8_FRAC_BITS_D0, frac_bits)

    m4 = abs_exp == HIF8_EXP_D1_BOUNDARY
    dot = torch.where(m4, HIF8_DOT_D1, dot)
    exp_bits = torch.where(m4, HIF8_EXP_BITS_D1, exp_bits)
    frac_bits = torch.where(m4, HIF8_FRAC_BITS_D1, frac_bits)

    m5 = (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX)
    dot = torch.where(m5, HIF8_DOT_D2, dot)
    exp_bits = torch.where(m5, HIF8_EXP_BITS_D2, exp_bits)
    frac_bits = torch.where(m5, HIF8_FRAC_BITS_D2, frac_bits)

    m6 = (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX)
    dot = torch.where(m6, HIF8_DOT_D3, dot)
    exp_bits = torch.where(m6, HIF8_EXP_BITS_D3, exp_bits)
    frac_bits = torch.where(m6, HIF8_FRAC_BITS_D3, frac_bits)

    m7 = (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX)
    dot = torch.where(m7, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m7, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m7, HIF8_FRAC_BITS_D4, frac_bits)

    m8 = exponent > HIF8_EXP_D4_MAX
    dot = torch.where(m8, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m8, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m8, HIF8_DOT_INVALID, frac_bits)

    # ------------------ TA 舍入分支 ------------------
    carry_ta = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ta = torch.zeros_like(exponent)

    m_zero_thresh = exponent == HIF8_EXP_ZERO_THRESHOLD
    carry_ta = torch.where(m_zero_thresh, True, carry_ta)

    m_ta_norm = ~m_zero_thresh
    shift_bits = torch.clamp(FP32_FRACTION_BITS - (frac_bits + 1), min=0)
    hif8_val_tmp = fraction_int >> shift_bits

    pow_frac = torch.pow(2, frac_bits + 1) - 1
    m_carry = m_ta_norm & (hif8_val_tmp == pow_frac)
    carry_ta = torch.where(m_carry, True, carry_ta)

    m_odd = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 1)
    frac_val_ta = torch.where(m_odd, (hif8_val_tmp + 1) >> 1, frac_val_ta)

    m_even = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 0)
    frac_val_ta = torch.where(m_even, hif8_val_tmp >> 1, frac_val_ta)

    # ------------------ SSR 舍入分支 ------------------
    carry_ssr = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ssr = torch.zeros_like(exponent)

    f14_v1 = (fraction_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
    t14_v1 = fraction_int & SSR_T14_MASK
    hif8_v1 = torch.zeros_like(fraction_int)

    s_bits = torch.clamp(FP32_FRACTION_BITS - frac_bits, min=0)
    hif8_v2 = fraction_int >> s_bits
    f14_t14 = fraction_int - (hif8_v2 << s_bits)
    s_bits_f14 = torch.clamp(FP32_FRACTION_BITS - frac_bits - SSR_RESERVED_BITS, min=0)
    f14_v2 = f14_t14 >> s_bits_f14
    t14_v2 = f14_t14 & SSR_T14_MASK

    f14_values = torch.where(m_zero_thresh, f14_v1, f14_v2)
    t14_values = torch.where(m_zero_thresh, t14_v1, t14_v2)
    hif8_value = torch.where(m_zero_thresh, hif8_v1, hif8_v2)

    m_ge = f14_values >= t14_values
    pow_frac_ssr = torch.pow(2, frac_bits) - 1
    m_ssr_carry = m_ge & (hif8_value == pow_frac_ssr)
    carry_ssr = torch.where(m_ssr_carry, True, carry_ssr)
    frac_val_ssr = torch.where(m_ge & (~m_ssr_carry), hif8_value + 1, frac_val_ssr)
    frac_val_ssr = torch.where(~m_ge, hif8_value, frac_val_ssr)

    # ------------------ 合并舍入结果 ------------------
    carry_exp_status = torch.where(cut_bit_is_ta, carry_ta, carry_ssr)
    hif8_frac_value = torch.where(cut_bit_is_ta, frac_val_ta, frac_val_ssr)

    exponent = torch.where(carry_exp_status, exponent + 1, exponent)
    abs_exp = torch.abs(exponent)

    dot = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_DOT_INVALID, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_DOT_DML,
            dot,
        ),
        dot,
    )
    dot = torch.where(carry_exp_status, torch.where(exponent == HIF8_EXP_D0, HIF8_DOT_D0, dot), dot)
    dot = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_DOT_D1, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_DOT_D2,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_DOT_D3,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_DOT_D4,
            dot,
        ),
        dot,
    )

    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_FRAC_BITS_DML, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_FRAC_BITS_DML,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_FRAC_BITS_D0, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_FRAC_BITS_D1, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_FRAC_BITS_D2,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_FRAC_BITS_D3,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_FRAC_BITS_D4,
            frac_bits,
        ),
        frac_bits,
    )

    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_EXP_BITS_DML, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_EXP_BITS_DML,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_EXP_BITS_D0, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_EXP_BITS_D1, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_EXP_BITS_D2,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_EXP_BITS_D3,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_EXP_BITS_D4,
            exp_bits,
        ),
        exp_bits,
    )

    # ------------------ 组合输出编码 ------------------
    hif8_int_value = torch.zeros_like(exponent)
    sig_exp = torch.where(exponent < 0, 1, 0)

    # 分支 A: dot <= 0
    m_a = dot <= 0
    val_a = torch.where(
        exponent <= HIF8_EXP_ZERO_THRESHOLD,
        0,
        sign_norm + exponent + HIF8_DML_EXP_OFFSET,
    )
    hif8_int_value = torch.where(m_a, val_a, hif8_int_value)

    # 分支 B: dot == 1
    m_b = dot == 1
    val_b = sign_norm + (dot << HIF8_DOT_BIT_SHIFT) + hif8_frac_value
    hif8_int_value = torch.where(m_b, val_b, hif8_int_value)

    # 分支 C: dot > 1
    m_c = dot > 1
    abs_exponent = torch.abs(exponent)
    abs_exponent = abs_exponent - torch.pow(2, exp_bits - 1)
    exponent_int_value = abs_exponent << frac_bits
    sig_exp_shifted = sig_exp << (exp_bits - 1 + frac_bits)
    dot_int_value = dot << HIF8_DOT_BIT_SHIFT
    val_c = sign_norm + dot_int_value + sig_exp_shifted + exponent_int_value + hif8_frac_value
    hif8_int_value = torch.where(m_c, val_c, hif8_int_value)

    hif8_int_value = torch.where(exponent < HIF8_EXP_ZERO_THRESHOLD, 0, hif8_int_value)

    out[mask_normal] = hif8_int_value

    return out.reshape(shape).to(torch.uint8)


def cvt_hifuint8_to_float32(x, over_mode=True):
    x = int(x)
    if x == HIF8_ZERO:
        return float(0)
    elif x == HIF8_NAN:
        if over_mode:
            return float("nan")
        else:
            return float(0)
    elif x == HIF8_NEG_INF:
        if over_mode:
            return -torch.inf
        else:
            return -HIF8_MAX_FINITE_VALUE
    elif x == HIF8_POS_INF:
        if over_mode:
            return torch.inf
        else:
            return HIF8_MAX_FINITE_VALUE
    else:
        if x >= HIF8_NAN:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & HIF8_DOT_MASK
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= HIF8_DOT_D4:
            exponent = x & HIF8_EXP_MASK_D4
            exponent_int = exponent >> 1
            if exponent_int >= 8:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 8

            fra_int = x & HIF8_FRAC_MASK_1BIT
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= HIF8_DOT_D3:
            exponent = x & HIF8_EXP_MASK_D3
            exponent_int = exponent >> 2
            if exponent_int >= 4:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 4

            fra_int = x & HIF8_FRAC_MASK_2BIT
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= HIF8_DOT_D2:
            exponent = x & HIF8_EXP_MASK_D2
            exponent_int = exponent >> 3
            if exponent_int >= 2:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 2

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= HIF8_DOT_D1:
            exponent = x & HIF8_EXP_SIGN_MASK_D1
            exponent_sign = exponent >> 3
            if exponent_sign >= 1:
                exponent_value = -1
            else:
                exponent_value = 1

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_D0:
            exponent_value = 0
            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_DML:
            m_value = 1
            exponent_value = (x & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET
        else:
            print("error, dot error")
            m_value = 0.0
            exponent_value = 0
        return sign * pow(2.0, exponent_value) * m_value


def trans_hifuint8_tensor_to_float(in_tensor, over_mode=True):
    """
    将 HiF8 编码的 uint8 Tensor 批量转换为 float32 Tensor (支持 CPU/GPU 矢量化)
    """
    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.int32)
    out = torch.zeros_like(x, dtype=torch.float32)

    # 1. 特殊值处理 (Masks)
    mask_zero = x == HIF8_ZERO
    mask_nan = x == HIF8_NAN
    mask_ninf = x == HIF8_NEG_INF
    mask_pinf = x == HIF8_POS_INF

    if over_mode:
        out = torch.where(mask_nan, torch.tensor(float("nan"), device=x.device), out)
        out = torch.where(mask_ninf, torch.tensor(-torch.inf, device=x.device), out)
        out = torch.where(mask_pinf, torch.tensor(torch.inf, device=x.device), out)
    else:
        out = torch.where(mask_nan, 0.0, out)
        out = torch.where(mask_ninf, float(-HIF8_MAX_FINITE_VALUE), out)
        out = torch.where(mask_pinf, float(HIF8_MAX_FINITE_VALUE), out)

    # 正常数值的 Mask (排除特殊值)
    mask_normal = ~(mask_zero | mask_nan | mask_ninf | mask_pinf)
    if not mask_normal.any():
        return out.reshape(shape)

    # 提取正常数值子集进行计算
    x_norm = x[mask_normal]

    # 符号位计算
    sign = torch.where(x_norm >= HIF8_NAN, -1.0, 1.0)

    # 提取 dot 档位
    dot_4_value = (x_norm & HIF8_DOT_MASK) >> 3

    # 初始化指数和尾数乘子
    exponent_value = torch.zeros_like(x_norm, dtype=torch.float32)
    m_value = torch.zeros_like(x_norm, dtype=torch.float32)

    # --- 档位 D4 ---
    m_d4 = dot_4_value >= HIF8_DOT_D4
    if m_d4.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D4) >> 1
        exponent_value = torch.where(
            m_d4,
            torch.where(exp_int >= 8, -exp_int, exp_int + 8).float(),
            exponent_value,
        )
        m_value = torch.where(m_d4, 1.0 + (x_norm & HIF8_FRAC_MASK_1BIT) * 0.5, m_value)

    # --- 档位 D3 ---
    m_d3 = (~m_d4) & (dot_4_value >= HIF8_DOT_D3)
    if m_d3.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D3) >> 2
        exponent_value = torch.where(
            m_d3,
            torch.where(exp_int >= 4, -exp_int, exp_int + 4).float(),
            exponent_value,
        )
        m_value = torch.where(m_d3, 1.0 + (x_norm & HIF8_FRAC_MASK_2BIT) * 0.25, m_value)

    # --- 档位 D2 ---
    m_d2 = (~(m_d4 | m_d3)) & (dot_4_value >= HIF8_DOT_D2)
    if m_d2.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D2) >> 3
        exponent_value = torch.where(
            m_d2,
            torch.where(exp_int >= 2, -exp_int, exp_int + 2).float(),
            exponent_value,
        )
        m_value = torch.where(m_d2, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value)

    # --- 档位 D1 ---
    m_d1 = (~(m_d4 | m_d3 | m_d2)) & (dot_4_value >= HIF8_DOT_D1)
    if m_d1.any():
        exp_sign = (x_norm & HIF8_EXP_SIGN_MASK_D1) >> 3
        exponent_value = torch.where(m_d1, torch.where(exp_sign >= 1, -1.0, 1.0), exponent_value)
        m_value = torch.where(m_d1, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value)

    # --- 档位 D0 ---
    m_d0 = dot_4_value == HIF8_DOT_D0
    if m_d0.any():
        exponent_value = torch.where(m_d0, 0.0, exponent_value)
        m_value = torch.where(m_d0, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value)

    # --- 档位 DML ---
    m_dml = dot_4_value == HIF8_DOT_DML
    if m_dml.any():
        exponent_value = torch.where(
            m_dml,
            ((x_norm & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET).float(),
            exponent_value,
        )
        m_value = torch.where(m_dml, 1.0, m_value)

    # 计算正常值结果并写回
    norm_res = sign * torch.pow(2.0, exponent_value) * m_value
    out[mask_normal] = norm_res

    return out.reshape(shape)
