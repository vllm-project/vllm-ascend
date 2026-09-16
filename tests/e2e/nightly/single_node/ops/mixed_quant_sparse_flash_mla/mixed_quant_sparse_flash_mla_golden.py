#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import torch
import random
import numpy as np
import math

# 两种运行模式
# 0 不切S2
# 1 切S2

RUN_MODE = 1

DATA_RANGE_LEFT = -10
DATA_RANGE_RIGHT = 10
FP8_DATA_RANGE_LEFT = -5
FP8_DATA_RANGE_RIGHT = 5


def restore_cmp_kv_lengths(seqused_cmp_kv, cmp_ratio, cmp_residual_kv=None):
    """Restore the logical CMP context lengths used by the Arch35 kernels."""
    if seqused_cmp_kv is None:
        return None
    if cmp_residual_kv is not None and len(cmp_residual_kv) != len(seqused_cmp_kv):
        raise ValueError("cmp_residual_kv and seqused_cmp_kv must have the same length")

    return [
        int(cmp_len) * cmp_ratio
        + (int(cmp_residual_kv[index]) if cmp_residual_kv is not None else 0)
        for index, cmp_len in enumerate(seqused_cmp_kv)
    ]


def get_kv_compute_dtype(kv_type):
    """Use FP8 semantics when the operator input is stored as disguised uint8."""
    return torch.float8_e4m3fn if kv_type == torch.uint8 else kv_type


def reinterpret_kv_for_operator(kv_tensor, kv_type):
    """Expose FP8 bytes as uint8 without changing their storage."""
    if kv_tensor is not None and kv_type == torch.uint8:
        return kv_tensor.view(torch.uint8)
    return kv_tensor


def resolve_input_data_ranges(q_datarange, ori_kv_datarange, cmp_kv_datarange):
    """Apply the canonical MQSMLA data range defaults."""
    default_range = [DATA_RANGE_LEFT, DATA_RANGE_RIGHT]
    return tuple(
        value if value is not None else list(default_range)
        for value in (q_datarange, ori_kv_datarange, cmp_kv_datarange)
    )


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


class GeneralizedSFAQuant:
    def __init__(
        self,
        layout_q,
        layout_kv,
        q_type,
        ori_kv_type,
        cmp_kv_type,
        B,
        S1,
        T1,
        N1,
        N2,
        D,
        K,
        block_num1,
        block_num2,
        block_size1,
        block_size2,
        cu_seqlens_q,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        quant_mode,
        tile_size,
        rope_head_dim,
        ori_topk_length,
        cmp_topk_length,
        template_run_mode,
    ):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_q = seqused_q
        self.seqused_ori_kv = seqused_ori_kv
        self.seqused_cmp_kv = seqused_cmp_kv
        self.cu_seqlens_ori_kv = cu_seqlens_ori_kv
        self.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv
        self.cmp_residual_kv = cmp_residual_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.quant_mode = quant_mode
        self.tile_size = tile_size
        self.rope_head_dim = rope_head_dim
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length
        self.template_run_mode = template_run_mode

    @staticmethod
    def _get_batch_consistency_reduce_size(ori_s2_size, cmp_s2_size):
        """Return the S2 reduction-block size used by batch-consistency metadata."""
        total_s2_size = ori_s2_size + cmp_s2_size
        # Match the kernel's integer ``s2Load / 32`` quotient before aligning
        # to the 128-token S2 base block.
        raw_reduce_size = total_s2_size // 32
        return max(((raw_reduce_size + 127) // 128) * 128, 128)

    @staticmethod
    def _update_s2_tile(
        q_fp32,
        k_tile,
        softmax_scale,
        softmax_dtype,
        score_max,
        sumexp,
        acc_o,
    ):
        """Use one 128-token S2 tile to update the online-softmax state."""
        mm1_res = torch.matmul(q_fp32, k_tile.T)
        scale_res = mm1_res * softmax_scale

        score_max_pre = score_max.clone()
        cur_score_max = scale_res.max(dim=-1)[0]
        score_max = torch.max(score_max, cur_score_max)
        score_max_pre = torch.exp(score_max_pre - score_max)

        acc_s = torch.exp(scale_res - score_max.unsqueeze(1))
        sumexp = sumexp * score_max_pre + acc_s.sum(dim=-1)
        acc_s_cast = acc_s.to(dtype=softmax_dtype).to(dtype=torch.float32)
        mm2_res = torch.matmul(acc_s_cast, k_tile)
        acc_o = acc_o * score_max_pre.unsqueeze(1) + mm2_res
        return score_max, sumexp, acc_o

    def _calculate_local_s2_block(
        self,
        q_fp32,
        k_block_fp32,
        initial_score_max,
        s2_base_size=128,
    ):
        """Calculate one NPU reduction block and retain its local softmax state."""
        score_max = initial_score_max.clone()
        sumexp = torch.ones_like(score_max, dtype=torch.float32)
        acc_o = torch.zeros(
            (q_fp32.shape[0], k_block_fp32.shape[-1]), dtype=torch.float32
        )

        for s2_start in range(0, k_block_fp32.shape[0], s2_base_size):
            k_tile = k_block_fp32[s2_start : s2_start + s2_base_size]
            score_max, sumexp, acc_o = self._update_s2_tile(
                q_fp32,
                k_tile,
                self.softmax_scale,
                self.q_type,
                score_max,
                sumexp,
                acc_o,
            )

        return score_max, sumexp, acc_o / sumexp.unsqueeze(1)

    def _calculate_batch_consistency(
        self,
        q_fp32,
        ori_k_fp32,
        cmp_k_fp32,
        sinks,
    ):
        """Follow the NPU S2 split and fixed FD reduction order."""
        ori_s2_size = 0 if ori_k_fp32 is None else ori_k_fp32.shape[0]
        cmp_s2_size = 0 if cmp_k_fp32 is None else cmp_k_fp32.shape[0]
        reduce_size = self._get_batch_consistency_reduce_size(ori_s2_size, cmp_s2_size)

        # Metadata splits ORI and CMP independently, so a reduction block never
        # crosses the boundary between the two KV regions.
        blocks = []
        for k_tensor in (ori_k_fp32, cmp_k_fp32):
            if k_tensor is None:
                continue
            for start in range(0, k_tensor.shape[0], reduce_size):
                blocks.append(k_tensor[start : start + reduce_size])

        merged_lse = None
        merged_sum = None
        merged_o = None
        for block_id, k_block in enumerate(blocks):
            initial_max = (
                sinks.clone()
                if block_id == 0 and sinks is not None
                else torch.full((q_fp32.shape[0],), -torch.inf, dtype=torch.float32)
            )
            local_max, local_sum, local_o = self._calculate_local_s2_block(
                q_fp32, k_block, initial_max
            )
            if merged_lse is None:
                merged_lse = local_max
                merged_sum = local_sum
                merged_o = local_o
                continue

            global_max = torch.max(merged_lse, local_max)
            prev_weight = torch.exp(merged_lse - global_max) * merged_sum
            cur_weight = torch.exp(local_max - global_max) * local_sum
            global_sum = prev_weight + cur_weight
            merged_o = (
                merged_o * prev_weight.unsqueeze(1) + local_o * cur_weight.unsqueeze(1)
            ) / global_sum.unsqueeze(1)

            # FD persists the merged state as (LSE, 1, normalized O), then
            # consumes it in the next step of the deterministic left fold.
            merged_lse = global_max + torch.log(global_sum)
            merged_sum = torch.ones_like(global_sum)

        if merged_lse is None:
            raise ValueError("batch-consistency calculation requires non-empty KV")
        return merged_o, merged_lse + torch.log(merged_sum)

    def calculate_by_bnsd(
        self,
        q_bnsd,
        ori_k_bnsd,
        cmp_k_bnsd,
        ori_sparse_indices_bnsd,
        cmp_sparse_indices_bnsd,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        ori_topk_length_bnsd,
        cmp_topk_length_bnsd,
        return_softmax_lse=False,
        batch_consistency=False,
    ):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        softmax_lse = None
        if return_softmax_lse:
            softmax_lse = torch.zeros(
                (
                    q_bnsd.shape[0],
                    ori_k_bnsd.shape[1],
                    q_bnsd.shape[2],
                    q_bnsd.shape[1] // ori_k_bnsd.shape[1],
                ),
                dtype=torch.float32,
            )
        B = q_bnsd.shape[0]
        act_q = self.seqused_q
        G = int(self.N1 / self.N2)
        s2_base_size = 128
        cmp_restored_lengths = restore_cmp_kv_lengths(
            seqused_cmp_kv, self.cmp_ratio, cmp_residual_kv
        )

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            cur_ori_act_kv = seqused_ori_kv[i_B]
            cur_cmp_restored = (
                cmp_restored_lengths[i_B] if cmp_restored_lengths is not None else 0
            )
            for i_N2 in range(self.N2):
                print(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = sinks[i_N2 * G : (i_N2 + 1) * G]
                cur_sinks_expand = cur_sinks.unsqueeze(1)
                for i_S1 in range(cur_act_q):
                    milestones = [
                        int(cur_act_q * pct / 100) for pct in range(10, 101, 10)
                    ]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        print(
                            f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}"
                        )
                    ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                    if self.ori_mask_mode == 3:
                        ori_win_start = 0
                        ori_win_end = min(max(ori_threshold, 0), cur_ori_act_kv)
                    elif self.ori_mask_mode == 4:
                        if self.ori_win_left == -1:
                            ori_win_start = 0
                        else:
                            ori_win_start = max(
                                ori_threshold - self.ori_win_left - 1, 0
                            )
                        if self.ori_win_right == -1:
                            ori_win_end = cur_ori_act_kv
                        else:
                            ori_win_end = min(
                                max(ori_threshold + self.ori_win_right, 0),
                                cur_ori_act_kv,
                            )
                    else:
                        ori_win_start = 0
                        ori_win_end = cur_ori_act_kv
                    ori_win_start = min(ori_win_start, cur_ori_act_kv)

                    if ori_win_start >= ori_win_end:
                        cur_ori_k_bnsd = torch.zeros(
                            [0, self.D], dtype=ori_k_bnsd.dtype
                        )
                    else:
                        cur_ori_k_bnsd = ori_k_bnsd[
                            i_B, i_N2, ori_win_start:ori_win_end, :
                        ]

                    if (
                        self.template_run_mode == "CSA"
                        and cmp_sparse_indices_bnsd is not None
                    ):
                        topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag, cur_cmp_k = self.gather_cmp_kv(
                            cmp_k_bnsd,
                            topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_cmp_restored,
                            cur_act_q,
                            cmp_topk_length_bnsd,
                        )
                    elif self.template_run_mode == "HCA":
                        empty_flag, cur_cmp_k = self.mask_cmp_kv(
                            cmp_k_bnsd, i_B, i_N2, i_S1, cur_cmp_restored, cur_act_q
                        )
                    elif (
                        self.template_run_mode == "ORI_SPARSE"
                        and ori_sparse_indices_bnsd is not None
                    ):
                        topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag, cur_ori_k = self.gather_ori_kv(
                            ori_k_bnsd,
                            topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_ori_act_kv,
                            cur_act_q,
                            ori_topk_length_bnsd,
                        )
                        if empty_flag is not True:
                            cur_ori_k_bnsd = cur_ori_k
                        cur_cmp_k = []
                        empty_flag = True
                    elif (
                        self.template_run_mode == "ORI_CMP_SPARSE"
                        and ori_sparse_indices_bnsd is not None
                        and cmp_sparse_indices_bnsd is not None
                    ):
                        ori_topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        ori_empty_flag, cur_ori_k = self.gather_ori_kv(
                            ori_k_bnsd,
                            ori_topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_ori_act_kv,
                            cur_act_q,
                            ori_topk_length_bnsd,
                        )
                        cmp_topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        cmp_empty_flag, cur_cmp_k = self.gather_cmp_kv(
                            cmp_k_bnsd,
                            cmp_topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_cmp_restored,
                            cur_act_q,
                            cmp_topk_length_bnsd,
                        )
                        if ori_empty_flag is not True:
                            cur_ori_k_bnsd = cur_ori_k
                        if cmp_empty_flag is True:
                            cur_cmp_k = []
                        empty_flag = cmp_empty_flag
                    else:
                        empty_flag = True
                        cur_cmp_k = []
                    if cur_cmp_k == []:
                        cmp_s2_loop_time = 0
                        cur_cmp_k_fp32 = []
                    else:
                        cmp_s2_loop_time = math.ceil(cur_cmp_k.size(0) / s2_base_size)
                        cur_cmp_k_fp32 = cur_cmp_k.to(dtype=torch.float32)

                    if cur_ori_k_bnsd.size(0) == 0 and cur_cmp_k == []:
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = torch.zeros(
                            [G, self.D], dtype=torch.float
                        )
                        continue

                    cur_attn_out = attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :]
                    q_curr = q_bnsd[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)
                    if batch_consistency:
                        ori_k_for_reduce = (
                            None
                            if cur_ori_k_bnsd.size(0) == 0
                            else cur_ori_k_bnsd.to(dtype=torch.float32)
                        )
                        cmp_k_for_reduce = (
                            None
                            if cur_cmp_k == []
                            else cur_cmp_k.to(dtype=torch.float32)
                        )
                        acc_o, final_lse = self._calculate_batch_consistency(
                            q_curr_fp32,
                            ori_k_for_reduce,
                            cmp_k_for_reduce,
                            cur_sinks,
                        )
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = acc_o.to(
                            dtype=q_bnsd.dtype
                        )
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = final_lse
                        continue

                    if RUN_MODE == 0:
                        if empty_flag:
                            k_concat = cur_ori_k_bnsd
                        else:
                            k_concat = torch.concat([cur_ori_k_bnsd, cur_cmp_k], dim=0)
                        k_concat_fp32 = k_concat.to(dtype=torch.float32)
                        v_concat_fp32 = k_concat_fp32.clone()

                        mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                        scale_res = mm1_res * self.softmax_scale
                        softmax_res, x_max, softmax_sum = self.sinks_softmax(
                            scale_res, cur_sinks_expand
                        )
                        softmax_res = softmax_res.to(q_bnsd.dtype).to(torch.float32)
                        mm2_res = torch.matmul(softmax_res, v_concat_fp32)
                        v2_res = torch.div(mm2_res, softmax_sum)
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = v2_res
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = x_max[:, 0] + torch.log(
                                softmax_sum[:, 0] + 1e-10
                            )
                    elif RUN_MODE == 1:
                        ori_s2_loop_time = math.ceil(
                            cur_ori_k_bnsd.size(0) / s2_base_size
                        )
                        total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time
                        cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                        row_sum = torch.empty((G), dtype=torch.float32).uniform_(
                            1.0, 1.0
                        )
                        row_max = torch.empty((G, 1), dtype=torch.float32)
                        row_max = cur_sinks

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time:  # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size : (i_S2 + 1) * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[
                                        i_S2 * s2_base_size :, :
                                    ]
                            else:  # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size : (
                                            i_S2 - ori_s2_loop_time + 1
                                        )
                                        * s2_base_size,
                                        :,
                                    ]
                                else:
                                    k_tile = cur_cmp_k_fp32[
                                        (i_S2 - ori_s2_loop_time) * s2_base_size :, :
                                    ]
                            v_tile = k_tile.clone()
                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = (
                                mm1_res * self.softmax_scale
                            )  # 外层for S1 循环，据实拷入数据，因此不需要mask

                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            # row_max_tmp = row_max_tmp.unsqueeze(1)
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            update_mul_expand = update_mul.unsqueeze(1)

                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(
                                cur_softmax_res, dim=1
                            )
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(
                                dtype=torch.float
                            )
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            cur_attn_out = cur_attn_out * update_mul_expand + cur_o
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = (
                            cur_attn_out / row_sum_expand
                        ).to(dtype=q_bnsd.dtype)
                        if return_softmax_lse:
                            softmax_lse[i_B, i_N2, i_S1, :] = row_max + torch.log(
                                row_sum + 1e-10
                            )
        return attn_out, softmax_lse

    def gather_ori_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        cur_act_kv,
        cur_act_q,
        ori_topk_length_bnsd,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
        left_bound = 0
        right_bound = cur_act_kv
        if self.ori_mask_mode == 3:
            left_bound = 0
            right_bound = min(max(ori_threshold, 0), cur_act_kv)
        elif self.ori_mask_mode == 4:
            if self.ori_win_left == -1:
                left_bound = 0
            else:
                left_bound = max(ori_threshold - self.ori_win_left - 1, 0)
            if self.ori_win_right == -1:
                right_bound = cur_act_kv
            else:
                right_bound = min(
                    max(ori_threshold + self.ori_win_right, 0), cur_act_kv
                )
        elif self.ori_mask_mode == 0:
            pass

        left_bound = min(left_bound, cur_act_kv)

        # 与kernel逻辑保持一致: topkLen中的值与sparseIndices.Dim[-1]取最小
        valid_count = min(topk_id.shape[0], right_bound)
        if ori_topk_length_bnsd is not None:
            valid_count = min(int(ori_topk_length_bnsd[i_B, 0, i_S1, 0]), valid_count)

        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]
            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = min(begin_idx + sparse_block_size, cur_act_kv)
            if begin_idx >= right_bound:
                continue
            if begin_idx < left_bound:
                continue
            if end_idx <= right_bound:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, right_bound))
        empty_flag = len(s2_sparse) == 0
        k_sparse = k_tensor[i_B, i_N2, s2_sparse, :] if not empty_flag else []
        return empty_flag, k_sparse

    def gather_cmp_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        cur_act_kv,
        cur_act_q,
        cmp_topk_length_bnsd=None,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        cur_cmp_act_kv = math.floor(cur_act_kv / self.cmp_ratio)
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / self.cmp_ratio)
        elif self.cmp_mask_mode == 0:
            threshold = cur_cmp_act_kv
        # 与kernel逻辑保持一致: topkLen中的值与sparseIndices.Dim[-1]取最小
        if cmp_topk_length_bnsd is not None:
            valid_count = min(
                int(cmp_topk_length_bnsd[i_B, 0, i_S1, 0]),
                topk_id.shape[0],
                math.ceil(threshold / sparse_block_size),
            )
        else:
            valid_count = min(self.K, math.ceil(threshold / sparse_block_size))
        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = (
                begin_idx + sparse_block_size
                if begin_idx + sparse_block_size <= cur_cmp_act_kv
                else cur_cmp_act_kv
            )
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            k_sparse = []
            empty_flag = True
        else:
            k_sparse = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, k_sparse

    def mask_cmp_kv(self, k_tensor, i_B, i_N2, i_S1, cur_act_kv, cur_act_q):
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = (cur_act_kv - cur_act_q + i_S1 + 1) // self.cmp_ratio
        elif self.cmp_mask_mode == 0:
            threshold = cur_act_kv // self.cmp_ratio
        empty_flag = True
        k_sparse = []
        if threshold > 0:
            empty_flag = False
            k_sparse = k_tensor[i_B, i_N2, :threshold, :]
        return empty_flag, k_sparse

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        return y, x_max, x_sum

    def trans_shape_to_bnsd(
        self, tensor, shape, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = shape[2]
            B = len(cu_seqlens_q) - 1
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_seqused, :] = tensor[
                        t_start : t_start + cur_seqused, n_index, :
                    ]
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_topk_length_shape_to_bnsd(
        self, tensor, shape, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            tensor = tensor.reshape(B, 1, S, 1)
            return tensor, [B, 1, S, 1]
        elif layout in ["TND"]:
            T = shape[0]
            B = len(cu_seqlens_q) - 1
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            new_tensor = torch.zeros((B, 1, max_s1, 1), dtype=tensor.dtype)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                # print("======mark new_tensor[b_index, 0, 0:cur_seqused, :].shape=", new_tensor[b_index, 0, 0:cur_seqused, :].shape)
                new_tensor[b_index, 0, 0:cur_seqused, :] = tensor[
                    t_start : t_start + cur_seqused, :
                ]
            return new_tensor, [B, 1, max_s1, 1]

    def trans_bnsd_to_target_layout(
        self, tensor, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = cu_seqlens_q[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            output = torch.zeros((T, N, D), dtype=torch.float)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                for n_index in range(N):
                    output[t_start : t_start + cur_seqused, n_index, :] = tensor[
                        b_index, n_index, :cur_seqused, :
                    ]
            return output
        else:
            return tensor

    def lse_trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:  # B N S G
            return tensor
        elif layout in ["TND"]:  # B N2 S1 G  --> T1 N2 G --> N2 T1 G
            T = act_seq[-1]
            B = tensor.shape[0]
            N2 = tensor.shape[1]
            G = tensor.shape[3]
            output = torch.zeros((N2, T, G), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(
                act_seq
            )  # prefix_sum_to_original 还原成每个batch的真实长度
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N2):
                    output[n_index, t_start:t_end, :] = tensor[
                        b_index, n_index, :cur_act_seq, :
                    ]
                t_start += cur_act_seq
            output = output.transpose(0, 1).contiguous()
            return output
        else:
            return tensor

    def forward(
        self,
        q,
        ori_k_bnsd,
        cmp_k_bnsd,
        ori_sparse_indices,
        cmp_sparse_indices,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        ori_topk_length,
        cmp_topk_length,
        return_softmax_lse,
        batch_consistency=False,
    ):
        print("cpu执行中...")
        print(f"template_run_mode = {self.template_run_mode}")

        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(
            q, q.shape, self.layout_q, cu_seqlens_q, self.seqused_q
        )

        ori_sparse_indices_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            and ori_sparse_indices is not None
        ):
            ori_sparse_indices_bnsd, _ = self.trans_shape_to_bnsd(
                ori_sparse_indices,
                ori_sparse_indices.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        cmp_sparse_indices_bnsd = None
        if (
            self.template_run_mode in ("CSA", "ORI_CMP_SPARSE")
            and cmp_sparse_indices is not None
        ):
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = (
                self.trans_shape_to_bnsd(
                    cmp_sparse_indices,
                    cmp_sparse_indices.shape,
                    self.layout_q,
                    cu_seqlens_q,
                    self.seqused_q,
                )
            )

        ori_topk_length_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            and ori_topk_length is not None
        ):
            ori_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                ori_topk_length,
                ori_topk_length.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        cmp_topk_length_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE", "CSA")
            and cmp_topk_length is not None
        ):
            cmp_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                cmp_topk_length,
                cmp_topk_length.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        attn_out, softmax_lse = self.calculate_by_bnsd(
            q_bnsd,
            ori_k_bnsd,
            cmp_k_bnsd,
            ori_sparse_indices_bnsd,
            cmp_sparse_indices_bnsd,
            cu_seqlens_q,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            sinks,
            ori_topk_length_bnsd,
            cmp_topk_length_bnsd,
            return_softmax_lse,
            batch_consistency,
        )

        attn_out = self.trans_bnsd_to_target_layout(
            attn_out, self.layout_q, cu_seqlens_q, self.seqused_q
        )
        if return_softmax_lse:
            softmax_lse = self.lse_trans_bnsd_to_target_layout(
                softmax_lse, self.layout_q, cu_seqlens_q
            )
        return attn_out, softmax_lse


def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(
            f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}"
        )

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor


def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q) - 1):
        current_diff = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff


def gen_sparse_indices_bsnd(
    cmp_ratio,
    B,
    S1,
    N2,
    K,
    seqused_q,
    seqused_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support fullK, random, which is {kv_topk_mode}"
        )

    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)

    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((B, S1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        cur_act_q = seqused_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"topklen sparse mask mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)
                # gen sparse indices
                if sparse_indices_mode == "random":
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[i_B, i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                # gen topk length
                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[i_B, i_S1, i_N2] = min(cur_valid_s2_max_update, K)
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[i_B, i_S1, i_N2] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )

    return sparse_data, topk_length


def gen_sparse_indices_tnd(
    cmp_ratio,
    B,
    T1,
    N2,
    K,
    cu_seqlens_q,
    seqused_q,
    seqused_ori_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support fullK, random, which is {kv_topk_mode}"
        )

    sparse_data = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)

    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((T1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        cur_act_q = seqused_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_ori_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"ori_mask_mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)
                # gen sparse indices
                if sparse_indices_mode == "random":
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                # gen topk length
                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[s1_prefix + i_S1, i_N2] = min(
                            cur_valid_s2_max_update, K
                        )
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[s1_prefix + i_S1, i_N2] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )

    return sparse_data, topk_length


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
    fraction_int = int(
        x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent)
        - pow(2, FP32_FRACTION_BITS)
    )
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = (
        get_hif8_fraction_bits_number(exponent)
    )
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    else:
        print("unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = (
            get_hif8_fraction_bits_number(exponent)
        )
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
        hif8_int_value = (
            sign_int_value
            + dot_int_value
            + sig_exp
            + exponent_int_value
            + hif8_frac_value
        )
    return hif8_int_value


def trans_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    tensor_shape = in_tensor.shape
    tensor_shape_size = in_tensor.numel()
    if tensor_shape_size == 1.0:
        tensor_shape_size = int(tensor_shape_size)

    out_tensor = torch.zeros(tensor_shape_size).to(torch.uint8)
    in_tensor = in_tensor.reshape(tensor_shape_size)
    for i in range(tensor_shape_size):
        out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.view(torch.uint8)
    out_tensor = out_tensor.reshape(tensor_shape)
    return out_tensor


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


def trans_hifuint8_tensor_to_float(in_tensor):
    tensor_shape = in_tensor.shape
    tensor_shape_size = in_tensor.numel()

    out_tensor = torch.zeros(tensor_shape_size).to(torch.float)
    in_tensor = in_tensor.reshape(tensor_shape_size)
    for i in range(tensor_shape_size):
        out_tensor[i] = cvt_hifuint8_to_float32(in_tensor[i])
    out_tensor = out_tensor.reshape(tensor_shape).to(torch.float)
    return out_tensor


def trans_kv_bnsd_to_tnd(
    kv_bnsd_npu, cu_seqlens_kv, seqused_kv, B, N2, d_aligned_32, kv_type
):
    total_t = int(cu_seqlens_kv[-1])
    kv_tnd = torch.zeros((total_t, N2, d_aligned_32), dtype=kv_type)
    for i_B in range(B):
        t_start = int(cu_seqlens_kv[i_B])
        cur_s = int(seqused_kv[i_B])
        if cur_s > 0:
            kv_tnd[t_start : t_start + cur_s, :, :] = kv_bnsd_npu[
                i_B, :, :cur_s, :
            ].permute(1, 0, 2)
    return kv_tnd


def rope_as_kv_bytes(rope, kv_type):
    """Reinterpret BF16 RoPE bytes, including an absent zero-width RoPE input."""
    if rope.shape[-1] == 0:
        return torch.empty(rope.shape, dtype=kv_type, device=rope.device)
    return rope.view(kv_type)


def gen_ori_kv(
    q_type,
    ori_kv_type,
    B,
    S1,
    T1,
    N2,
    rope_head_dim,
    nope_head_dim,
    tile_size,
    quant_scale_head_dim,
    d_aligned_32,
    pad_d,
    block_num1,
    block_size1,
    ori_max_s2,
    ori_max_block_num_per_batch,
    seqused_ori_kv,
    cu_seqlens_ori_kv,
    quant_param_range_left,
    quant_param_range_right,
    quant_mode,
    layout_kv="PA_BBND",
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    K1=None,
    template_run_mode=None,
    layout_q=None,
    cu_seqlens_q=None,
    seqused_q=None,
    ori_mask_mode=0,
    ori_sparse_indices_mode="full",
    ori_kv_topk_mode="no",
    ori_topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if quant_mode == 10:
        quant_param = random.uniform(quant_param_range_left, quant_param_range_right)
        quant_range_left = quant_param
        quant_range_right = quant_param
    else:
        quant_range_left = quant_param_range_left
        quant_range_right = quant_param_range_right
    ori_kv_quant_param_tensor_npu = torch.tensor(
        np.random.uniform(
            quant_range_left,
            quant_range_right,
            (B, N2, ori_max_s2, quant_scale_head_dim),
        )
    ).to(q_type)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)

    if quant_mode == 10:
        ori_k_nope_bnsd_npu = torch.tensor(
            np.random.uniform(
                data_range_left, data_range_right, (B, N2, ori_max_s2, nope_head_dim)
            )
        ).to(torch.float)
        ori_k_nope_bnsd_npu = trans_float_tensor_to_hifuint8(
            ori_k_nope_bnsd_npu, round_mode="hybrid", over_mode=True
        )
        ori_k_nope_bnsd = trans_hifuint8_tensor_to_float(ori_k_nope_bnsd_npu).to(q_type)
    else:
        ori_k_nope_bnsd_npu = torch.tensor(
            np.random.uniform(
                data_range_left, data_range_right, (B, N2, ori_max_s2, nope_head_dim)
            )
        ).to(torch.float8_e4m3fn)
        ori_k_nope_bnsd = ori_k_nope_bnsd_npu.to(q_type)

    ori_k_rope_bnsd = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, ori_max_s2, rope_head_dim)
        )
    ).to(q_type)

    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_bnsd[:, :, :, offset : offset + 1] = torch.mul(
                ori_k_nope_bnsd[:, :, :, offset : offset + 1],
                ori_kv_quant_param_tensor[:, :, :, d_loop : d_loop + 1],
            )
    ori_k_bnsd = torch.concat([ori_k_nope_bnsd, ori_k_rope_bnsd], dim=3)

    ori_pad_tensor = torch.tensor(
        np.random.uniform(0, 0, (B, N2, ori_max_s2, pad_d))
    ).to(torch.float8_e8m0fnu)
    ori_k_bnsd_npu = torch.concat(
        [
            rope_as_kv_bytes(ori_k_rope_bnsd, ori_kv_type),
            ori_k_nope_bnsd_npu,
            ori_kv_quant_param_tensor_npu.view(ori_kv_type),
            ori_pad_tensor.view(ori_kv_type),
        ],
        dim=3,
    )

    if layout_kv == "TND":
        ori_k_in_pa_shape = trans_kv_bnsd_to_tnd(
            ori_k_bnsd_npu,
            cu_seqlens_ori_kv,
            seqused_ori_kv,
            B,
            N2,
            d_aligned_32,
            ori_kv_type,
        )
        ori_block_table = None
    elif layout_kv == "BSND":
        ori_block_table = None
        ori_k_in_pa_shape = (
            ori_k_bnsd_npu.squeeze(1)
            .reshape(B, ori_max_s2, N2, ori_k_bnsd_npu.shape[3])
            .contiguous()
        )
    else:
        ori_block_num_per_batch = []
        ori_block_num_sum = 0

        for cur_ori_act_kv in seqused_ori_kv:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num

        if block_num1 < ori_block_num_sum:
            raise ValueError(
                f"ori_kv actual_block_num < needed_block_num, which is {block_num1 < ori_block_num_sum}"
            )

        ori_block_id_list = np.arange(block_num1)
        ori_block_id_list = np.random.permutation(ori_block_id_list).astype(np.int32)
        cur_block_id = 0
        ori_block_table = np.full(
            (B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        batch_idx = 0
        for cur_block_id_threshold in ori_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                ori_block_table[batch_idx][i_block_id] = ori_block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        ori_k_expand = torch.zeros(
            (B, N2, ori_max_block_num_per_batch * block_size1, d_aligned_32),
            dtype=ori_kv_type,
        )
        ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd_npu
        ori_k_in_pa_shape = torch.zeros(
            (block_num1, block_size1, N2, d_aligned_32), dtype=ori_kv_type
        )

        for i_B in range(B):
            for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
                block_start_pos = i_block * block_size1
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = ori_k_expand[
                            i_B,
                            i_N2,
                            block_start_pos : block_start_pos + block_size1,
                            :,
                        ]
        ori_block_table = torch.tensor(ori_block_table).to(torch.int32)

    ori_sparse_indices = None
    ori_topk_length = None
    if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE") and K1 is not None:
        if layout_q == "BSND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_bsnd(
                1,
                B,
                S1,
                N2,
                K1,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
        elif layout_q == "TND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_tnd(
                1,
                B,
                T1,
                N2,
                K1,
                cu_seqlens_q,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
    return (
        ori_k_bnsd,
        ori_k_in_pa_shape,
        ori_block_table,
        ori_sparse_indices,
        ori_topk_length,
    )


# kv_quant_2 目前支持PA
def gen_ori_kv_quant_2_pa(
    q_type,
    ori_kv_type,
    B,
    S1,
    T1,
    N2,
    rope_head_dim,
    nope_head_dim,
    tile_size,
    quant_scale_head_dim,
    pad_d,
    block_num1,
    block_size1,
    ori_max_s2,
    ori_max_block_num_per_batch,
    seqused_ori_kv,
    quant_param_range_left,
    quant_param_range_right,
    d_combined_quant_2,
    layout_kv="PA_BBND",
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    K1=None,
    template_run_mode=None,
    layout_q=None,
    cu_seqlens_q=None,
    seqused_q=None,
    ori_mask_mode=0,
    ori_sparse_indices_mode="full",
    ori_kv_topk_mode="no",
    ori_topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    # 1. 生成并处理 Nope (448) 和 Rope (64) -> Feature (512)
    feature_bytes = nope_head_dim + 2 * rope_head_dim
    metadata_bytes = quant_scale_head_dim + pad_d
    ori_k_nope_bnsd_npu = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, ori_max_s2, nope_head_dim)
        )
    ).to(torch.float8_e4m3fn)
    ori_k_nope_bnsd = ori_k_nope_bnsd_npu.to(q_type)
    ori_k_rope_bnsd = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, ori_max_s2, rope_head_dim)
        )
    ).to(q_type)
    # 2. 生成 Scale (7) 和 Padding (1) -> Metadata (8)
    ori_kv_quant_param_tensor_npu = torch.tensor(
        np.random.uniform(
            quant_param_range_left,
            quant_param_range_right,
            (B, N2, ori_max_s2, quant_scale_head_dim),
        )
    ).to(torch.float8_e8m0fnu)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)
    ori_pad_tensor = torch.zeros((B, N2, ori_max_s2, pad_d)).to(torch.float8_e8m0fnu)
    # 3. nope部分*scale，转成fp8，保存为bin文件，再转回bf16
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_bnsd[:, :, :, offset : offset + 1] = torch.mul(
                ori_k_nope_bnsd[:, :, :, offset : offset + 1],
                ori_kv_quant_param_tensor[:, :, :, d_loop : d_loop + 1],
            )
    ori_k_bnsd = torch.concat([ori_k_nope_bnsd, ori_k_rope_bnsd], dim=3)

    # 4. 生成blockTable: Block映射逻辑 (保持不变)
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    for cur_ori_act_kv in seqused_ori_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num
    ori_block_id_list = np.random.permutation(np.arange(block_num1)).astype(
        np.int32
    )  # 生成随机映射
    ori_block_table = np.full(
        (B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32
    )  # 初始化blockTable
    cur_block_id = 0
    for b in range(B):
        num = ori_block_num_per_batch[b]
        ori_block_table[b, :num] = ori_block_id_list[cur_block_id : cur_block_id + num]
        cur_block_id += num

    ori_k_in_pa_shape = torch.zeros(
        (block_num1, block_size1, N2, d_combined_quant_2 + pad_d), dtype=ori_kv_type
    )
    for i_B in range(B):
        for i_block, cur_phys_block_id in enumerate(ori_block_table[i_B]):
            if cur_phys_block_id == -1:
                continue

            # 计算该 Block 在逻辑序列中的起始 Token 位置
            start_s = i_block * block_size1
            end_s = start_s + block_size1

            # 计算实际有效的长度（处理边界）
            actual_end_s = min(end_s, ori_max_s2)
            valid_len = actual_end_s - start_s
            if valid_len <= 0:
                continue

            # --- 填充 Feature 部分 (0:feature_bytes) ---
            # 排布：block_size * (nope + rope)
            feat_nope = ori_k_nope_bnsd_npu[
                i_B, :, start_s:actual_end_s, :
            ]  # [N, S, 448]
            # 关键点：将 Rope (BF16) view 为 FP8 格式，长度从 64 变为 128
            feat_rope_raw = ori_k_rope_bnsd[
                i_B, :, start_s:actual_end_s, :
            ].contiguous()
            feat_rope_fp8 = rope_as_kv_bytes(feat_rope_raw, torch.float8_e4m3fn)  # [N, S, 128]

            feat_all = torch.concat([feat_nope, feat_rope_fp8], dim=-1)  # [N, S, feature_bytes]

            # 写入物理内存：前 block_size * feature_bytes 字节
            # 为了实现 block_size 连排，需要将 [N, S, feature_bytes] 转为 [N, S*feature_bytes]
            feat_flat = feat_all.view(N2, -1)
            # 计算在物理块中的起始偏移
            ori_k_in_pa_shape.permute(0, 2, 1, 3).view(block_num1, N2, -1)[
                cur_phys_block_id, :, 0 : valid_len * feature_bytes
            ] = feat_flat

            # --- B. 准备 Metadata 数据 [N, S, 8] ---
            meta_scale = ori_kv_quant_param_tensor_npu[
                i_B, :, start_s:actual_end_s, :
            ].view(torch.float8_e4m3fn)
            meta_pad = ori_pad_tensor[i_B, :, start_s:actual_end_s, :].view(
                torch.float8_e4m3fn
            )

            meta_all = torch.concat([meta_scale, meta_pad], dim=-1)  # [N, S, 8]
            # print("meta_all: ", meta_all)
            meta_flat = meta_all.view(N2, -1)
            # 写入物理内存：从 block_size * feature_bytes 字节处开始
            metadata_start_offset = block_size1 * feature_bytes
            ori_k_in_pa_shape.permute(0, 2, 1, 3).view(block_num1, N2, -1)[
                cur_phys_block_id,
                :,
                metadata_start_offset : metadata_start_offset + valid_len * metadata_bytes,
            ] = meta_flat
    ori_block_table = torch.tensor(ori_block_table).to(torch.int32)
    ori_topk_length = None
    ori_sparse_indices = None
    if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE") and K1 is not None:
        if layout_q == "BSND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_bsnd(
                1,
                B,
                S1,
                N2,
                K1,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
        elif layout_q == "TND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_tnd(
                1,
                B,
                T1,
                N2,
                K1,
                cu_seqlens_q,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
    return (
        ori_k_bnsd,
        ori_k_in_pa_shape,
        ori_block_table,
        ori_sparse_indices,
        ori_topk_length,
    )


def gen_cmp_kv_quant_2_pa(
    q_type,
    cmp_kv_type,
    B,
    S1,
    T1,
    N2,
    K,
    rope_head_dim,
    nope_head_dim,
    tile_size,
    quant_scale_head_dim,
    d_combined_quant_2,
    pad_d,
    block_num2,
    block_size2,
    cmp_max_s2,
    cmp_max_block_num_per_batch,
    layout_q,
    cu_seqlens_q,
    seqused_q,
    seqused_cmp_kv,
    cmp_residual_kv,
    cmp_ratio,
    cmp_mask_mode,
    template_run_mode,
    quant_param_range_left,
    quant_param_range_right,
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    cmp_kv_topk_mode="no",
    cmp_sparse_indices_mode="full",
    cmp_topk_length_override=None,
):
    feature_bytes = nope_head_dim + 2 * rope_head_dim
    metadata_bytes = quant_scale_head_dim + pad_d
    if cmp_max_s2 == 0:
        return None, None, None, None
    # --- 1. 生成原始数据 ---
    # 量化参数 (7字节)
    cmp_kv_quant_param_tensor_npu = torch.tensor(
        np.random.uniform(
            quant_param_range_left,
            quant_param_range_right,
            (B, N2, cmp_max_s2, quant_scale_head_dim),
        )
    ).to(torch.float8_e8m0fnu)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)

    # Nope 部分 (448字节, FP8)
    cmp_k_nope_bnsd_npu = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, cmp_max_s2, nope_head_dim)
        )
    ).to(torch.float8_e4m3fn)

    # Rope 部分 (64个元素, BF16/FP16)
    cmp_k_rope_bnsd_npu = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, cmp_max_s2, rope_head_dim)
        )
    ).to(q_type)

    # 模拟量化计算 (用于生成golden计算数据)
    cmp_k_nope_bnsd = cmp_k_nope_bnsd_npu.to(q_type)
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_bnsd[:, :, :, offset : offset + 1] = torch.mul(
                cmp_k_nope_bnsd[:, :, :, offset : offset + 1],
                cmp_kv_quant_param_tensor[:, :, :, d_loop : d_loop + 1],
            )
    # 逻辑上的 K (用于对比)
    cmp_k_bnsd = torch.concat([cmp_k_nope_bnsd, cmp_k_rope_bnsd_npu], dim=3)

    # Padding 部分 (1字节)
    cmp_pad_tensor = torch.zeros((B, N2, cmp_max_s2, pad_d)).to(torch.float8_e8m0fnu)

    # --- 2. 计算 Block 映射 ---
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0
    for cur_cmp_act_kv in seqused_cmp_kv:
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num

    if block_num2 < cmp_block_num_sum:
        raise ValueError("cmp_kv actual_block_num < needed_block_num")

    cmp_block_id_list = np.random.permutation(np.arange(block_num2)).astype(np.int32)
    cmp_block_table = np.full(
        (B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32
    )
    cur_block_id_idx = 0
    for b in range(B):
        for i in range(cmp_block_num_per_batch[b]):
            cmp_block_table[b][i] = cmp_block_id_list[cur_block_id_idx]
            cur_block_id_idx += 1

    # --- 3. 实现 [block_size*feature_bytes + block_size*8] 排布 ---
    total_bytes_per_head_block = block_size2 * (feature_bytes + metadata_bytes)
    cmp_k_in_pa_shape = torch.zeros(
        (block_num2, N2, total_bytes_per_head_block), dtype=cmp_kv_type
    )

    for i_B in range(B):
        for i_block, cur_phys_block_id in enumerate(cmp_block_table[i_B]):
            if cur_phys_block_id == -1:
                continue

            start_s = i_block * block_size2
            end_s = start_s + block_size2
            actual_end_s = min(end_s, cmp_max_s2)
            valid_len = actual_end_s - start_s
            if valid_len <= 0:
                continue

            # --- A. 准备 Feature 数据 (nope + rope) ---
            # nope: [N, S, 448]
            f_nope = cmp_k_nope_bnsd_npu[i_B, :, start_s:actual_end_s, :]
            # rope: [N, S, 64] BF16 -> view 为 [N, S, 128] FP8
            f_rope = rope_as_kv_bytes(
                cmp_k_rope_bnsd_npu[i_B, :, start_s:actual_end_s, :].contiguous(),
                torch.float8_e4m3fn,
            )

            # 拼接成 [N, S, feature_bytes]
            feat_all = torch.concat([f_nope, f_rope], dim=-1)

            # --- B. 准备 Metadata 数据 (scale + pad) ---
            m_scale = cmp_kv_quant_param_tensor_npu[
                i_B, :, start_s:actual_end_s, :
            ].view(torch.float8_e4m3fn)
            m_pad = cmp_pad_tensor[i_B, :, start_s:actual_end_s, :].view(
                torch.float8_e4m3fn
            )

            # 拼接成 [N, S, 8]
            meta_all = torch.concat([m_scale, m_pad], dim=-1)

            # --- C. 写入物理内存 ---
            # 按照要求的 Planar 布局：Feature 块在前，Metadata 块在后
            # 对于每个 Head N2：
            for head_idx in range(N2):
                # 写入 Feature: 前 block_size * feature_bytes 字节
                # 将该 head 下有效 token 的 feature_bytes 字节拉平写入
                cmp_k_in_pa_shape[cur_phys_block_id, head_idx, 0 : valid_len * feature_bytes] = (
                    feat_all[head_idx].reshape(-1)
                )

                # 写入 Metadata: 起始偏移量为 block_size * feature_bytes
                meta_offset = block_size2 * feature_bytes
                cmp_k_in_pa_shape[
                    cur_phys_block_id,
                    head_idx,
                    meta_offset : meta_offset + valid_len * metadata_bytes,
                ] = meta_all[head_idx].reshape(-1)

    cmp_k_in_pa_shape = cmp_k_in_pa_shape.reshape(
        block_num2, block_size2, N2, d_combined_quant_2 + pad_d
    )
    cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)

    # --- 4. 生成 Sparse Indices ---
    cmp_sparse_indices = None
    cmp_topk_length = None
    if template_run_mode in ("CSA", "ORI_CMP_SPARSE") and cmp_max_s2 != 0:
        cmp_restored_lengths = restore_cmp_kv_lengths(
            seqused_cmp_kv, cmp_ratio, cmp_residual_kv
        )
        if layout_q == "BSND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_bsnd(
                cmp_ratio,
                B,
                S1,
                N2,
                K,
                seqused_q,
                cmp_restored_lengths,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )
        elif layout_q == "TND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_tnd(
                cmp_ratio,
                B,
                T1,
                N2,
                K,
                cu_seqlens_q,
                seqused_q,
                cmp_restored_lengths,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )

    return (
        cmp_k_bnsd,
        cmp_k_in_pa_shape,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_topk_length,
    )


def gen_cmp_kv(
    q_type,
    layout_q,
    cmp_kv_type,
    B,
    S1,
    T1,
    N2,
    D,
    K,
    rope_head_dim,
    nope_head_dim,
    tile_size,
    quant_scale_head_dim,
    d_aligned_32,
    pad_d,
    block_num2,
    block_size2,
    cmp_max_s2,
    cmp_max_block_num_per_batch,
    cu_seqlens_q,
    seqused_q,
    seqused_cmp_kv,
    cu_seqlens_cmp_kv,
    cmp_residual_kv,
    cmp_ratio,
    cmp_mask_mode,
    template_run_mode,
    quant_param_range_left,
    quant_param_range_right,
    quant_mode,
    layout_kv="PA_BBND",
    data_range_left=DATA_RANGE_LEFT,
    data_range_right=DATA_RANGE_RIGHT,
    cmp_kv_topk_mode="no",
    cmp_sparse_indices_mode="full",
    cmp_topk_length_override=None,
):
    if cmp_max_s2 == 0:
        return None, None, None, None, None
    if quant_mode == 10:
        quant_param = random.uniform(quant_param_range_left, quant_param_range_right)
        quant_range_left = quant_param
        quant_range_right = quant_param
    else:
        quant_range_left = quant_param_range_left
        quant_range_right = quant_param_range_right
    cmp_kv_quant_param_tensor_npu = torch.tensor(
        np.random.uniform(
            quant_range_left,
            quant_range_right,
            (B, N2, cmp_max_s2, quant_scale_head_dim),
        )
    ).to(q_type)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)

    if quant_mode == 10:
        cmp_k_nope_bnsd_npu = torch.tensor(
            np.random.uniform(
                data_range_left, data_range_right, (B, N2, cmp_max_s2, nope_head_dim)
            )
        ).to(torch.float)
        cmp_k_nope_bnsd_npu = trans_float_tensor_to_hifuint8(
            cmp_k_nope_bnsd_npu, round_mode="hybrid", over_mode=True
        )
        cmp_k_nope_bnsd = trans_hifuint8_tensor_to_float(cmp_k_nope_bnsd_npu).to(q_type)
    else:
        cmp_k_nope_bnsd_npu = torch.tensor(
            np.random.uniform(
                data_range_left, data_range_right, (B, N2, cmp_max_s2, nope_head_dim)
            )
        ).to(torch.float8_e4m3fn)
        cmp_k_nope_bnsd = cmp_k_nope_bnsd_npu.to(q_type)

    cmp_k_rope_bnsd = torch.tensor(
        np.random.uniform(
            data_range_left, data_range_right, (B, N2, cmp_max_s2, rope_head_dim)
        )
    ).to(q_type)

    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_bnsd[:, :, :, offset : offset + 1] = torch.mul(
                cmp_k_nope_bnsd[:, :, :, offset : offset + 1],
                cmp_kv_quant_param_tensor[:, :, :, d_loop : d_loop + 1],
            )
    cmp_k_bnsd = torch.concat([cmp_k_nope_bnsd, cmp_k_rope_bnsd], dim=3)

    cmp_pad_tensor = torch.tensor(
        np.random.uniform(0, 0, (B, N2, cmp_max_s2, pad_d))
    ).to(torch.float8_e8m0fnu)
    cmp_k_bnsd_npu = torch.concat(
        [
            rope_as_kv_bytes(cmp_k_rope_bnsd, cmp_kv_type),
            cmp_k_nope_bnsd_npu,
            cmp_kv_quant_param_tensor_npu.view(cmp_kv_type),
            cmp_pad_tensor.view(cmp_kv_type),
        ],
        dim=3,
    )

    if layout_kv == "TND":
        cmp_k_in_pa_shape = trans_kv_bnsd_to_tnd(
            cmp_k_bnsd_npu,
            cu_seqlens_cmp_kv,
            seqused_cmp_kv,
            B,
            N2,
            d_aligned_32,
            cmp_kv_type,
        )
        cmp_block_table = None
    elif layout_kv == "BSND":
        cmp_block_table = None
        cmp_k_in_pa_shape = (
            cmp_k_bnsd_npu.squeeze(1)
            .reshape(B, cmp_max_s2, N2, cmp_k_bnsd_npu.shape[3])
            .contiguous()
        )
    else:
        cmp_block_num_per_batch = []
        cmp_block_num_sum = 0

        for cur_cmp_act_kv in seqused_cmp_kv:
            cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
            cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
            cmp_block_num_sum += cur_cmp_kv_block_num

        if block_num2 < cmp_block_num_sum:
            raise ValueError(
                f"cmp_kv actual_block_num < needed_block_num, which is {block_num2 < cmp_block_num_sum}"
            )

        cmp_block_id_list = np.arange(block_num2)
        cmp_block_id_list = np.random.permutation(cmp_block_id_list).astype(np.int32)
        cur_block_id = 0
        cmp_block_table = np.full(
            (B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        batch_idx = 0
        for cur_block_id_threshold in cmp_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                cmp_block_table[batch_idx][i_block_id] = cmp_block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        cmp_k_expand = torch.zeros(
            (B, N2, cmp_max_block_num_per_batch * block_size2, d_aligned_32),
            dtype=cmp_kv_type,
        )
        cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k_bnsd_npu
        cmp_k_in_pa_shape = torch.zeros(
            (block_num2, block_size2, N2, d_aligned_32), dtype=cmp_kv_type
        )

        for i_B in range(B):
            for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
                block_start_pos = i_block * block_size2
                if cur_block_id == -1:
                    continue
                else:
                    for i_N2 in range(N2):
                        cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = cmp_k_expand[
                            i_B,
                            i_N2,
                            block_start_pos : block_start_pos + block_size2,
                            :,
                        ]

        cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)

    # generate cmp_sparse_indices
    cmp_sparse_indices = None
    cmp_topk_length = None
    if template_run_mode in ("CSA", "ORI_CMP_SPARSE") and cmp_max_s2 != 0:
        cmp_restored_lengths = restore_cmp_kv_lengths(
            seqused_cmp_kv, cmp_ratio, cmp_residual_kv
        )
        if layout_q == "BSND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_bsnd(
                cmp_ratio,
                B,
                S1,
                N2,
                K,
                seqused_q,
                cmp_restored_lengths,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )
        elif layout_q == "TND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_tnd(
                cmp_ratio,
                B,
                T1,
                N2,
                K,
                cu_seqlens_q,
                seqused_q,
                cmp_restored_lengths,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )
    return (
        cmp_k_bnsd,
        cmp_k_in_pa_shape,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_topk_length,
    )


def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]

    # 生成文件名
    input_filename = f"qsmla_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)

    # 保存数据
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")

    return input_filepath


def gen_data(params, generate_golden=True):
    """
    生成input param及cpuout
    runNpu: 生成完毕后执行npu计算
    return test_data
    """

    Testcase_Name = params["Testcase_Name"]
    layout_q = params["layout_q"]
    layout_kv = params["layout_kv"]
    q_type = params["q_type"]
    ori_kv_type = params["ori_kv_type"]
    cmp_kv_type = params["cmp_kv_type"]
    ori_kv_compute_type = get_kv_compute_dtype(ori_kv_type)
    cmp_kv_compute_type = get_kv_compute_dtype(cmp_kv_type)
    B = params["B"]
    S1 = params["S1"]
    T1 = params["T1"]
    N1 = params["N1"]
    N2 = params["N2"]
    D = params["D"]
    K = params["K"]
    block_num1 = params["block_num1"]
    block_num2 = params["block_num2"]
    block_size1 = params["block_size1"]
    block_size2 = params["block_size2"]
    cu_seqlens_q = params["cu_seqlens_q"]
    seqused_q = params["seqused_q"]
    seqused_ori_kv = params["seqused_ori_kv"]
    seqused_cmp_kv = params["seqused_cmp_kv"]
    cu_seqlens_ori_kv = params["cu_seqlens_ori_kv"]
    cu_seqlens_cmp_kv = params["cu_seqlens_cmp_kv"]
    cmp_residual_kv = params["cmp_residual_kv"]
    softmax_scale = params["softmax_scale"]
    cmp_ratio = params["cmp_ratio"]
    ori_mask_mode = params["ori_mask_mode"]
    cmp_mask_mode = params["cmp_mask_mode"]
    ori_win_left = params["ori_win_left"]
    ori_win_right = params["ori_win_right"]
    quant_mode = params["quant_mode"]
    tile_size = params["tile_size"]
    rope_head_dim = params["rope_head_dim"]
    template_run_mode = params["template_run_mode"]
    topk_value_mode = params.get("topk_value_mode", 1)
    return_softmax_lse = params.get("return_softmax_lse", False)
    batch_consistency = params.get("batch_consistency", False)
    K1 = params.get("K1")
    ori_kv_topk_mode = params.get("ori_kv_topk_mode", "no")
    cmp_kv_topk_mode = params.get("cmp_kv_topk_mode", "no")
    ori_sparse_indices_mode = params.get("ori_sparse_indices_mode", "full")
    cmp_sparse_indices_mode = params.get("cmp_sparse_indices_mode", "full")
    ori_topk_length_override = params.get("ori_topk_length", None)
    cmp_topk_length_override = params.get("cmp_topk_length", None)

    q_datarange, ori_kv_datarange, cmp_kv_datarange = resolve_input_data_ranges(
        params.get("q_datarange"),
        params.get("ori_kv_datarange"),
        params.get("cmp_kv_datarange"),
    )

    if seqused_q is None:
        raise ValueError("seqused_q must not be None")
    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_q = torch.tensor(seqused_q).to(torch.int32)
    seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    seqused_cmp_kv = (
        torch.tensor(seqused_cmp_kv).to(torch.int32)
        if seqused_cmp_kv is not None
        else None
    )

    # convert topk_length override to tensor with proper shape
    if ori_topk_length_override is not None and not isinstance(
        ori_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        elif layout_q == "TND":
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)
    if cmp_topk_length_override is not None and not isinstance(
        cmp_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        elif layout_q == "TND":
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)
    # generate q
    if layout_q == "BSND":
        q = torch.tensor(
            np.random.uniform(q_datarange[0], q_datarange[1], (B, S1, N1, D))
        ).to(q_type)
    elif layout_q == "TND":
        q = torch.tensor(
            np.random.uniform(q_datarange[0], q_datarange[1], (T1, N1, D))
        ).to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(
                f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}"
            )
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    if len(seqused_ori_kv) != B:
        raise ValueError(
            f"len(seqused_ori_kv) != B, which is {len(seqused_ori_kv)} != {B}"
        )
    else:
        ori_max_s2 = int(get_max_adjacent_diff(cu_seqlens_ori_kv))
        ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

        cmp_max_s2 = (
            int(get_max_adjacent_diff(cu_seqlens_cmp_kv))
            if cu_seqlens_cmp_kv is not None
            else 0
        )
        cmp_max_block_num_per_batch = (
            math.ceil(cmp_max_s2 / block_size2) if cmp_max_s2 > 0 else 0
        )

    if quant_mode != 1 and quant_mode != 2:
        raise ValueError(f"input quant_mode = {quant_mode}, only support 1 and 2")

    # 计算kv每个区域D轴长度
    nope_head_dim = D - rope_head_dim
    quant_scale_head_dim = (nope_head_dim + tile_size - 1) // tile_size
    d_aligned_32 = nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim * 2 + 18
    d_combined = nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim * 2
    d_combined_quant_2 = nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim
    print(
        f"d_aligned_32={d_aligned_32}, nope_head_dim={nope_head_dim}, rope_head_dim={rope_head_dim}, quant_scale_head_dim={quant_scale_head_dim}"
    )
    pad_d = (
        1
        if quant_mode == 2
        else d_aligned_32 - nope_head_dim - rope_head_dim * 2 - quant_scale_head_dim * 2
    )
    # 根据输入的data range，计算scale范围，生成scale tensor，取倒数保存为bin
    ori_quant_param_range_left = ori_kv_datarange[0] / FP8_DATA_RANGE_LEFT
    ori_quant_param_range_right = ori_kv_datarange[1] / FP8_DATA_RANGE_RIGHT
    cmp_quant_param_range_left = cmp_kv_datarange[0] / FP8_DATA_RANGE_LEFT
    cmp_quant_param_range_right = cmp_kv_datarange[1] / FP8_DATA_RANGE_RIGHT

    # generate sinks tensor
    sinks = torch.tensor(
        np.random.uniform(q_datarange[0] / 10, q_datarange[1] / 10, (N1))
    ).to(torch.float)

    # generate ori_kv tensor
    if quant_mode == 1:
        (
            ori_k_bnsd,
            ori_k_in_pa_shape,
            ori_block_table,
            ori_sparse_indices,
            ori_topk_length,
        ) = gen_ori_kv(
            q_type,
            ori_kv_compute_type,
            B,
            S1,
            T1,
            N2,
            rope_head_dim,
            nope_head_dim,
            tile_size,
            quant_scale_head_dim,
            d_aligned_32,
            pad_d,
            block_num1,
            block_size1,
            ori_max_s2,
            ori_max_block_num_per_batch,
            seqused_ori_kv,
            cu_seqlens_ori_kv,
            ori_quant_param_range_left,
            ori_quant_param_range_right,
            quant_mode,
            layout_kv,
            data_range_left=ori_kv_datarange[0],
            data_range_right=ori_kv_datarange[1],
            K1=K1,
            template_run_mode=template_run_mode,
            layout_q=layout_q,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            ori_mask_mode=ori_mask_mode,
            ori_sparse_indices_mode=ori_sparse_indices_mode,
            ori_kv_topk_mode=ori_kv_topk_mode,
            ori_topk_length_override=ori_topk_length_override,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )
    else:
        (
            ori_k_bnsd,
            ori_k_in_pa_shape,
            ori_block_table,
            ori_sparse_indices,
            ori_topk_length,
        ) = gen_ori_kv_quant_2_pa(
            q_type,
            ori_kv_compute_type,
            B,
            S1,
            T1,
            N2,
            rope_head_dim,
            nope_head_dim,
            tile_size,
            quant_scale_head_dim,
            pad_d,
            block_num1,
            block_size1,
            ori_max_s2,
            ori_max_block_num_per_batch,
            seqused_ori_kv,
            ori_quant_param_range_left,
            ori_quant_param_range_right,
            d_combined_quant_2,
            layout_kv,
            data_range_left=ori_kv_datarange[0],
            data_range_right=ori_kv_datarange[1],
            K1=K1,
            template_run_mode=template_run_mode,
            layout_q=layout_q,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            ori_mask_mode=ori_mask_mode,
            ori_sparse_indices_mode=ori_sparse_indices_mode,
            ori_kv_topk_mode=ori_kv_topk_mode,
            ori_topk_length_override=ori_topk_length_override,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )
    # generate cmp_kv and sparse_indices
    if template_run_mode in ("HCA", "CSA", "ORI_CMP_SPARSE"):
        if quant_mode == 1:
            (
                cmp_k_bnsd,
                cmp_k_in_pa_shape,
                cmp_block_table,
                cmp_sparse_indices,
                cmp_topk_length,
            ) = gen_cmp_kv(
                q_type,
                layout_q,
                cmp_kv_compute_type,
                B,
                S1,
                T1,
                N2,
                D,
                K,
                rope_head_dim,
                nope_head_dim,
                tile_size,
                quant_scale_head_dim,
                d_aligned_32,
                pad_d,
                block_num2,
                block_size2,
                cmp_max_s2,
                cmp_max_block_num_per_batch,
                cu_seqlens_q,
                seqused_q,
                seqused_cmp_kv,
                cu_seqlens_cmp_kv,
                cmp_residual_kv,
                cmp_ratio,
                cmp_mask_mode,
                template_run_mode,
                cmp_quant_param_range_left,
                cmp_quant_param_range_right,
                quant_mode,
                layout_kv,
                data_range_left=cmp_kv_datarange[0],
                data_range_right=cmp_kv_datarange[1],
                cmp_kv_topk_mode=cmp_kv_topk_mode,
                cmp_sparse_indices_mode=cmp_sparse_indices_mode,
                cmp_topk_length_override=cmp_topk_length_override,
            )
        else:
            (
                cmp_k_bnsd,
                cmp_k_in_pa_shape,
                cmp_block_table,
                cmp_sparse_indices,
                cmp_topk_length,
            ) = gen_cmp_kv_quant_2_pa(
                q_type,
                cmp_kv_compute_type,
                B,
                S1,
                T1,
                N2,
                K,
                rope_head_dim,
                nope_head_dim,
                tile_size,
                quant_scale_head_dim,
                d_combined_quant_2,
                pad_d,
                block_num2,
                block_size2,
                cmp_max_s2,
                cmp_max_block_num_per_batch,
                layout_q,
                cu_seqlens_q,
                seqused_q,
                seqused_cmp_kv,
                cmp_residual_kv,
                cmp_ratio,
                cmp_mask_mode,
                template_run_mode,
                cmp_quant_param_range_left,
                cmp_quant_param_range_right,
                data_range_left=cmp_kv_datarange[0],
                data_range_right=cmp_kv_datarange[1],
                cmp_kv_topk_mode=cmp_kv_topk_mode,
                cmp_sparse_indices_mode=cmp_sparse_indices_mode,
                cmp_topk_length_override=cmp_topk_length_override,
            )
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k_bnsd = None
        cmp_topk_length = None

    # Golden tensors above retain FP8 semantics. Only the tensors crossing the
    # operator/save boundary are reinterpreted as uint8 when requested.
    ori_k_in_pa_shape = reinterpret_kv_for_operator(ori_k_in_pa_shape, ori_kv_type)
    cmp_k_in_pa_shape = reinterpret_kv_for_operator(cmp_k_in_pa_shape, cmp_kv_type)

    # 0轴非连续
    # if layout_kv == "PA_BBND" and (template_run_mode == "HCA" or template_run_mode == "CSA"):
    #     total_block = block_size1 + block_size2
    #     fusion_base = torch.zeros((block_num, total_block, N2, d_combined + pad_d), dtype=ori_kv_type, device="npu")
    #     fusion_base[:, :block_size1, :, :] = ori_k_in_pa_shape
    #     fusion_base[:, block_size1:, :, :] = cmp_k_in_pa_shape
    #     stride_n = total_block * N2 * (d_combined + pad_d)
    #     stride_bs = N2 * (d_combined + pad_d)
    #     stride_n2 = d_combined + pad_d
    #     stride_d = 1
    #     ori_k_in_pa_shape = torch.as_strided(
    #             fusion_base,
    #             size=[block_num, block_size1, N2, d_combined + pad_d],
    #             stride=[stride_n, stride_bs, stride_n2, stride_d])
    #     cmp_k_in_pa_shape = torch.as_strided(
    #             fusion_base,
    #             size=[block_num, block_size2, N2, d_combined + pad_d],
    #             stride=[stride_n, stride_bs, stride_n2, stride_d],
    #             storage_offset=block_size1 * N2 * (d_combined + pad_d))

    golden_state = {
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "q_type": q_type,
        "ori_kv_type": ori_kv_compute_type,
        "cmp_kv_type": cmp_kv_compute_type,
        "B": B,
        "S1": S1,
        "T1": T1,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K": K,
        "block_num1": block_num1,
        "block_num2": block_num2,
        "block_size1": block_size1,
        "block_size2": block_size2,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "softmax_scale": softmax_scale,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": ori_mask_mode,
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": ori_win_left,
        "ori_win_right": ori_win_right,
        "quant_mode": quant_mode,
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "ori_topk_length": ori_topk_length,
        "cmp_topk_length": cmp_topk_length,
        "template_run_mode": template_run_mode,
        "q": q,
        "ori_k_bnsd": ori_k_bnsd,
        "cmp_k_bnsd": cmp_k_bnsd,
        "ori_sparse_indices": ori_sparse_indices,
        "cmp_sparse_indices": cmp_sparse_indices,
        "sinks": sinks,
        "return_softmax_lse": return_softmax_lse,
        "batch_consistency": batch_consistency,
    }
    if generate_golden:
        generate_cpu_golden({"golden_state": golden_state})
        cpu_result = golden_state["cpu_output"]
        cpu_lse = golden_state["cpu_lse"]
    else:
        cpu_result = None
        cpu_lse = None

    print("mode:%s\n", template_run_mode)

    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    seqused_cmp_kv = (
        torch.tensor(seqused_cmp_kv).to(torch.int32)
        if seqused_cmp_kv is not None
        else None
    )
    cu_seqlens_ori_kv = (
        torch.tensor(cu_seqlens_ori_kv).to(torch.int32)
        if cu_seqlens_ori_kv is not None
        else None
    )
    cu_seqlens_cmp_kv = (
        torch.tensor(cu_seqlens_cmp_kv).to(torch.int32)
        if cu_seqlens_cmp_kv is not None
        else None
    )
    cmp_residual_kv = (
        torch.tensor(cmp_residual_kv).to(torch.int32)
        if cmp_residual_kv is not None
        else None
    )
    max_seqlen_q = S1
    if layout_q == "TND":
        max_seqlen_q = cu_seqlens_q.max().item()
    else:
        cu_seqlens_q = None
        max_seqlen_q = S1
    max_seqlen_ori_kv = seqused_ori_kv.max().item()
    max_seqlen_cmp_kv = seqused_cmp_kv.max().item() if seqused_cmp_kv is not None else 0

    # ORI_SPARSE/ORI_CMP_SPARSE: seqused (actualLength) not passed to op, determined by sparse_indices and topkLength
    # cu_seqlens still passed for TND kv layout (needed for data addressing in kernel)
    no_actual_length = template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
    op_seqused_ori_kv = None if no_actual_length else seqused_ori_kv
    op_seqused_cmp_kv = None if no_actual_length else seqused_cmp_kv
    op_cu_seqlens_ori_kv = cu_seqlens_ori_kv if layout_kv == "TND" else None
    op_cu_seqlens_cmp_kv = cu_seqlens_cmp_kv if layout_kv == "TND" else None

    input_data = {
        "Testcase_Name": Testcase_Name,
        "params": params,
        "metadata_input": {
            "num_heads_q": N1,
            "num_heads_kv": N2,
            "head_dim": D,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "cu_seqlens_ori_kv": op_cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": op_cu_seqlens_cmp_kv,
            "seqused_ori_kv": op_seqused_ori_kv,
            "seqused_cmp_kv": op_seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": ori_topk_length,
            "cmp_topk_length": cmp_topk_length,
            "batch_size": B,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_ori_kv": max_seqlen_ori_kv,
            "max_seqlen_cmp_kv": max_seqlen_cmp_kv,
            "topk": K if template_run_mode in ("CSA", "ORI_CMP_SPARSE") else 0,
            "ori_topk": K1
            if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            else 0,
            "cmp_ratio": cmp_ratio
            if template_run_mode not in ("SWA", "ORI_SPARSE")
            else 1,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "has_ori_kv": True,
            "has_cmp_kv": False if template_run_mode in ("SWA", "ORI_SPARSE") else True,
            "quant_mode": quant_mode,
            "rope_head_dim": rope_head_dim,
        },
        "op_input": {
            "q": q,
            "ori_kv": ori_k_in_pa_shape,
            "cmp_kv": cmp_k_in_pa_shape,
            "ori_sparse_indices": ori_sparse_indices,
            "cmp_sparse_indices": cmp_sparse_indices,
            "ori_topk_length": ori_topk_length,
            "cmp_topk_length": cmp_topk_length,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "cu_seqlens_ori_kv": op_cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": op_cu_seqlens_cmp_kv,
            "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks,
            "quant_mode": quant_mode,
            "tile_size": 64,
            "rope_head_dim": rope_head_dim,
            "softmax_scale": softmax_scale,
            "cmp_ratio": cmp_ratio
            if template_run_mode not in ("SWA", "ORI_SPARSE")
            else 1,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "topk_value_mode": topk_value_mode,
            "return_softmax_lse": return_softmax_lse,
        },
        "golden_state": golden_state,
        "cpu_output": cpu_result,
        "cpu_lse": cpu_lse if return_softmax_lse else None,
    }

    return input_data


def generate_cpu_golden(input_data):
    """Calculate MQSMLA CPU Golden from input-stage state without new random data."""
    state = input_data["golden_state"]
    test_qsmla = GeneralizedSFAQuant(
        state["layout_q"],
        state["layout_kv"],
        state["q_type"],
        state["ori_kv_type"],
        state["cmp_kv_type"],
        state["B"],
        state["S1"],
        state["T1"],
        state["N1"],
        state["N2"],
        state["D"],
        state["K"],
        state["block_num1"],
        state["block_num2"],
        state["block_size1"],
        state["block_size2"],
        state["cu_seqlens_q"],
        state["seqused_q"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cu_seqlens_ori_kv"],
        state["cu_seqlens_cmp_kv"],
        state["cmp_residual_kv"],
        state["softmax_scale"],
        state["cmp_ratio"],
        state["ori_mask_mode"],
        state["cmp_mask_mode"],
        state["ori_win_left"],
        state["ori_win_right"],
        state["quant_mode"],
        state["tile_size"],
        state["rope_head_dim"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
        state["template_run_mode"],
    )
    cpu_output, cpu_lse = test_qsmla.forward(
        state["q"],
        state["ori_k_bnsd"],
        state["cmp_k_bnsd"],
        state["ori_sparse_indices"],
        state["cmp_sparse_indices"],
        state["cu_seqlens_q"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cmp_residual_kv"],
        state["sinks"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
        state["return_softmax_lse"],
        state.get("batch_consistency", False),
    )
    state["cpu_output"] = cpu_output
    state["cpu_lse"] = cpu_lse
    input_data["cpu_output"] = cpu_output
    input_data["cpu_lse"] = cpu_lse
    return cpu_output, cpu_lse
