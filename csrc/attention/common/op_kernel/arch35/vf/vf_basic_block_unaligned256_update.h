/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_basic_block_unaligned256.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED256_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED256_UPDATE_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateGeneralImpl256VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ pseShiftType * pseUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb,
    __ubuf__ T * expMaxUb, __ubuf__ T * inMaxUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb,
    __ubuf__ T * tmpMaxUb2, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, __ubuf__ uint32_t * maskUb3,
    __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * dropMaskUb1, __ubuf__ uint32_t * dropMaskUb2,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1,
    const uint32_t oriTailN2, const uint32_t tailN1, const uint32_t tailN2, uint32_t pltOriTailN1,
    uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2, float divValue, uint32_t pltN,
    const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel1;
    RegTensor<float> vreg_sel2;
    RegTensor<float> vreg_sel3;
    RegTensor<float> vreg_sel4;
    RegTensor<float> vreg_sel3_new;
    RegTensor<float> vreg_sel4_new;
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_input_x3_new;
    RegTensor<float> vreg_input_x4_new;
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_exp_sum1;
    RegTensor<float> vreg_exp_sum2;
    RegTensor<float> vreg_exp_sum3;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_exp_even1;
    RegTensor<float> vreg_exp_odd1;
    RegTensor<float> vreg_exp_even2;
    RegTensor<float> vreg_exp_odd2;
    RegTensor<float> vreg_pse1;
    RegTensor<float> vreg_pse2;
    RegTensor<float> vreg_pse3;
    RegTensor<float> vreg_pse4;
    RegTensor<float> vreg_alibi1;
    RegTensor<float> vreg_alibi2;
    RegTensor<float> vreg_alibi3;
    RegTensor<float> vreg_alibi4;
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_sink_input;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even1_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp1_bf16;
    RegTensor<bfloat16_t> vreg_exp2_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src1;
    RegTensor<bfloat16_t> vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse1_bf16;
    RegTensor<bfloat16_t> vreg_pse2_bf16;
    RegTensor<bfloat16_t> vreg_pse3_bf16;
    RegTensor<bfloat16_t> vreg_pse4_bf16;
    // half
    RegTensor<half> vreg_exp_even1_f16;
    RegTensor<half> vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16;
    RegTensor<half> vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp1_f16;
    RegTensor<half> vreg_exp2_f16;
    RegTensor<half> vreg_pse_f16_src1;
    RegTensor<half> vreg_pse_f16_src2;
    RegTensor<half> vreg_pse1_f16;
    RegTensor<half> vreg_pse2_f16;
    RegTensor<half> vreg_pse3_f16;
    RegTensor<half> vreg_pse4_f16;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
    MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
    MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
    MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
    MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    MaskReg preg_compare4;
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;

    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi1, posShift);
        Arange(vreg_alibi2, posShift + floatRepSize);
        Arange(vreg_alibi3, posShift + floatRepSize * 2);
        Arange(vreg_alibi4, posShift + floatRepSize * 3);
    }
    // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
        LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
        LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
            Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                            pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Abs(vreg_pse1, vreg_alibi1, preg_all);
                Abs(vreg_pse2, vreg_alibi2, preg_all);
                Abs(vreg_pse3, vreg_alibi3, preg_all);
                Abs(vreg_pse4, vreg_alibi4, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse1, vreg_pse1, preg_all);
                    Sqrt(vreg_pse2, vreg_pse2, preg_all);
                    Sqrt(vreg_pse3, vreg_pse3, preg_all);
                    Sqrt(vreg_pse4, vreg_pse4, preg_all);
                }
                Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
                Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
                Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
                Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
                Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
                Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
                Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
                Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            } else {
                if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                } else if constexpr (IsSameType<pseShiftType, half>::value) {
                    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                }
            }
            Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
            Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);  
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);              
            Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
            Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
            Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
            Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
            Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_sel4_new, preg_all);
            Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
            Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp3, preg_all);
        } else {
            Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
            Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3_new, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4_new, preg_all);
            Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
            Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp3, preg_all);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
        }
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2); // 获取新的max[s1, 1]
    
    Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max, tmpMaxUb2 + i);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
        ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
        ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
        ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
        ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
        Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
        Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum3, vreg_exp_sum3, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3, ureg_exp_sum, 1);

        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb1, s2BaseSize >> 3);
            MaskInterleave<half>(preg3, preg4, preg1, preg2);
            MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb2, s2BaseSize >> 3);
            MaskInterleave<half>(preg3, preg4, preg1, preg2);
            MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
            (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
            (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
        } else if constexpr (IsSameType<T2, half>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
            Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}

// update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool hasSink = false>
__aicore__ inline void ProcessVec1UpdateGeneralImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb, const float sinkValue = 0.0f)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ uint32_t * maskUb1 = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb2 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t * maskUb3 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 2);
    __ubuf__ uint32_t * maskUb4 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 3);
    __ubuf__ uint32_t * dropMaskUb1 = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
    __ubuf__ uint32_t * dropMaskUb2 = (__ubuf__ uint32_t *)(dropTensor.GetPhyAddr() + s2BaseSize / 16);

    ProcessVec1UpdateGeneralImpl256VF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, hasSink>(
        expUb1, expUb2, pseUb, maxUb, srcUb, expMaxUb, inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2, 
        maskUb1, maskUb2, maskUb3, maskUb4, dropMaskUb1, dropMaskUb2, nPadding, blockStride, repeatStride, 
        oriTailN1, oriTailN2, tailN1, tailN2, pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 
        divValue, pltN, m, pseStride, slopes, posShift, scale, minValue, sinkValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED256_UPDATE_H
