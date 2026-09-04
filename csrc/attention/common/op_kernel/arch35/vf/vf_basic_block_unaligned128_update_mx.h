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
 * \file vf_basic_block_unaligned128_update_mx.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED128_UPDATE_MX_H
#define VF_BASIC_BLOCK_UNALIGNED128_UPDATE_MX_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateGeneralImpl128Mxfp8FullquantVFSubloop0(
    __ubuf__ T2 *expUb, __ubuf__ T2 *x_expUb, __ubuf__ pseShiftType *pseUb, __ubuf__ T *maxUb, __ubuf__ T *maxUbStart,
    __ubuf__ T *srcUb, __ubuf__ T *expMaxUb, __ubuf__ T *inMaxUb, __ubuf__ T *expSumUb, __ubuf__ T *inExpSumUb,
    __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpExpSumUb2, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
    __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb, __ubuf__ uint32_t *maskUbUnroll,
    __ubuf__ uint32_t *dropMaskUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb,
    __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb, const uint32_t nPadding, const uint32_t blockStride,
    const uint32_t dupStride, const uint32_t oriTailN, const uint32_t tailN, const float dScale,
    uint32_t pltOriTailN, uint32_t pltTailN, float divValue, uint32_t pltN, const uint16_t m, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue,
    const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f, const float pScale = 1.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_sel_unroll;
    RegTensor<float> vreg_sel_unroll_new;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_input_x_unroll_new;
    // expsum
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_subloop_update;
    RegTensor<float> vreg_pre_loop_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_exp_even;
    RegTensor<float> vreg_exp_odd;
    // expsum
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_exp_sum;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_pre_sum;
    // 编码（PSE）
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_pse_unroll;
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
    // mxfp8
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
    // half
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_odd_f16;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
    MaskReg preg_tail_n = UpdateMask<T>(pltTailN);
    MaskReg preg_ori_tail_n = UpdateMask<T>(pltOriTailN);
    MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
    MaskReg preg_compare;
    MaskReg preg_compare_unroll;
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;

    // pScale 计算
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    if constexpr (hasAtten == 1 && isMlaSgd) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare, ((__ubuf__ uint32_t *)(maskUb)));
        MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare_unroll,
                                                                   ((__ubuf__ uint32_t *)(maskUbUnroll)));
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
        // pse 模式
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, dScale, preg_all); // Muls(scale)
            Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_ori_tail_n);
        } else {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                          IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all); // Muls(dScaleQK)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_ori_tail_n);
            }
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {  // inner 模式
                Abs(vreg_pse, vreg_alibi, preg_all);
                Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse, vreg_pse, preg_all);
                    Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                }
                Muls(vreg_pse, vreg_pse, slopes, preg_all);
                Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
            } else {   // outer模式
                if constexpr (IsSameType<pseShiftType, float>::value) {
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                } else {
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                }
            }
            Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
            Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_ori_tail_n);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, scale, preg_all); // scale
            Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            if (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, nPadding);
            }
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
            Select(vreg_sel_unroll_new, vreg_sel_unroll, vreg_min, preg_ori_tail_n);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize,
                vreg_sel, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_sel_unroll_new, preg_tail_n);
            Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll_new, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all);
        } else {
            Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_input_x_unroll_new, preg_tail_n);
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
        }
        Muls(vreg_input_max, vreg_input_max, INV_LN2, preg_all);
        Truncate<T, RoundMode::CAST_CEIL>(vreg_input_max, vreg_input_max, preg_all);
        Muls(vreg_input_max, vreg_input_max, LN2, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb),
            vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopMaxUb,
        vreg_in_max, preg_all); // loop 0的时候，把之前的全局最大值写入ub
    Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
    ExpSub(vreg_exp_max, vreg_in_max, vreg_max_new, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);

    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();  // 同步

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
        Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
            LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x, vreg_input_x_unroll,
                                                              srcUb + i * s2BaseSize);
        }
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum,
                                                                                          preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum,
                                                                     ureg_exp_sum, 1);

        // dropmask Compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            if constexpr (IsSameType<T2, float>::value) {
                MaskInterleave<half>(preg5, preg6, preg1, preg2);
            } else {
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            }
            Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
            Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
            Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, dupStride, preg_all);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, dupStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_bf16, (RegTensor<uint16_t> &)vreg_exp_even_bf16,
               (RegTensor<uint16_t> &)vreg_exp_odd_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, dupStride, preg_n_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t> &)vreg_exp_even_f8e5m2,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e5m2, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            // 硬件不支持 float → int8 直接转换，需要分两步：float → half → int8
            RegTensor<int8_t> vreg_exp_merge_tmp_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // float → half → Or → half → int8 → Gather
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_even_f16, vreg_exp_even, preg_all_f16);
            Cast<half, T, castTrait1>(vreg_exp_odd_f16, vreg_exp_odd, preg_all_f16);
            Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16, (MicroAPI::RegTensor<uint16_t> &)vreg_exp_even_f16,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_odd_f16, preg_all_b16);
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_exp_f16, preg_all_b16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t> &)vreg_exp_merge_int8,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_merge_tmp_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&)expUb), vreg_exp_merge_int8, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t> &)vreg_exp_even_hif8,
               (RegTensor<uint8_t> &)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, dupStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_f16, (RegTensor<uint16_t> &)vreg_exp_even_f16,
               (RegTensor<uint16_t> &)vreg_exp_odd_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, dupStride, preg_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_in_exp_sum, inExpSumUb);
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopSumUb, vreg_in_exp_sum, preg_all);
    Mul(vreg_exp_max, vreg_exp_max, vreg_in_exp_sum, preg_all);
    Add(vreg_exp_max, vreg_exp_max, vreg_exp_sum_brc, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)firstLoopSumUb, vreg_exp_sum_brc,
                                                      preg_all); // 传给loop 1用于更新rowSum
    Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_all_b8);
    StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0, preg_all_b8);
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateGeneralImpl128Mxfp8FullquantVFSubloop1(
    __ubuf__ T2 *expUb, __ubuf__ T2 *x_expUb, __ubuf__ pseShiftType *pseUb, __ubuf__ T *maxUb, __ubuf__ T *maxUbStart,
    __ubuf__ T *srcUb, __ubuf__ T *expMaxUb, __ubuf__ T *inMaxUb, __ubuf__ T *expSumUb, __ubuf__ T *inExpSumUb,
    __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpExpSumUb2, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
    __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb, __ubuf__ uint32_t *maskUbUnroll,
    __ubuf__ uint32_t *dropMaskUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb,
    __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb, const uint32_t nPadding, const uint32_t blockStride,
    const uint32_t dupStride, const uint32_t oriTailN, const uint32_t tailN, const float dScale,
    uint32_t pltOriTailN, uint32_t pltTailN, float divValue, uint32_t pltN, const uint16_t m, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue,
    const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f, const float pScale = 1.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_sel_unroll;
    RegTensor<float> vreg_sel_unroll_new;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_input_x_unroll_new;
    // MAX相关
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_subloop_update;
    RegTensor<float> vreg_pre_loop_max;
    RegTensor<float> vreg_max_new;
    // expsum
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_exp_sum;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_pre_sum;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_exp_even;
    RegTensor<float> vreg_exp_odd;
    // POS编码
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_pse_unroll;
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;

    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;
    // BF16
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
    // MXFP8
    RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_pad;
    // FP16
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_odd_f16;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
    MaskReg preg_tail_n = UpdateMask<T>(pltTailN);
    MaskReg preg_ori_tail_n = UpdateMask<T>(pltOriTailN);
    MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
    uint32_t maskLen = 128;
    MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
    MaskReg preg_all_b8_half = CreateMask<int8_t, MaskPattern::ALL>();
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;
    MaskReg preg_compare;
    MaskReg preg_compare_unroll;

    // PScale 计算
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    if constexpr (hasAtten == 1 && isMlaSgd) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare, ((__ubuf__ uint32_t *)(maskUb)));
        MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare_unroll,
                                                                   ((__ubuf__ uint32_t *)(maskUbUnroll)));
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
        // PSE 模式
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, dScale, preg_all); // Muls(scale)
            Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_ori_tail_n);
        } else {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                          IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all); // Muls(dScaleQK)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_ori_tail_n);
            }
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {  // inner
                Abs(vreg_pse, vreg_alibi, preg_all);
                Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse, vreg_pse, preg_all);
                    Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                }
                Muls(vreg_pse, vreg_pse, slopes, preg_all);
                Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
            } else {  // outer
                if constexpr (IsSameType<pseShiftType, float>::value) {
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                } else {
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                }
            }
            Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
            Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_ori_tail_n);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, scale, preg_all); // Muls scale
            Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            if (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, nPadding);
            }
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
            Select(vreg_sel_unroll_new, vreg_sel_unroll, vreg_min, preg_ori_tail_n);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize,
                vreg_sel, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_sel_unroll_new, preg_tail_n);
            Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll_new, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                              vreg_max_tmp, preg_all);
        } else {
            Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_input_x_unroll_new, preg_tail_n);
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
        }
        Muls(vreg_input_max, vreg_input_max, INV_LN2, preg_all);
        Truncate<T, RoundMode::CAST_CEIL>(vreg_input_max, vreg_input_max, preg_all);
        Muls(vreg_input_max, vreg_input_max, LN2, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb),
            vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);
    Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
    ExpSub(vreg_subloop_update, vreg_in_max, vreg_max_new, preg_all);
    LoadAlign(vreg_pre_loop_max, preLoopMaxUb);
    ExpSub(vreg_exp_max, vreg_pre_loop_max, vreg_max_new, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);

    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();  // 同步

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
        Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
            LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x, vreg_input_x_unroll,
                                                              srcUb + i * s2BaseSize);
        }
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);

        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp_sum, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb),
            vreg_exp_sum, ureg_exp_sum, 1);

        // dropmask
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            if constexpr (IsSameType<T2, float>::value) {
                MaskInterleave<half>(preg5, preg6, preg1, preg2);
            } else {
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            }
            Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
            Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
            Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
            Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, dupStride, preg_all);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, dupStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_bf16, (RegTensor<uint16_t> &)vreg_exp_even_bf16,
               (RegTensor<uint16_t> &)vreg_exp_odd_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, dupStride, preg_n_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t> &)vreg_exp_even_f8e5m2,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e5m2, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            // 硬件不支持 float → int8 直接转换，需要分两步：float → half → int8
            RegTensor<int8_t> vreg_exp_merge_tmp_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // float → half → Or → half → int8 → Gather
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_even_f16, vreg_exp_even, preg_all_f16);
            Cast<half, T, castTrait1>(vreg_exp_odd_f16, vreg_exp_odd, preg_all_f16);
            Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16, (MicroAPI::RegTensor<uint16_t> &)vreg_exp_even_f16,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_odd_f16, preg_all_b16);
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_exp_f16, preg_all_b16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t> &)vreg_exp_merge_int8,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_merge_tmp_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&)expUb), vreg_exp_merge_int8, blockStride, dupStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t> &)vreg_exp_even_hif8,
               (RegTensor<uint8_t> &)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, dupStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_f16, (RegTensor<uint16_t> &)vreg_exp_even_f16,
               (RegTensor<uint16_t> &)vreg_exp_odd_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, dupStride, preg_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();  // 同步
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
    RegTensor<float> vreg_first_sum;
    LoadAlign(vreg_first_sum, firstLoopSumUb);
    LoadAlign(vreg_pre_sum, preLoopSumUb);
    Mul(vreg_first_sum, vreg_subloop_update, vreg_first_sum, preg_all);
    Add(vreg_first_sum, vreg_first_sum, vreg_exp_sum_brc, preg_all);
    Mul(vreg_pre_sum, vreg_exp_max, vreg_pre_sum, preg_all);
    Add(vreg_pre_sum, vreg_first_sum, vreg_pre_sum, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb,
                                                      vreg_pre_sum, preg_all);
    // pScale
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
    Duplicate(vreg_p_scale, static_cast<float>(1.0f));
    Mul(vreg_p_scale, vreg_subloop_update, vreg_p_scale, preg_all);
    Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_p_scale, preg_all);
    Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0,
                                                    vreg_p_scale_bf16_0, preg_all_b16);
    Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_p_scale, preg_all);
    Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_all_b16);
    Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
       (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_all_b8);
    Duplicate(vreg_p_scale_f8e8m0_1, 0x7f, preg_all_b8);
    DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_1);
    StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0_dst0, preg_all_b8);
}

// update, 64 < originN <= 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0,
          bool isMlaSgd = false, bool isMlaFullQuant = false, bool hasSink = false>
__aicore__ inline void ProcessVec1UpdateGeneralImpl128Mxfp8Fullquant(
    const LocalTensor<T2> &dstTensor, const LocalTensor<uint8_t> &indexesTensor, const LocalTensor<T> &expSumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inExpSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &maskTensor,
    const LocalTensor<pseShiftType> &pseTensor, const LocalTensor<uint8_t> &dropTensor,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LocalTensor<float> &preLoopMaxTensor, const LocalTensor<float> &preLoopSumTensor,
    const LocalTensor<float> &firstLoopSumTensor, uint32_t subLoop, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T> &queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f, const float pScale = 1.0f)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t dupStride = 1;
    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = s2BaseSize - floatRepSize;
    const float dScale = scale * dScaleQK;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltTailN = tailN;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;
    
    __ubuf__ T2 *x_expUb = nullptr;
    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *inExpSumUb = (__ubuf__ T *)inExpSumTensor.GetPhyAddr();
    __ubuf__ T *expMaxUb = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ T *inMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T *tmpMaxUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesTensor.GetPhyAddr();
    __ubuf__ uint32_t *dropMaskUb = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
    __ubuf__ float *preLoopMaxUb = (__ubuf__ T *)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float *preLoopSumUb = (__ubuf__ T *)preLoopSumTensor.GetPhyAddr();
    __ubuf__ float *firstLoopSumUb = (__ubuf__ T *)firstLoopSumTensor.GetPhyAddr();

    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUbUnroll = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();

    if (subLoop == 0) {
        ProcessVec1UpdateGeneralImpl128Mxfp8FullquantVFSubloop0<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                                                                pseMode, hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            expUb, x_expUb, pseUb, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
            tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb, maskUbUnroll, dropMaskUb, pScaleSubLoop0Ub,
            preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, dupStride, oriTailN, tailN, dScale,
            pltOriTailN, pltTailN, divValue, pltN, m, pseStride, slopes, posShift, scale, dScaleQK,
            minValue, deSCaleKValue, sinkValue, pScale);
    } else {
        ProcessVec1UpdateGeneralImpl128Mxfp8FullquantVFSubloop1<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                                                                pseMode, hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            expUb, x_expUb, pseUb, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
            tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb, maskUbUnroll, dropMaskUb, pScaleSubLoop0Ub,
            preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, dupStride, oriTailN, tailN, dScale,
            pltOriTailN, pltTailN, divValue, pltN, m, pseStride, slopes, posShift,
            scale, dScaleQK, minValue, deSCaleKValue, sinkValue, pScale);
    }
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED128_UPDATE_H
