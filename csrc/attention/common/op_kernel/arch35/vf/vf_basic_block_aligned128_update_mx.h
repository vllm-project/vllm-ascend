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
 * \file vf_basic_block_aligned128_update_mx.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED128_UPDATE_MX_H
#define VF_BASIC_BLOCK_ALIGNED128_UPDATE_MX_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
// update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateImpl128Mxfp8FullquantVFSubloop0(
    __ubuf__ T2 *expUb, __ubuf__ T2 *x_expUb, __ubuf__ pseShiftType *pseUb, __ubuf__ T *maxUb, __ubuf__ T *maxUbStart,
    __ubuf__ T *srcUb, __ubuf__ T *expMaxUb, __ubuf__ T *inMaxUb, __ubuf__ T *expSumUb, __ubuf__ T *inExpSumUb,
    __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpExpSumUb2, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
    __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb, __ubuf__ uint32_t *maskUbUnroll,
    __ubuf__ uint32_t *dropMaskUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb,
    __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb, float divValue, const uint32_t blockStride,
    const uint32_t repeatStride, const float dScale, const uint16_t m, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const float dScaleQK, const T minValue, const float deSCaleKValue = 1.0f,
    const float sinkValue = 0.0f, const float pScale = 1.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_select;
    RegTensor<float> vreg_select_unroll;
    RegTensor<float> vreg_x_src;
    RegTensor<float> vreg_x_src_unroll;
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_subloop_update;
    RegTensor<float> vreg_pre_loop_max;
    RegTensor<float> vreg_zero;
    // SUM
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_max_new;
    // EXP相关
    RegTensor<float> vreg_input_exp_sum;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_pre_sum;
    RegTensor<float> vreg_exp_even;
    RegTensor<float> vreg_exp_odd;
    // 位置编码 PSE
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_pse_unroll;
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;
    RegTensor<float> vreg_select_drop;
    RegTensor<float> vreg_select_drop2;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;
    // half
    RegTensor<half> vreg_exp_f16_even;
    RegTensor<half> vreg_exp_f16_odd;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_bf16_even;
    RegTensor<bfloat16_t> vreg_exp_bf16_odd;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
    // mxfp8
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all_float = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
    MaskReg preg_compare;
    MaskReg preg_compare_unroll;
    MaskReg preg0;
    MaskReg preg1 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg2;
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    // pScale计算
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all_float);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    // MASK 相关
    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare, ((__ubuf__ uint32_t *)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(
                preg_compare_unroll, ((__ubuf__ uint32_t *)(maskUbUnroll)));
        }
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_x_src, srcUb + i * s2BaseSize);
        LoadAlign(vreg_x_src_unroll, srcUb + floatRepSize + i * s2BaseSize);
        // PSE模式
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_x_src, vreg_x_src, dScale, preg_all_float); // Muls(scale)
            Muls(vreg_x_src_unroll, vreg_x_src_unroll, dScale, preg_all_float);
        } else {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value ||
                          IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_x_src, vreg_x_src, dScaleQK, preg_all_float); // Muls(dScaleQK)
                Muls(vreg_x_src_unroll, vreg_x_src_unroll, dScaleQK, preg_all_float);
            }
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                          pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {  // inner
                Abs(vreg_pse, vreg_alibi, preg_all_float);
                Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all_float);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse, vreg_pse, preg_all_float);
                    Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all_float);
                }
                Muls(vreg_pse, vreg_pse, slopes, preg_all_float);
                Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all_float);
                Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all_float);
                Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all_float);
            } else {    // outer
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
            Add(vreg_x_src, vreg_x_src, vreg_pse, preg_all_float);
            Add(vreg_x_src_unroll, vreg_x_src_unroll, vreg_pse_unroll, preg_all_float);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_x_src, vreg_x_src, scale, preg_all_float); // Muls(scale)
            Muls(vreg_x_src_unroll, vreg_x_src_unroll, scale, preg_all_float);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            if constexpr (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize);
            }
            Select(vreg_select, vreg_min, vreg_x_src, preg_compare);
            Select(vreg_select_unroll, vreg_min, vreg_x_src_unroll, preg_compare_unroll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_select,
                                                              preg_all_float);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_select_unroll, preg_all_float);
            Max(vreg_max_tmp, vreg_select, vreg_select_unroll, preg_all_float);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all_float);
        } else {
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize,
                                                              vreg_x_src, preg_all_float);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_x_src_unroll, preg_all_float);
            Max(vreg_max_tmp, vreg_x_src, vreg_x_src_unroll, preg_all_float);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all_float);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all_float);
        }
        Muls(vreg_input_max, vreg_input_max, INV_LN2, preg_all_float);
        Truncate<T, RoundMode::CAST_CEIL>(vreg_input_max, vreg_input_max, preg_all_float);
        Muls(vreg_input_max, vreg_input_max, LN2, preg_all_float);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max,
                                                                     ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopMaxUb, vreg_in_max, preg_all_float);
    Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all_float);
    ExpSub(vreg_exp_max, vreg_in_max, vreg_max_new, preg_all_float);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all_float);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all_float);

    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all_float);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
        Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all_float);
        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_x_src, srcUb + i * s2BaseSize);
            LoadAlign(vreg_x_src_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_x_src, vreg_x_src_unroll,
                                                              srcUb + i * s2BaseSize);
        }
        ExpSub(vreg_exp_even, vreg_x_src, vreg_max, preg_all_float);
        ExpSub(vreg_exp_odd, vreg_x_src_unroll, vreg_max, preg_all_float);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_float);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum,
                                                                                          preg_all_float);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum,
                                                                     ureg_exp_sum, 1);

        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg0, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            if constexpr (IsSameType<T2, float>::value) {
                MaskInterleave<half>(preg4, preg5, preg0, preg1);
            } else {
                MaskInterleave<half>(preg2, preg3, preg0, preg1);
                MaskDeInterleave<T>(preg4, preg5, preg2, preg3);
            }
            Select(vreg_select_drop, vreg_exp_even, vreg_zero, preg4);
            Muls(vreg_exp_even, vreg_select_drop, divValue, preg_all_float);
            Select(vreg_select_drop2, vreg_exp_odd, vreg_zero, preg5);
            Muls(vreg_exp_odd, vreg_select_drop2, divValue, preg_all_float);
        }

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all_float);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all_float);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_bf16_even, vreg_exp_even, preg_all_float);
            Cast<T2, T, castTraitOne>(vreg_exp_bf16_odd, vreg_exp_odd, preg_all_float);
            Or((RegTensor<uint16_t> &)vreg_exp_bf16, (RegTensor<uint16_t> &)vreg_exp_bf16_even,
               (RegTensor<uint16_t> &)vreg_exp_bf16_odd, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all_float);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all_float);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t> &)vreg_exp_even_f8e5m2,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e5m2, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all_float);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all_float);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
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
            Cast<half, T, castTrait0>(vreg_exp_f16_even, vreg_exp_even, preg_all_float);
            Cast<half, T, castTrait1>(vreg_exp_f16_odd, vreg_exp_odd, preg_all_float);
            Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16, (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16_even,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16_odd, preg_all_f16);
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_exp_f16, preg_all_f16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t> &)vreg_exp_merge_int8,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_merge_tmp_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&)expUb), vreg_exp_merge_int8, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all_float);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all_float);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t> &)vreg_exp_even_hif8,
               (RegTensor<uint8_t> &)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_f16_even, vreg_exp_even, preg_all_float);
            Cast<T2, T, castTraitOne>(vreg_exp_f16_odd, vreg_exp_odd, preg_all_float);
            Or((RegTensor<uint16_t> &)vreg_exp_f16, (RegTensor<uint16_t> &)vreg_exp_f16_even,
               (RegTensor<uint16_t> &)vreg_exp_f16_odd, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_exp_sum, inExpSumUb);
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopSumUb, vreg_input_exp_sum, preg_all_float);
    Mul(vreg_exp_max, vreg_exp_max, vreg_input_exp_sum, preg_all_float);
    Add(vreg_exp_max, vreg_exp_max, vreg_exp_sum_brc, preg_all_float);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_max, preg_all_float);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)firstLoopSumUb, vreg_exp_sum_brc,
                                                      preg_all_float); // 传给loop 1用于更新rowSum
    Duplicate(vreg_p_scale_f8, 0x7f, preg_all_b8);
    StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8, preg_all_b8);
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateImpl128Mxfp8FullquantVFSubloop1(
    __ubuf__ T2 *expUb, __ubuf__ T2 *x_expUb, __ubuf__ pseShiftType *pseUb, __ubuf__ T *maxUb, __ubuf__ T *maxUbStart,
    __ubuf__ T *srcUb, __ubuf__ T *expMaxUb, __ubuf__ T *inMaxUb, __ubuf__ T *expSumUb, __ubuf__ T *inExpSumUb,
    __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpExpSumUb2, __ubuf__ T *tmpMaxUb, __ubuf__ T *tmpMaxUb2,
    __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb, __ubuf__ uint32_t *maskUbUnroll,
    __ubuf__ uint32_t *dropMaskUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb,
    __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb, float divValue, const uint32_t blockStride,
    const uint32_t repeatStride, const float dScale, const uint16_t m, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const float dScaleQK, const T minValue, const float deSCaleKValue = 1.0f,
    const float sinkValue = 0.0f, const float pScale = 1.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_select;
    RegTensor<float> vreg_select_unroll;
    RegTensor<float> vreg_x_src;
    RegTensor<float> vreg_x_src_unroll;
    // 位置编码
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_pse_unroll;
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;
    // MAX SUM
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_subloop_update;
    RegTensor<float> vreg_pre_loop_max;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_input_exp_sum;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_pre_sum;
    RegTensor<float> vreg_exp_even;
    RegTensor<float> vreg_exp_odd;

    RegTensor<float> vreg_select_drop;
    RegTensor<float> vreg_select_drop2;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_bf16_even;
    RegTensor<bfloat16_t> vreg_exp_bf16_odd;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
    // mxfp8
    RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8_0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8_pad;
    // half
    RegTensor<half> vreg_exp_f16_even;
    RegTensor<half> vreg_exp_f16_odd;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
    uint32_t maskLen = 128;
    MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
    MaskReg preg_all_b8_half = CreateMask<int8_t, MaskPattern::ALL>();
    MaskReg preg_compare;
    MaskReg preg_compare_unroll;
    MaskReg preg0;
    MaskReg preg1 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg2;
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;

    RegTensor<float> vreg_p_scale;    // PScale相关
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    // mask 相关
    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare, ((__ubuf__ uint32_t *)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare_unroll,
                ((__ubuf__ uint32_t *)(maskUbUnroll)));
        }
    }
    // PSE
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_x_src, srcUb + i * s2BaseSize);
        LoadAlign(vreg_x_src_unroll, srcUb + floatRepSize + i * s2BaseSize);
        // pse模式
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_x_src, vreg_x_src, dScale, preg_all); // Muls scale
            Muls(vreg_x_src_unroll, vreg_x_src_unroll, dScale, preg_all);
        } else {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                          IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_x_src, vreg_x_src, dScaleQK, preg_all);
                Muls(vreg_x_src_unroll, vreg_x_src_unroll, dScaleQK, preg_all);
            }
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE ||
                          pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE) {  // INNER
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
            } else {    // OUTER
                if constexpr (IsSameType<pseShiftType, float>::value) {
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                } else {  // fp16
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                }
            }
            Add(vreg_x_src, vreg_x_src, vreg_pse, preg_all);
            Add(vreg_x_src_unroll, vreg_x_src_unroll, vreg_pse_unroll, preg_all);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_x_src, vreg_x_src, scale, preg_all); // Muls(scale)
            Muls(vreg_x_src_unroll, vreg_x_src_unroll, scale, preg_all);
        }
        // atten mask
        if constexpr (hasAtten == 1) {
            if constexpr (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize);
            }
            Select(vreg_select, vreg_min, vreg_x_src, preg_compare);
            Select(vreg_select_unroll, vreg_min, vreg_x_src_unroll, preg_compare_unroll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_select,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_select_unroll, preg_all);
            Max(vreg_max_tmp, vreg_select, vreg_select_unroll, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all);
        } else {
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_x_src,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_x_src_unroll, preg_all);
            Max(vreg_max_tmp, vreg_x_src, vreg_x_src_unroll, preg_all);
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
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);  // store
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
        Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_x_src, srcUb + i * s2BaseSize);
            LoadAlign(vreg_x_src_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_x_src,
                vreg_x_src_unroll, srcUb + i * s2BaseSize);
        }
        ExpSub(vreg_exp_even, vreg_x_src, vreg_max, preg_all);
        ExpSub(vreg_exp_odd, vreg_x_src_unroll, vreg_max, preg_all);
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp_sum, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb),
            vreg_exp_sum, ureg_exp_sum, 1);

        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg0, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            if constexpr (IsSameType<T2, float>::value) {
                MaskInterleave<half>(preg4, preg5, preg0, preg1);
            } else {
                MaskInterleave<half>(preg2, preg3, preg0, preg1);
                MaskDeInterleave<T>(preg4, preg5, preg2, preg3);
            }
            Select(vreg_select_drop, vreg_exp_even, vreg_zero, preg4);
            Muls(vreg_exp_even, vreg_select_drop, divValue, preg_all);
            Select(vreg_select_drop2, vreg_exp_odd, vreg_zero, preg5);
            Muls(vreg_exp_odd, vreg_select_drop2, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_bf16_even, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_bf16_odd, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_bf16, (RegTensor<uint16_t> &)vreg_exp_bf16_even,
               (RegTensor<uint16_t> &)vreg_exp_bf16_odd, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_tmp_merge_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_tmp_merge_f8e5m2, (RegTensor<uint8_t> &)vreg_exp_even_f8e5m2,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e5m2, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_tmp_merge_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_tmp_merge_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_tmp_merge_f8e4m3, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_tmp_merge_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            // 硬件不支持 float → int8 直接转换，需要分两步：float → half → int8
            RegTensor<int8_t> vreg_exp_tmp_merge_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // float → half → Or → half → int8 → Gather
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_f16_even, vreg_exp_even, preg_all);
            Cast<half, T, castTrait1>(vreg_exp_f16_odd, vreg_exp_odd, preg_all);
            Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16, (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16_even,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_f16_odd, preg_all_f16);
            Cast<T2, half, castTrait0>(vreg_exp_tmp_merge_int8, vreg_exp_f16, preg_all_f16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t> &)vreg_exp_merge_int8,
                (MicroAPI::RegTensor<uint16_t> &)vreg_exp_tmp_merge_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&)expUb), vreg_exp_merge_int8, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_tmp_merge_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_tmp_merge_hif8, (RegTensor<uint8_t> &)vreg_exp_even_hif8,
               (RegTensor<uint8_t> &)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_tmp_merge_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_f16_even, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_f16_odd, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t> &)vreg_exp_f16, (RegTensor<uint16_t> &)vreg_exp_f16_even,
               (RegTensor<uint16_t> &)vreg_exp_f16_odd, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
    RegTensor<float> vreg_sum_first;
    LoadAlign(vreg_sum_first, firstLoopSumUb);
    LoadAlign(vreg_pre_sum, preLoopSumUb);
    Mul(vreg_sum_first, vreg_subloop_update, vreg_sum_first, preg_all);
    Add(vreg_sum_first, vreg_sum_first, vreg_exp_sum_brc, preg_all);
    Mul(vreg_pre_sum, vreg_exp_max, vreg_pre_sum, preg_all);
    Add(vreg_pre_sum, vreg_sum_first, vreg_pre_sum, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_pre_sum, preg_all);
    // pScale
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8_dst0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8_dst1;
    Duplicate(vreg_p_scale, static_cast<float>(1.0f));
    Mul(vreg_p_scale, vreg_subloop_update, vreg_p_scale, preg_all);
    Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_p_scale, preg_all);
    Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8_0,
                                                    vreg_p_scale_bf16_0, preg_all_b16);
    Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_p_scale, preg_all);
    Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8_1, vreg_p_scale_bf16_1, preg_all_b16);
    Or((RegTensor<uint8_t> &)vreg_p_scale_f8_0, (RegTensor<uint8_t> &)vreg_p_scale_f8_0,
       (RegTensor<uint8_t> &)vreg_p_scale_f8_1, preg_all_b8);
    Duplicate(vreg_p_scale_f8_1, 0x7f, preg_all_b8);
    DeInterleave(vreg_p_scale_f8_dst0, vreg_p_scale_f8_dst1, vreg_p_scale_f8_0, vreg_p_scale_f8_1);
    StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8_dst0, preg_all_b8);
}
// update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool hasSink = false>
__aicore__ inline void ProcessVec1UpdateImpl128Mxfp8Fullquant(
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
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const float dScale = scale * dScaleQK;
    float divValue = 1.0f / keepProb;

    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpExpSumUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *inExpSumUb = (__ubuf__ T *)inExpSumTensor.GetPhyAddr();
    __ubuf__ T *expMaxUb = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ T *inMaxUb = (__ubuf__ T *)inMaxTensor.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T *tmpMaxUb2 = (__ubuf__ T *)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesTensor.GetPhyAddr();
    __ubuf__ float *preLoopMaxUb = (__ubuf__ T *)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float *preLoopSumUb = (__ubuf__ T *)preLoopSumTensor.GetPhyAddr();
    __ubuf__ float *firstLoopSumUb = (__ubuf__ T *)firstLoopSumTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUbUnroll = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t *dropMaskUb = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    if (subLoop == 0) {
        ProcessVec1UpdateImpl128Mxfp8FullquantVFSubloop0<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                         hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            expUb, x_expUb, pseUb, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
            tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb, maskUbUnroll, dropMaskUb, pScaleSubLoop0Ub,
            preLoopMaxUb, preLoopSumUb, firstLoopSumUb, divValue, blockStride, repeatStride, dScale, m, pseStride,
            slopes, posShift, scale, dScaleQK, minValue, deSCaleKValue, sinkValue, pScale);
    } else {
        ProcessVec1UpdateImpl128Mxfp8FullquantVFSubloop1<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                                                         hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            expUb, x_expUb, pseUb, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
            tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb, maskUbUnroll, dropMaskUb, pScaleSubLoop0Ub,
            preLoopMaxUb, preLoopSumUb, firstLoopSumUb, divValue, blockStride, repeatStride, dScale, m, pseStride,
            slopes, posShift, scale, dScaleQK, minValue, deSCaleKValue, sinkValue, pScale);
    }
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_ALIGNED128_UPDATE_H
