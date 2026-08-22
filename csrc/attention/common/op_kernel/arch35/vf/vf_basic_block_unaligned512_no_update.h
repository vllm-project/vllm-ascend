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
 * \file vf_basic_block_unaligned512_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned512_common.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, uint32_t s2BaseSize>
__simd_callee__ static inline void PostReduceStage512(
    __ubuf__ T *&maxUb, UnalignRegForStore &ureg_max,
    __ubuf__ T *maxUbStart, RegTensor<float> &vreg_max_brc, const uint16_t m, __ubuf__ T *srcUb,
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2, RegTensor<float> &vreg_input_x3,
    RegTensor<float> &vreg_input_x4, RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_sum3, UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb,
    __ubuf__ T2 *expUb1, __ubuf__ T2 *expUb2, __ubuf__ T2 *expUb3, __ubuf__ T2 *expUb4,
    const uint32_t blockStride, const uint32_t repeatStride, MaskReg &preg_all, MaskReg &preg_all_b16)
{
    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), ureg_max, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);

        LoadInputDinterleave4<T>(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8, srcUb, i, s2BaseSize);

        ExpSub8(vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2, vreg_exp_even3, vreg_exp_odd3,
            vreg_exp_even4, vreg_exp_odd4, vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8, vreg_max_brc, preg_all);

        ExpSumReduceStore512<float, T2>(vreg_exp_sum3, vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2,
            vreg_exp_odd2, vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
            ureg_exp_sum, expSumUb, preg_all);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            CastStoreExpBf16_512<T2>(vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
                vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
                expUb1, expUb2, expUb3, expUb4, blockStride, repeatStride, preg_all, preg_all_b16);
        } else if constexpr (IsSameType<T2, half>::value) {
            CastStoreExpF16_512<T2>(vreg_exp_even1, vreg_exp_odd1,
                vreg_exp_even2, vreg_exp_odd2, vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
                expUb1, expUb2, expUb3, expUb4, blockStride, repeatStride, preg_all, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1NoUpdateGeneralImpl512VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ OUTPUT_T * pseUb,
    __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb,
    VREG_FLOAT_MASKUB_8_DECL,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1,
    const uint32_t oriTailN2, const uint32_t oriTailN3, const uint32_t oriTailN4, const uint32_t tailN1,
    const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4, uint32_t pltOriTailN1, uint32_t pltOriTailN2,
    uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3,
    uint32_t pltTailN4, float divValue, const uint16_t m, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const T minValue)
{
    VREG_FLOAT_DECL_8(vreg_sel);
    RegTensor<float> vreg_sel5_new;
    RegTensor<float> vreg_sel6_new;
    RegTensor<float> vreg_sel7_new;
    RegTensor<float> vreg_sel8_new;

    RegTensor<float> vreg_min;
    VREG_FLOAT_DECL_8(vreg_src_x);
    RegTensor<float> vreg_src_x5_new;
    RegTensor<float> vreg_src_x6_new;
    RegTensor<float> vreg_src_x7_new;
    RegTensor<float> vreg_src_x8_new;

    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_exp_sum3;

    VREG_FLOAT_DECL_EXP_PAIRS_4(vreg_exp);
    VREG_FLOAT_DECL_8(vreg_pse);
    VREG_FLOAT_DECL_8(vreg_alibi);

    MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
    MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
    MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
    MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
    MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
    MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);
    MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
    MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    VREG_PREG_COMPARE_DECL_8();

    Duplicate(vreg_min, minValue);
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        InitAlibiOffsets8(vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
            vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8, posShift, preg_all);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadInput8<T>(vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
            vreg_src_x5, vreg_src_x6, vreg_src_x7, vreg_src_x8, srcUb, i, s2BaseSize);

        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput4x4(vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
                vreg_src_x5, vreg_src_x6, vreg_src_x7, vreg_src_x8, scale, preg_all,
                preg_ori_tail_n1, preg_ori_tail_n2, preg_ori_tail_n3, preg_ori_tail_n4);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    ComputePseInnerMulAddSqrt8(ARG_FLOAT_PSE_ALIBI_8);
                } else {
                    ComputePseInnerMulAdd8(ARG_FLOAT_PSE_ALIBI_8);
                }
            } else {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    LoadCastPseBf16_8(ARG_FLOAT_PSE_8_LOAD);
                } else if constexpr (IsSameType<T2, half>::value) {
                    LoadCastPseF16_8(ARG_FLOAT_PSE_8_LOAD);
                }
            }
            AddPseToInput4x4(vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4, vreg_src_x5, vreg_src_x6,
                vreg_src_x7, vreg_src_x8, vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8, preg_all, preg_ori_tail_n1,
                preg_ori_tail_n2, preg_ori_tail_n3, preg_ori_tail_n4);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput4x4(vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
                vreg_src_x5, vreg_src_x6, vreg_src_x7, vreg_src_x8,
                scale, preg_all, preg_ori_tail_n1, preg_ori_tail_n2, preg_ori_tail_n3, preg_ori_tail_n4);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            LoadAttenMask8(preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8,
                maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8, nPadding);

            ApplyAttenMaskSelect8(
                vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8, vreg_min,
                vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
                vreg_src_x5, vreg_src_x6, vreg_src_x7, vreg_src_x8,
                preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8);

            Select(vreg_sel5_new, vreg_sel5, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel6_new, vreg_sel6, vreg_min, preg_ori_tail_n2);
            Select(vreg_sel7_new, vreg_sel7, vreg_min, preg_ori_tail_n3);
            Select(vreg_sel8_new, vreg_sel8, vreg_min, preg_ori_tail_n4);

            StoreAlign8<T>(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, vreg_sel5_new, vreg_sel6_new,
                vreg_sel7_new, vreg_sel8_new, srcUb, i, s2BaseSize, preg_all);

            MaxReduce4(vreg_input_max,
                vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5_new, vreg_sel6_new, vreg_sel7_new, vreg_sel8_new,
                preg_all);
        } else {
            Select(vreg_src_x5_new, vreg_src_x5, vreg_min, preg_ori_tail_n1);
            Select(vreg_src_x6_new, vreg_src_x6, vreg_min, preg_ori_tail_n2);
            Select(vreg_src_x7_new, vreg_src_x7, vreg_min, preg_ori_tail_n3);
            Select(vreg_src_x8_new, vreg_src_x8, vreg_min, preg_ori_tail_n4);

            StoreAlign8<T>(vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
                vreg_src_x5_new, vreg_src_x6_new, vreg_src_x7_new, vreg_src_x8_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce4(vreg_input_max, vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4,
                vreg_src_x5_new, vreg_src_x6_new, vreg_src_x7_new, vreg_src_x8_new, preg_all);
        }

        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    PostReduceStage512<T, T2, s2BaseSize>(
        maxUb, ureg_max, maxUbStart, vreg_max_brc, m, srcUb,
        vreg_src_x1, vreg_src_x2, vreg_src_x3, vreg_src_x4, vreg_src_x5, vreg_src_x6, vreg_src_x7, vreg_src_x8,
        vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
        vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
        vreg_exp_sum3, ureg_exp_sum, expSumUb, expUb1, expUb2, expUb3, expUb4,
        blockStride, repeatStride, preg_all, preg_all_b16);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ __no_simd_vf_fusion__ void ProcessVec1NoUpdateGeneralImpl512VFStage1(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ OUTPUT_T * pseUb,
    __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, VREG_FLOAT_MASKUB_8_DECL,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1,
    const uint32_t oriTailN2, const uint32_t oriTailN3, const uint32_t oriTailN4, const uint32_t tailN1,
    const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4, uint32_t pltOriTailN1, uint32_t pltOriTailN2,
    uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3,
    uint32_t pltTailN4, float divValue, const uint16_t m, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const T minValue)
{
    ProcessVec1TailLoop512<T, T2, OUTPUT_T, s2BaseSize, hasAtten, pseMode>(
        srcUb, pseUb, maskUb7, maskUb8, nPadding,
        pltOriTailN3, pltOriTailN4, m, pseStride, slopes, posShift, scale, minValue);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ __no_simd_vf_fusion__ void ProcessVec1NoUpdateGeneralImpl512VFStage2(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4,
    __ubuf__ OUTPUT_T * pseUb, __ubuf__ T * expSumUb, __ubuf__ T * maxUb,
    __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, VREG_FLOAT_MASKUB_8_DECL,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3,
    const uint32_t oriTailN4, const uint32_t tailN1, const uint32_t tailN2,
    const uint32_t tailN3, const uint32_t tailN4, uint32_t pltOriTailN1,
    uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4,
    float divValue, const uint16_t m, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const T minValue)
{
    VREG_FLOAT_DECL_8(vreg_x_src);
    RegTensor<float> vreg_x_src5_new;
    RegTensor<float> vreg_x_src6_new;
    RegTensor<float> vreg_x_src7_new;
    RegTensor<float> vreg_x_src8_new;
    RegTensor<float> vreg_min;
    VREG_FLOAT_DECL_6(vreg_sel);
    RegTensor<float> vreg_sel5_new;
    RegTensor<float> vreg_sel6_new;
    RegTensor<float> vreg_sel7_new;
    RegTensor<float> vreg_sel8_new;

    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_exp_sum3;

    VREG_FLOAT_DECL_8(vreg_pse);
    VREG_FLOAT_DECL_6(vreg_alibi);
    VREG_FLOAT_DECL_EXP_PAIRS_4(vreg_exp);

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;
    MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
    MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    VREG_PREG_COMPARE_DECL_6();

    Duplicate(vreg_min, minValue);
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        InitAlibiOffsets6(vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4, vreg_alibi5, vreg_alibi6,
            posShift, preg_all);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadInput6<T>(vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4,
            vreg_x_src5, vreg_x_src6, srcUb, i, s2BaseSize);

        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput4x2(vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4, vreg_x_src5, vreg_x_src6, scale,
                preg_all, preg_ori_tail_n1, preg_ori_tail_n2);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    ComputePseInnerMulAddSqrt6(ARG_FLOAT_PSE_ALIBI_6);
                } else {
                    ComputePseInnerMulAdd6(ARG_FLOAT_PSE_ALIBI_6);
                }
            } else {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    LoadCastPseBf16_6(ARG_FLOAT_PSE_6_LOAD);
                } else if constexpr (IsSameType<T2, half>::value) {
                    LoadCastPseF16_6(ARG_FLOAT_PSE_6_LOAD);
                }
            }
            AddPseToInput4x2(vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4, vreg_x_src5, vreg_x_src6,
                vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4, vreg_pse5, vreg_pse6,
                preg_all, preg_ori_tail_n1, preg_ori_tail_n2);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput4x2(vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4, vreg_x_src5, vreg_x_src6,
                scale, preg_all, preg_ori_tail_n1, preg_ori_tail_n2);
        }

        if constexpr (hasAtten == 1) {
            // atten mask
            LoadAttenMask6(preg_compare1, preg_compare2, preg_compare3, preg_compare4, preg_compare5,
                preg_compare6, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, nPadding);

            ApplyAttenMaskSelect6(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, vreg_sel5, vreg_sel6,
                vreg_min, vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4, vreg_x_src5, vreg_x_src6,
                preg_compare1, preg_compare2, preg_compare3, preg_compare4, preg_compare5, preg_compare6);

            Select(vreg_sel5_new, vreg_sel5, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel6_new, vreg_sel6, vreg_min, preg_ori_tail_n2);
            LoadAlign(vreg_sel7_new, srcUb + floatRepSize * 6 + i * s2BaseSize);
            LoadAlign(vreg_sel8_new, srcUb + floatRepSize * 7 + i * s2BaseSize);

            StoreAlign6<T>(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, vreg_sel5_new, vreg_sel6_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce4(vreg_input_max, vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, vreg_sel5_new,
                vreg_sel6_new, vreg_sel7_new, vreg_sel8_new, preg_all);
        } else {
            Select(vreg_x_src5_new, vreg_x_src5, vreg_min, preg_ori_tail_n1);
            Select(vreg_x_src6_new, vreg_x_src6, vreg_min, preg_ori_tail_n2);
            LoadAlign(vreg_x_src7_new, srcUb + floatRepSize * 6 + i * s2BaseSize);
            LoadAlign(vreg_x_src8_new, srcUb + floatRepSize * 7 + i * s2BaseSize);

            StoreAlign6<T>(vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4, vreg_x_src5_new,
                vreg_x_src6_new, srcUb, i, s2BaseSize, preg_all);

            MaxReduce4(vreg_input_max, vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4,
                vreg_x_src5_new, vreg_x_src6_new, vreg_x_src7_new, vreg_x_src8_new, preg_all);
        }
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb),
            vreg_input_max, ureg_max, 1);
    }
    PostReduceStage512<T, T2, s2BaseSize>(maxUb, ureg_max, maxUbStart, vreg_max_brc, m, srcUb,
        vreg_x_src1, vreg_x_src2, vreg_x_src3, vreg_x_src4,
        vreg_x_src5, vreg_x_src6, vreg_x_src7, vreg_x_src8,
        vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
        vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
        vreg_exp_sum3, ureg_exp_sum, expSumUb, expUb1, expUb2, expUb3, expUb4,
        blockStride, repeatStride, preg_all, preg_all_b16);
}

// 256 < Orignin N <=512
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl512(const LocalTensor<T2>& dstTensor,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    const uint32_t repeatStride = 1;
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2* ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3* ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    DO_MASKUB_INIT_8(maskTensor);

    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 4 < floatRepSize ? originN - floatRepSize * 4 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 5) <= 0 ? 0 : originN - floatRepSize * 5;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 6) <= 0 ? 0 : originN - floatRepSize * 6;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 7) <= 0 ? 0 : originN - floatRepSize * 7;

    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;

    const uint32_t tailN1 = s2BaseSize - floatRepSize * 4;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 5;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 6;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 7;

    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    float divValue = 1.0f / keepProb;

    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE && hasAtten == 1) {
        ProcessVec1NoUpdateGeneralImpl512VFStage1<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb,
            maxUbStart, srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4, pltOriTailN1, pltOriTailN2,
            pltOriTailN3, pltOriTailN4, pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
        ProcessVec1NoUpdateGeneralImpl512VFStage2<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb,
            maxUbStart, srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4, pltOriTailN1, pltOriTailN2,
            pltOriTailN3, pltOriTailN4, pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
    } else {
        ProcessVec1NoUpdateGeneralImpl512VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb,
            maxUbStart, srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4, pltOriTailN1, pltOriTailN2,
            pltOriTailN3, pltOriTailN4, pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
    }
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H
