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
 * \file vf_basic_block_unaligned1024_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned1024_common.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1UpdateGeneralImpl1024VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4,
    __ubuf__ T2 * expUb5, __ubuf__ T2 * expUb6, __ubuf__ T2 * expUb7, __ubuf__ T2 * expUb8,
    __ubuf__ OUTPUT_T * pseUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb,
    __ubuf__ T * expMaxUb, __ubuf__ T * inMaxUb, __ubuf__ T * tmpExpSumUb,
    __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2, VREG_FLOAT_MASKUB_16_DECL,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3,
    const uint32_t oriTailN4, const uint32_t oriTailN5, const uint32_t oriTailN6,
    const uint32_t oriTailN7, const uint32_t oriTailN8, const uint32_t tailN1,
    const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4,
    const uint32_t tailN5, const uint32_t tailN6, const uint32_t tailN7,
    const uint32_t tailN8, uint32_t pltOriTailN1, uint32_t pltOriTailN2,
    uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltOriTailN5,
    uint32_t pltOriTailN6, uint32_t pltOriTailN7, uint32_t pltOriTailN8,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3,
    uint32_t pltTailN4, uint32_t pltTailN5, uint32_t pltTailN6,
    uint32_t pltTailN7, uint32_t pltTailN8, uint32_t pltN, float divValue,
    const uint16_t m, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const T minValue)
{
    VREG_FLOAT_DECL_16(vreg_sel);
    RegTensor<float> vreg_sel9_new;
    RegTensor<float> vreg_sel10_new;
    RegTensor<float> vreg_sel11_new;
    RegTensor<float> vreg_sel12_new;
    RegTensor<float> vreg_sel13_new;
    RegTensor<float> vreg_sel14_new;
    RegTensor<float> vreg_sel15_new;
    RegTensor<float> vreg_sel16_new;

    VREG_FLOAT_DECL_16(vreg_input_x);

    RegTensor<float> vreg_input_x9_new;
    RegTensor<float> vreg_input_x10_new;
    RegTensor<float> vreg_input_x11_new;
    RegTensor<float> vreg_input_x12_new;
    RegTensor<float> vreg_input_x13_new;
    RegTensor<float> vreg_input_x14_new;
    RegTensor<float> vreg_input_x15_new;
    RegTensor<float> vreg_input_x16_new;

    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_new;

    RegTensor<float> vreg_exp_sum1;

    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;

    VREG_FLOAT_DECL_EXP_PAIRS_8(vreg_exp);

    VREG_FLOAT_DECL_16(vreg_pse);

    VREG_FLOAT_DECL_16(vreg_alibi);

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
    MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
    MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);
    MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);
    MaskReg preg_ori_tail_n5 = UpdateMask<T>(pltOriTailN5);
    MaskReg preg_ori_tail_n6 = UpdateMask<T>(pltOriTailN6);
    MaskReg preg_ori_tail_n7 = UpdateMask<T>(pltOriTailN7);
    MaskReg preg_ori_tail_n8 = UpdateMask<T>(pltOriTailN8);

    ProcessVec1MainLoop1024<T, T2, hasAtten, pseMode>(
        ARG_FLOAT_INPUT_X_16_NO_IDX,
        ARG_FLOAT_INPUT_X_NEW_8,
        ARG_FLOAT_PSE_16,
        ARG_FLOAT_SEL_16,
        ARG_FLOAT_SEL_NEW_8,
        ARG_FLOAT_ALIBI_16_VREG,
        vreg_input_max, preg_all, preg_all_b16,
        preg_ori_tail_n1, preg_ori_tail_n2, preg_ori_tail_n3, preg_ori_tail_n4,
        preg_ori_tail_n5, preg_ori_tail_n6, preg_ori_tail_n7, preg_ori_tail_n8,
        tmpMaxUb, pseUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8,
        maskUb9, maskUb10, maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16,
        minValue, m, slopes, scale, posShift, pseStride, nPadding, ureg_max);

    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);

    Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max, tmpMaxUb2 + i);

        LoadInputDinterleave8<T>(ARG_FLOAT_INPUT_X_16);

        ExpSub16(vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
            vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
            vreg_exp_even5, vreg_exp_odd5, vreg_exp_even6, vreg_exp_odd6,
            vreg_exp_even7, vreg_exp_odd7, vreg_exp_even8, vreg_exp_odd8,
            vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
            vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
            vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
            vreg_max, preg_all);

        ExpSumReduceStore1024<float, T2>(vreg_exp_sum1,
            vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
            vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
            vreg_exp_even5, vreg_exp_odd5, vreg_exp_even6, vreg_exp_odd6,
            vreg_exp_even7, vreg_exp_odd7, vreg_exp_even8, vreg_exp_odd8,
            ureg_exp_sum, tmpExpSumUb, preg_all);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            CastStoreExpBf16_1024<T2>(ARG_FLOAT_CAST_STORE_EXP_1024);
        } else if constexpr (IsSameType<T2, half>::value) {
            CastStoreExpF16_1024<T2>(ARG_FLOAT_CAST_STORE_EXP_1024);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}

// noupdate, 512 < originN <= 1024
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1UpdateGeneralImpl1024(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 8 < floatRepSize ? originN - floatRepSize * 8 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 9) <= 0 ? 0 : originN - floatRepSize * 9;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 10) <= 0 ? 0 : originN - floatRepSize * 10;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 11) <= 0 ? 0 : originN - floatRepSize * 11;
    const uint32_t oriTailN5 = static_cast<int32_t>(originN - floatRepSize * 12) <= 0 ? 0 : originN - floatRepSize * 12;
    const uint32_t oriTailN6 = static_cast<int32_t>(originN - floatRepSize * 13) <= 0 ? 0 : originN - floatRepSize * 13;
    const uint32_t oriTailN7 = static_cast<int32_t>(originN - floatRepSize * 14) <= 0 ? 0 : originN - floatRepSize * 14;
    const uint32_t oriTailN8 = static_cast<int32_t>(originN - floatRepSize * 15) <= 0 ? 0 : originN - floatRepSize * 15;

    const uint32_t tailN1 = s2BaseSize - floatRepSize * 8;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 9;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 10;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 11;
    const uint32_t tailN5 = s2BaseSize - floatRepSize * 12;
    const uint32_t tailN6 = s2BaseSize - floatRepSize * 13;
    const uint32_t tailN7 = s2BaseSize - floatRepSize * 14;
    const uint32_t tailN8 = s2BaseSize - floatRepSize * 15;

    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;
    uint32_t pltOriTailN5 = oriTailN5;
    uint32_t pltOriTailN6 = oriTailN6;
    uint32_t pltOriTailN7 = oriTailN7;
    uint32_t pltOriTailN8 = oriTailN8;

    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    uint32_t pltTailN5 = tailN5;
    uint32_t pltTailN6 = tailN6;
    uint32_t pltTailN7 = tailN7;
    uint32_t pltTailN8 = tailN8;

    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb5 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 4 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb6 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 5 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb7 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 6 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb8 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 7 * ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    DO_MASKUB_INIT_16(maskTensor);

    ProcessVec1UpdateGeneralImpl1024VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
        expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8, pseUb, maxUb, srcUb, expMaxUb,
        inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8, maskUb9, maskUb10, maskUb11,
        maskUb12, maskUb13, maskUb14, maskUb15, maskUb16, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2,
        oriTailN3, oriTailN4, oriTailN5, oriTailN6, oriTailN7, oriTailN8, tailN1, tailN2, tailN3, tailN4, tailN5,
        tailN6, tailN7, tailN8, pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4, pltOriTailN5, pltOriTailN6,
        pltOriTailN7, pltOriTailN8, pltTailN1, pltTailN2, pltTailN3, pltTailN4,
        pltTailN5, pltTailN6, pltTailN7, pltTailN8,
        pltN, divValue, m, pseStride, slopes, posShift, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H
