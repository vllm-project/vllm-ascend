/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_mul_sel_softmaxflashv2_cast_nz_sfa.h
 * \brief
 */
#ifndef MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_SFA_INTERFACE_H
#define MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_SFA_INTERFACE_H

#include "vf_basic_block_aligned128_no_update_sfa.h"
#include "vf_basic_block_aligned128_update_sfa.h"
#include "vf_basic_block_unaligned64_update_sfa.h"
#include "vf_basic_block_unaligned64_no_update_sfa.h"
#include "vf_basic_block_unaligned128_no_update_sfa.h"
#include "vf_basic_block_unaligned128_update_sfa.h"

using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using AscendC::LocalTensor;

enum class OriginNRange {
    EQ_128_SFA = 0,         // originN == 128, better performance than GT_64_AND_LTE_128 (s2BaseSize=128)
    GT_0_AND_LTE_64_SFA,    // 0 < originN <= 64 (s2BaseSize <= 64 or tail s2)
    GT_64_AND_LTE_128_SFA,  // 64 < originN <= 128, support for non-alignment (s2BaseSize=128)
    N_INVALID_SFA
};
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = OriginNRange::EQ_128_SFA>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<T>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    if constexpr (oriNRange == OriginNRange::EQ_128_SFA) {
        ProcessVec1NoUpdateImpl128<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, expSumTensor, maxTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == OriginNRange::GT_0_AND_LTE_64_SFA) {
        ProcessVec1NoUpdateImpl64<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, expSumTensor, maxTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == OriginNRange::GT_64_AND_LTE_128_SFA) {
        ProcessVec1NoUpdateGeneralImpl128<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, expSumTensor, maxTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    }
}

template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = OriginNRange::EQ_128_SFA>
__aicore__ inline void ProcessVec1Update(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<T>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    if constexpr (oriNRange == OriginNRange::EQ_128_SFA) {
        ProcessVec1UpdateImpl128<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == OriginNRange::GT_0_AND_LTE_64_SFA) {
        ProcessVec1UpdateImpl64<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else if constexpr (oriNRange == OriginNRange::GT_64_AND_LTE_128_SFA) {
        ProcessVec1UpdateGeneralImpl128<T, T2, s1BaseSize, s2BaseSize>(
            dstTensor, srcTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    }
}

template <typename T, typename T2, bool isUpdate = false, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = OriginNRange::EQ_128_SFA>
__aicore__ inline void ProcessVec1Vf(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<T>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    static_assert(IsSameType<T, float>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be float");
    static_assert((IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value),
        "VF mul_sel_softmaxflashv2_cast_nz, T2 must be half or bfloat16");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdate<T, T2, s1BaseSize, s2BaseSize, oriNRange>(
            dstTensor, srcTensor, expSumTensor, maxTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    } else {
        ProcessVec1Update<T, T2, s1BaseSize, s2BaseSize, oriNRange>(
            dstTensor, srcTensor, expSumTensor, maxTensor, inMaxTensor, sharedTmpBuffer, m, originN, scale, minValue);
    }
}

template <typename T>
__simd_vf__ inline void UpdateExpSumAndExpMaxVF(__ubuf__ T * maxUb, __ubuf__ T * inMaxUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * expSumUb, __ubuf__ T * inExpSumUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, const uint32_t m)
{
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_exp_sum;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_exp_sum_update;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    // 注意：当m大于64的时候需要开启循环
    LoadAlign(vreg_max, tmpMaxUb);
    LoadAlign(vreg_in_max, inMaxUb);
    FusedExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)maxUb, vreg_max, preg_all);
    LoadAlign(vreg_in_exp_sum, inExpSumUb);

    // x_sum = exp_max * insum + x_sum
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb);
    Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
    Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
}

template <typename T>
__aicore__ inline void SFAUpdateExpSumAndExpMax(
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor,  const LocalTensor<T>& sharedTmpBuffer, const uint32_t m)
{
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();

    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();

    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    UpdateExpSumAndExpMaxVF<T>(maxUb, inMaxUb, expMaxUb, expSumUb, inExpSumUb, tmpExpSumUb, tmpMaxUb, m);
}

template <typename T>
__simd_vf__ inline void DuplicateSumWithR0VF(__ubuf__ T * sumUb, const T R0, uint32_t m) {
    AscendC::MicroAPI::RegTensor<T> vreg_sum;
    AscendC::MicroAPI::MaskReg preg_m = AscendC::MicroAPI::UpdateMask<T>(m);
    AscendC::MicroAPI::UnalignRegForStore ureg;
    AscendC::MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_sum, R0, preg_m);
    AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(sumUb, vreg_sum, preg_m);
}

template <typename T>
__aicore__ inline void DuplicateSumWithR0(const LocalTensor<T>& sumTensor, const T R0, uint32_t m)
{
    __ubuf__ T * sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    DuplicateSumWithR0VF<T>(sumUb, R0, m);
}
} // namespace
#endif // MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_SFA_INTERFACE_H
