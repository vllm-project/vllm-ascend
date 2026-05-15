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
 * \file vf_flashupdate_new_scfa.h
 * \brief
 */
#ifndef FLASH_UPDATE_NEW_INTERFACE_SCFA_H
#define FLASH_UPDATE_NEW_INTERFACE_SCFA_H

#include "vf_basic_block_utils.h"
#include "../util_regbase.h"
#include "../kv_quant_sparse_attn_sharedkv_common_arch35.h"

namespace SCFaVectorApi {
constexpr uint16_t REDUCE_SIZE = 1;
/* **************************************************************************************************
 * FlashUpdate, fp32
 * ************************************************************************************************* */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__simd_vf__ inline void FlashUpdateBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, const uint16_t m)
{
    constexpr uint16_t dLoops = srcD / floatRepSize;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_max;
    AscendC::MicroAPI::RegTensor<float> vreg_input_pre;
    AscendC::MicroAPI::RegTensor<float> vreg_input_cur;
    AscendC::MicroAPI::RegTensor<float> vreg_mul;
    AscendC::MicroAPI::RegTensor<float> vreg_add;

    AscendC::MicroAPI::MaskReg preg_all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);  // [m,8]

        for (uint16_t j = 0; j < dLoops; ++j) {
            AscendC::MicroAPI::LoadAlign(vreg_input_pre, preUb + i * srcD + j * floatRepSize);
            AscendC::MicroAPI::LoadAlign(vreg_input_cur, curUb + i * srcD + j * floatRepSize);
            AscendC::MicroAPI::MulDstAdd(vreg_input_pre, vreg_exp_max, vreg_input_cur, preg_all);
            AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * srcD + j * floatRepSize, vreg_input_pre, preg_all);
        }
    }
}

/*
 * @ingroup FlashUpdate
 * @brief compute, dstTensor = preTensor * expMaxTensor + curTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] srcD, input colums, should be 32 bytes aligned
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void FlashUpdateNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    FlashUpdateBasicVF<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE>(dstUb, curUb, preUb, expMaxUb, m);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, const uint16_t m)
{
    constexpr uint16_t dLoops = srcD / floatRepSize;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_max;
    AscendC::MicroAPI::RegTensor<float> vreg_input_pre;
    AscendC::MicroAPI::RegTensor<float> vreg_input_cur;
    AscendC::MicroAPI::RegTensor<float> vreg_mul;
    AscendC::MicroAPI::RegTensor<float> vreg_add;
    AscendC::MicroAPI::RegTensor<float> vreg_div;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_sum;

    AscendC::MicroAPI::MaskReg preg_all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);

        for (uint16_t j = 0; j < dLoops; ++j) {
            AscendC::MicroAPI::LoadAlign(vreg_input_pre, preUb + i * srcD + j * floatRepSize);
            AscendC::MicroAPI::LoadAlign(vreg_input_cur, curUb + i * srcD + j * floatRepSize);
            AscendC::MicroAPI::MulDstAdd(vreg_input_pre, vreg_exp_max, vreg_input_cur, preg_all);
            AscendC::MicroAPI::Div(vreg_div, vreg_input_pre, vreg_exp_sum, preg_all);
            AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * srcD + j * floatRepSize, vreg_div, preg_all);
        }
    }
}

/*
 * @ingroup FlashUpdateLast
 * @brief compute, dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] srcD, input colums, 32 bytes align
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void FlashUpdateLastNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor, 
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor, uint16_t m)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE>(dstUb, curUb, preUb, expMaxUb, expSumUb, m);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__simd_vf__ inline void LastDivNewVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    const uint16_t dLoops = srcD >> 6;
    AscendC::MicroAPI::RegTensor<float> vreg_input_cur;
    AscendC::MicroAPI::RegTensor<float> vreg_div;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_sum;
    AscendC::MicroAPI::MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    uint32_t sreg_init = srcD;
    AscendC::MicroAPI::MaskReg preg_update = UpdateMask<float>(sreg_init);

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            AscendC::MicroAPI::LoadAlign(vreg_input_cur, curUb + i * srcD + j * floatRepSize);
            AscendC::MicroAPI::Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_all);
            AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * srcD + j * floatRepSize, vreg_div, preg_update);
        }
    }
}

// dstTensor = curTensor / expSumTensor, curTensor: [64,128], expSumTensor: [64,8]
template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__aicore__ inline void LastDivNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    LastDivNewVF<T, INPUT_T, OUTPUT_T, srcD>(dstUb, curUb, expSumUb, m);
}
} // namespace

#endif // FLASH_UPDATE_NEW_INTERFACE_SCFA_H
