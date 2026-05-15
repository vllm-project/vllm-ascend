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
 * \file vf_basic_block_aligned128_update_scfa.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED128_UPDATE_SCFA_H
#define VF_BASIC_BLOCK_ALIGNED128_UPDATE_SCFA_H

#include "vf_basic_block_utils.h"
#include "../util_regbase.h"
#include "../kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace regbaseutil;

namespace SCFaVectorApi {
// update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128>
__simd_vf__ void ProcessVec1UpdateImpl128VF(
    __ubuf__ T2 * expUb,  __ubuf__ T * srcUb, __ubuf__ T * inMaxUb,
    __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2, const uint32_t blockStride, const uint32_t repeatStride, 
    const uint16_t m, const T scale, const T minValue)
{
    AscendC::MicroAPI::RegTensor<float> vreg_input_x;
    AscendC::MicroAPI::RegTensor<float> vreg_input_x_unroll;
    AscendC::MicroAPI::RegTensor<float> vreg_max_tmp;
    AscendC::MicroAPI::RegTensor<float> vreg_in_max;
    AscendC::MicroAPI::RegTensor<float> vreg_max_new;
    AscendC::MicroAPI::RegTensor<float> vreg_max_brc;
    AscendC::MicroAPI::RegTensor<float> vreg_cur_max;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_sum;
    AscendC::MicroAPI::RegTensor<float> vreg_in_exp_sum;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_even;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_odd;

    // bfloat16_t
    AscendC::MicroAPI::RegTensor<bfloat16_t> vreg_exp_even_bf16;
    AscendC::MicroAPI::RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    AscendC::MicroAPI::RegTensor<bfloat16_t> vreg_exp_bf16;

    AscendC::MicroAPI::UnalignRegForStore ureg_max;
    AscendC::MicroAPI::UnalignRegForStore ureg_exp_sum;
    AscendC::MicroAPI::MaskReg preg_all =  AscendC::MicroAPI::CreateMask<float,  AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg_all_b16 =  AscendC::MicroAPI::CreateMask<uint16_t,  AscendC::MicroAPI::MaskPattern::ALL>();

    // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        AscendC::MicroAPI::LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
    
        AscendC::MicroAPI::Muls(vreg_input_x, vreg_input_x, scale, preg_all);  // Muls(scale)
        AscendC::MicroAPI::Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_all);
        AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
           (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
        AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
           (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
        AscendC::MicroAPI::Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
           vreg_max_tmp, vreg_max_tmp, preg_all);
        AscendC::MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), vreg_max_tmp, ureg_max, 1);
    }
    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
           ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    AscendC::MicroAPI::LoadAlign(vreg_in_max, inMaxUb);
    AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    AscendC::MicroAPI::LoadAlign(vreg_cur_max, tmpMaxUb2); // 获取新的max[s1, 1]
    AscendC::MicroAPI::Max(vreg_max_new, vreg_cur_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
    AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
       (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, tmpMaxUb2 + i);
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
           vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
        AscendC::MicroAPI::ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        AscendC::MicroAPI::ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        AscendC::MicroAPI::Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
           vreg_exp_sum, vreg_exp_sum, preg_all);
        AscendC::MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            AscendC::MicroAPI::Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            AscendC::MicroAPI::Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            AscendC::MicroAPI::Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
            (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
            AscendC::MicroAPI::StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
        }
    }
    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}

// update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__aicore__ inline void ProcessVec1UpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& inMaxTensor, 
    const LocalTensor<T>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    ProcessVec1UpdateImpl128VF <T, T2, s1BaseSize, s2BaseSize>(
        expUb, srcUb, inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2, blockStride, repeatStride, m, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_ALIGNED128_UPDATE_SCFA_H
