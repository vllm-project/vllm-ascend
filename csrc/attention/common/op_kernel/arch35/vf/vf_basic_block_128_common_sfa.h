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
 * \file vf_basic_block_128_common_sfa.h
 * \brief shared __simd_callee__ helpers for 128-element aligned/unaligned SFA kernels
 */
#ifndef VF_BASIC_BLOCK_128_COMMON_SFA_H
#define VF_BASIC_BLOCK_128_COMMON_SFA_H

#include "vf_basic_block_utils.h"

using namespace regbaseutil;

namespace FaVectorApi {

template<typename T, typename T2>
__simd_callee__ inline void CastStoreExp128(
    RegTensor<float> &vreg_exp_even, RegTensor<float> &vreg_exp_odd,
    __ubuf__ T2 *&expUb, const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<half> vreg_exp_even_fp16;
    RegTensor<half> vreg_exp_odd_fp16;
    RegTensor<half> vreg_exp_fp16;
    if constexpr (IsSameType<T2, bfloat16_t>::value) {
        Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
        Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
        Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
            (RegTensor<uint16_t>&)vreg_exp_odd_bf16, storeMask);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, storeMask);
    } else if constexpr (IsSameType<T2, half>::value) {
        Cast<T2, T, castTraitZero>(vreg_exp_even_fp16, vreg_exp_even, preg_all);
        Cast<T2, T, castTraitOne>(vreg_exp_odd_fp16, vreg_exp_odd, preg_all);
        Or((RegTensor<uint16_t>&)vreg_exp_fp16, (RegTensor<uint16_t>&)vreg_exp_even_fp16,
            (RegTensor<uint16_t>&)vreg_exp_odd_fp16, storeMask);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)expUb), vreg_exp_fp16, blockStride, repeatStride, storeMask);
    }
}

template<typename T>
__simd_callee__ inline void ExpSumReduceStore128(
    RegTensor<float> &vreg_exp_sum, RegTensor<float> &vreg_exp_even, RegTensor<float> &vreg_exp_odd,
    UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb, MaskReg &preg_all)
{
    Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
    Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_exp_sum, vreg_exp_sum, preg_all);
    StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
}

template<typename T>
__simd_callee__ inline void TailScaleStoreMax128(
    RegTensor<float> &vreg_input_x, RegTensor<float> &vreg_input_x_unroll,
    RegTensor<float> &vreg_input_x_unroll_new, RegTensor<float> &vreg_min,
    RegTensor<float> &vreg_max_tmp,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    const float scale, MaskReg &preg_all, MaskReg &preg_ori_tail_n,
    MaskReg &preg_tail_n)
{
    LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
    LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
    Muls(vreg_input_x, vreg_input_x, scale, preg_all);
    Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
    Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll_new, preg_tail_n);
    Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
}

template<typename T>
__simd_callee__ inline void AlignedScaleStoreMax128(
    RegTensor<float> &vreg_input_x, RegTensor<float> &vreg_input_x_unroll,
    RegTensor<float> &vreg_max_tmp,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    const float scale, MaskReg &preg_all)
{
    LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
    LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
    Muls(vreg_input_x, vreg_input_x, scale, preg_all);
    Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
    Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
}

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_128_COMMON_SFA_H
