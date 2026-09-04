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
 * \file vf_basic_block_unaligned64_common_sfa.h
 * \brief shared __simd_callee__ helpers for unaligned64 SFA kernels
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED64_COMMON_SFA_H
#define VF_BASIC_BLOCK_UNALIGNED64_COMMON_SFA_H

#include "vf_basic_block_utils.h"

using namespace regbaseutil;

namespace FaVectorApi {

template<typename T, typename T2>
__simd_callee__ inline void CastStoreExp64(
    RegTensor<float> &vreg_exp, __ubuf__ T2 *&expUb,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all_b16, MaskReg &storeMask)
{
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    RegTensor<half> vreg_exp_fp16;
    RegTensor<half> vreg_dst_even_fp16;
    RegTensor<half> vreg_dst_odd_fp16;
    if constexpr (IsSameType<T2, bfloat16_t>::value) {
        Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
        DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16, vreg_exp_bf16, vreg_exp_bf16);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, storeMask);
    } else if constexpr (IsSameType<T2, half>::value) {
        Cast<T2, T, castTraitZero>(vreg_exp_fp16, vreg_exp, preg_all_b16);
        DeInterleave(vreg_dst_even_fp16, vreg_dst_odd_fp16, vreg_exp_fp16, vreg_exp_fp16);
        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)expUb), vreg_dst_even_fp16, blockStride, repeatStride, storeMask);
    }
}

template<typename T>
__simd_callee__ inline void ExpSumReduceStore64(
    RegTensor<float> &vreg_exp_sum, RegTensor<float> &vreg_exp,
    UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb,
    MaskReg &preg_ori_src_n)
{
    Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_exp_sum, vreg_exp, preg_ori_src_n);
    StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
}

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED64_COMMON_SFA_H
