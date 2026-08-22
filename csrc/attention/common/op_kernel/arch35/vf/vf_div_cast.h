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
 * \file vf_div_cast.h
 * \brief
 */
#ifndef VF_DIV_CAST_INTERFACE_H
#define VF_DIV_CAST_INTERFACE_H

#include "kernel_tensor.h"

namespace FaVectorApi {
template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImpl64VF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    RegTensor<T> vreg_src;
    RegTensor<T> vreg_div;
    RegTensor<T> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_even_bf16;
    RegTensor<bfloat16_t> vreg_div_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    // half
    RegTensor<half> vreg_div_even_f16;
    RegTensor<half> vreg_div_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        LoadAlign(vreg_src, srcUb + i * srcD);
        Div(vreg_div, vreg_src, vreg_exp_sum, preg_all);

        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div, preg_all);
        } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_bf16, vreg_div, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_bf16, preg_all);
        } else {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_f16, vreg_div, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_f16, preg_all);
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImpl64(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImpl64VF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImpl128VF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    RegTensor<float> vreg_src_even, vreg_src_odd;
    RegTensor<float> vreg_div_even, vreg_div_odd;
    RegTensor<float> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_even_bf16;
    RegTensor<bfloat16_t> vreg_div_odd_bf16;
    RegTensor<bfloat16_t> vreg_cast_bf16;
    // half
    RegTensor<half> vreg_div_even_f16;
    RegTensor<half> vreg_div_odd_f16;
    RegTensor<half> vreg_cast_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            LoadAlign(vreg_src_even, srcUb + i * srcD);
            LoadAlign(vreg_src_odd, srcUb + i * srcD + (srcD >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_src_even, vreg_src_odd, srcUb + i * srcD);
        }
        Div(vreg_div_even, vreg_src_even, vreg_exp_sum, preg_all);
        Div(vreg_div_odd, vreg_src_odd, vreg_exp_sum, preg_all);

        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_even, preg_all);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + (srcD >> 1), vreg_div_odd, preg_all);
        } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_bf16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_bf16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_bf16, (RegTensor<uint16_t>&)vreg_div_even_bf16,
                (RegTensor<uint16_t>&)vreg_div_odd_bf16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_bf16, preg_all_b16);
        } else {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_f16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_f16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_f16, (RegTensor<uint16_t>&)vreg_div_even_f16, (RegTensor<uint16_t>&)vreg_div_odd_f16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_f16, preg_all_b16);
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImpl128(const LocalTensor<OUTPUT_T>& dstTensor,
                                      const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
                                      const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImpl128VF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImplGeneralVF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb,
    __ubuf__ float * expSumUb, const uint16_t m)
{
    RegTensor<float> vreg_src;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    // half
    RegTensor<half> vreg_div_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    constexpr uint16_t dLoops = srcD >> 6;
    constexpr uint16_t floatRepSize = 64;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_src, srcUb + i * srcD + j * floatRepSize);
            Div(vreg_div, vreg_src, vreg_exp_sum, preg_all);

            if constexpr (IsSameType<OUTPUT_T, float>::value) {
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div, preg_all);
            } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
                Cast<OUTPUT_T, T, castTraitZero>(vreg_div_bf16, vreg_div, preg_all_b16);
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div_bf16, preg_all);
            } else {
                Cast<OUTPUT_T, T, castTraitZero>(vreg_div_f16, vreg_div, preg_all_b16);
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div_f16, preg_all);
            }
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImplGeneral(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImplGeneralVF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

/*
 * @ingroup DivCast
 * @brief compute, dstTensor = cast(srcTensor / expSumTensor)
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor, shape is [m, 8]
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 * @param [in] srcD, should be 64 or 128
 */
template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCast(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    if constexpr (srcD == 64) {
        DivCastImpl64<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    } else if constexpr (srcD == 128) {
        DivCastImpl128<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    } else {    
        DivCastImplGeneral<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    }
}
} // namespace

#endif // VF_DIV_CAST_INTERFACE_H
