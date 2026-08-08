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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_regbase_v2.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H

#include "kernel_tensor.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * only 128*128 support
 * only high performance support (expSum expMax use fp32)************************************************************************************************* */
// originN = 128, No update
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_input_x;
        MicroAPI::RegTensor<T> vreg_input_max;
        MicroAPI::RegTensor<T> vreg_max_brc;
        MicroAPI::RegTensor<float> vreg_exp_sum;
        MicroAPI::RegTensor<float> vreg_exp_even;
        MicroAPI::RegTensor<float> vreg_exp_odd;

        MicroAPI::UnalignReg ureg_max;
        MicroAPI::UnalignReg ureg_exp_sum;

        MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_s8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::VL128>();

        MicroAPI::RegTensor<half> vreg_exp_res;
        MicroAPI::RegTensor<half> vreg_muls_res;
        MicroAPI::RegTensor<T2> vreg_cast;
        MicroAPI::RegTensor<T2> vreg_res;      

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vreg_input_x, srcUb + i * sInner); // fp16 data 256B one row
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_x, vreg_input_x,
                    scale, preg_all_b16); // Muls(scale)
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>
                    (srcUb + i * sInner, vreg_input_x, preg_all_b16);  
            MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_input_max, vreg_input_x, preg_all_b16);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                    (maxUb, vreg_input_max, ureg_max, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (maxUb, ureg_max, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>
                    (vreg_max_brc, maxUbStart + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vreg_input_x, srcUb + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                    MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_input_x, vreg_max_brc, preg_all_b16);

            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T2 *&)expUb, vreg_exp_res, blockStride,
                    repeatStride, preg_all_b16);
            
            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            MicroAPI::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            MicroAPI::ReduceSum<float, float, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                    (expSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, ureg_exp_sum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float * expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_input_x_1;
        MicroAPI::RegTensor<T> vreg_input_x_2;
        MicroAPI::RegTensor<T> vreg_input_max_tmp;
        MicroAPI::RegTensor<T> vreg_input_max;
        MicroAPI::RegTensor<T> vreg_max_brc;
        MicroAPI::RegTensor<float> vreg_exp_sum;
        MicroAPI::RegTensor<float> vreg_exp_even;
        MicroAPI::RegTensor<float> vreg_exp_odd;

        MicroAPI::UnalignReg ureg_max;
        MicroAPI::UnalignReg ureg_exp_sum;

        MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_s8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::VL128>();

        MicroAPI::RegTensor<T> vreg_exp_res;
        MicroAPI::RegTensor<T> vreg_exp_res_1;
        MicroAPI::RegTensor<T> vreg_exp_res_2;

        for (uint16_t i = 0; i < rows; ++i) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_x_1, vreg_input_x_1, scale, preg_all_b16);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_x_2, vreg_input_x_2, scale, preg_all_b16);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vreg_input_x_1, preg_all_b16);  
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vreg_input_x_2, preg_all_b16);
                MicroAPI::Max(vreg_input_max_tmp, vreg_input_x_1, vreg_input_x_2, preg_all_b16);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_max_tmp, preg_all_b16);
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxUb, vreg_input_max, ureg_max, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxUb, ureg_max, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_max_brc, maxUbStart + i);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
                MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                        MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res_1, vreg_input_x_1, vreg_max_brc, preg_all_b16);
                MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                        MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res_2, vreg_input_x_2, vreg_max_brc, preg_all_b16);
                static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
                static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                        MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T2 *&)expUb1, vreg_exp_res_1, blockStride,
                        repeatStride, preg_all_b16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                        MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T2 *&)expUb2, vreg_exp_res_2, blockStride,
                        repeatStride, preg_all_b16);
    
                // x_sum = sum(x_exp, axis=-1, keepdims=True)
                MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_res, vreg_exp_res_1, vreg_exp_res_2, preg_all_b16);
                MicroAPI::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
                MicroAPI::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
                MicroAPI::ReduceSum<float, float, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_sum, vreg_exp_sum, preg_all_b32);
                MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                        (expSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (expSumUb, ureg_exp_sum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510NoUpdate8(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510NoUpdateImpl128<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510NoUpdateImpl256<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

// originN = 128, Update
template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510UpdateImpl128(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ float*)inExpSumTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T * tmpMaxUbStart = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_input_x;
        MicroAPI::RegTensor<T> vreg_input_max;
        MicroAPI::RegTensor<float> vreg_exp_sum;
        MicroAPI::RegTensor<float> vreg_exp_sum_brc_even;
        MicroAPI::RegTensor<float> vreg_exp_sum_brc_odd;
        MicroAPI::RegTensor<T> vreg_in_max;
        MicroAPI::RegTensor<T> vreg_max;
        MicroAPI::RegTensor<T> vreg_exp_max;
        MicroAPI::RegTensor<float> vreg_exp_max_even;
        MicroAPI::RegTensor<float> vreg_exp_max_odd;
        MicroAPI::RegTensor<float> vreg_in_exp_sum_even;
        MicroAPI::RegTensor<float> vreg_in_exp_sum_odd;
        MicroAPI::RegTensor<float> vreg_exp_sum_update_even;
        MicroAPI::RegTensor<float> vreg_exp_sum_update_odd;
        MicroAPI::RegTensor<T> vreg_exp_res;
        MicroAPI::RegTensor<float> vreg_exp_even;
        MicroAPI::RegTensor<float> vreg_exp_odd;

        MicroAPI::UnalignReg ureg_max;
        MicroAPI::UnalignReg ureg_exp_sum;

        MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_s8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::VL128>();

        MicroAPI::RegTensor<half> vreg_cast_b16;
        MicroAPI::RegTensor<half> vreg_cast_b16_unroll;
        MicroAPI::RegTensor<half> vreg_cast_res;
        MicroAPI::RegTensor<half> vreg_muls_res;
        // MicroAPI::RegTensor<half> vregAddsRes;
        MicroAPI::RegTensor<int8_t> vreg_cast;
        MicroAPI::RegTensor<int8_t> vreg_res;

        static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE,
            MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vreg_input_x, srcUb + i * sInner);
            MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_input_x, vreg_input_x, scale, preg_all_b16);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>
                    (srcUb + i * sInner, vreg_input_x, preg_all_b16);
            MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_input_max, vreg_input_x, preg_all_b16);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                    (tmpMaxUb, vreg_input_max, ureg_max, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpMaxUb, ureg_max, 0);
        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                (vreg_in_max, inMaxUb);
        mem_bar(VST_VLD);
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                (vreg_input_max, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_max, vreg_input_max, vreg_in_max, preg_all_b16);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_max, vreg_in_max, vreg_max, preg_all_b16);
        MicroAPI::Cast<float, half, castTrait0>(vreg_exp_max_even, vreg_exp_max, preg_all_b16);
        MicroAPI::Cast<float, half, castTrait1>(vreg_exp_max_odd, vreg_exp_max, preg_all_b16);
        // store exp_max
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>
                (expMaxUb, vreg_exp_max_even, vreg_exp_max_odd, preg_all_b32);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>
                (maxUb, vreg_max, preg_all_b16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>
                    (vreg_max, maxUb + i);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>
                    (vreg_input_x, srcUb + i * sInner);
            MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                    MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res, vreg_input_x, vreg_max, preg_all_b16);

            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(vreg_muls_res, vreg_exp_res, (half)quantScaleP, preg_all_b16);
            // MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(vregAddsRes, vreg_muls_res, (half)offset, preg_all_b16);

            MicroAPI::Cast<int8_t, half, castTrait0>(vreg_cast, vreg_muls_res, preg_all_b16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vreg_res, 
                    (MicroAPI::RegTensor<uint16_t>&)vreg_cast);

            MicroAPI::DataCopy<int8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ int8_t *&) expUb),
                    vreg_res, blockStride, repeatStride, preg_s8);  

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            MicroAPI::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
            MicroAPI::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
            MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
            MicroAPI::ReduceSum<float, float, MicroAPI::MaskMergeMode::ZEROING>
                    (vreg_exp_sum, vreg_exp_sum, preg_all_b32);
            MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                    (tmpExpSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, ureg_exp_sum, 0);
        mem_bar(VST_VLD);

        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vreg_in_exp_sum_even, vreg_in_exp_sum_odd, inExpSumUb);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vreg_exp_sum_brc_even, vreg_exp_sum_brc_odd, tmpExpSumUbStart);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_even, vreg_exp_max_even, vreg_in_exp_sum_even, preg_all_b32);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_odd, vreg_exp_max_odd, vreg_in_exp_sum_odd, preg_all_b32);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_even, vreg_exp_sum_update_even, vreg_exp_sum_brc_even, preg_all_b32);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_odd, vreg_exp_sum_update_odd, vreg_exp_sum_brc_odd, preg_all_b32);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>
                (expSumUb, vreg_exp_sum_update_even, vreg_exp_sum_update_odd, preg_all_b32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510UpdateImpl256(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float * expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * inExpSumUb = (__ubuf__ float*)inExpSumTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float * tmpExpSumUb = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float * tmpExpSumUbStart = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T * tmpMaxUbStart = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg_input_x_1;
        MicroAPI::RegTensor<T> vreg_input_x_2;
        MicroAPI::RegTensor<T> vreg_input_max;
        MicroAPI::RegTensor<T> vreg_input_max_tmp;
        MicroAPI::RegTensor<float> vreg_exp_sum;
        MicroAPI::RegTensor<float> vreg_exp_sum_brc_even;
        MicroAPI::RegTensor<float> vreg_exp_sum_brc_odd;
        MicroAPI::RegTensor<T> vreg_in_max;
        MicroAPI::RegTensor<T> vreg_max;
        MicroAPI::RegTensor<T> vreg_exp_max;
        MicroAPI::RegTensor<float> vreg_exp_max_even;
        MicroAPI::RegTensor<float> vreg_exp_max_odd;
        MicroAPI::RegTensor<float> vreg_in_exp_sum_even;
        MicroAPI::RegTensor<float> vreg_in_exp_sum_odd;
        MicroAPI::RegTensor<float> vreg_exp_sum_update_even;
        MicroAPI::RegTensor<float> vreg_exp_sum_update_odd;
        MicroAPI::RegTensor<T> vreg_exp_res;
        MicroAPI::RegTensor<T> vreg_exp_res_1;
        MicroAPI::RegTensor<T> vreg_exp_res_2;
        MicroAPI::RegTensor<float> vreg_exp_even;
        MicroAPI::RegTensor<float> vreg_exp_odd;

        MicroAPI::UnalignReg ureg_max;
        MicroAPI::UnalignReg ureg_exp_sum;

        MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b16 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_all_b8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg_s8 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::VL128>();

        MicroAPI::RegTensor<T> vreg_cast_b16;
        MicroAPI::RegTensor<T> vreg_cast_b16_unroll;
        MicroAPI::RegTensor<T> vreg_cast_res;
        MicroAPI::RegTensor<T> vreg_muls_res;
        MicroAPI::RegTensor<int8_t> vreg_cast;
        MicroAPI::RegTensor<int8_t> vreg_res;

        static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE,
            MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < rows; ++i) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_x_1, vreg_input_x_1, scale, preg_all_b16);
                MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_x_2, vreg_input_x_2, scale, preg_all_b16);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vreg_input_x_1, preg_all_b16);  
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vreg_input_x_2, preg_all_b16);
                MicroAPI::Max(vreg_input_max_tmp, vreg_input_x_1, vreg_input_x_2, preg_all_b16);
                MicroAPI::ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max, vreg_input_max_tmp, preg_all_b16);
                MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vreg_input_max, ureg_max, 1);
        }
        MicroAPI::DataCopyUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, ureg_max, 0);
        // load history max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_in_max, inMaxUb);
        mem_bar(VST_VLD);
        // load current max
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_max, tmpMaxUbStart);
        // max(history max, current max)
        MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(vreg_max, vreg_input_max, vreg_in_max, preg_all_b16);
        // exp_max = exp(inmax - x_max)
        MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_max, vreg_in_max, vreg_max, preg_all_b16);
        MicroAPI::Cast<float, half, castTrait0>(vreg_exp_max_even, vreg_exp_max, preg_all_b16);
        MicroAPI::Cast<float, half, castTrait1>(vreg_exp_max_odd, vreg_exp_max, preg_all_b16);
        // store exp_max
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>
                (expMaxUb, vreg_exp_max_even, vreg_exp_max_odd, preg_all_b32);
        // store max
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>
                (maxUb, vreg_max, preg_all_b16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(vreg_max, maxUb + i);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_1, srcUb + i * sInner);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vreg_input_x_2, srcUb + i * sInner + halfRepSize);
                MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                        MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res_1, vreg_input_x_1, vreg_max, preg_all_b16);
                MicroAPI::FusedExpSub<T, T, MicroAPI::RegLayout::ONE,
                        MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_res_2, vreg_input_x_2, vreg_max, preg_all_b16);

                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                        MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T2 *&)expUb1, vreg_exp_res_1, blockStride,
                        repeatStride, preg_all_b16);
                MicroAPI::DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                        MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T2 *&)expUb2, vreg_exp_res_2, blockStride,
                        repeatStride, preg_all_b16);

                // x_sum = sum(x_exp, axis=-1, keepdims=True)
                MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_res, vreg_exp_res_1, vreg_exp_res_2, preg_all_b16);
                MicroAPI::Cast<float, half, castTrait0>(vreg_exp_even, vreg_exp_res, preg_all_b16);
                MicroAPI::Cast<float, half, castTrait1>(vreg_exp_odd, vreg_exp_res, preg_all_b16);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all_b32);
                MicroAPI::ReduceSum<float, float, MicroAPI::MaskMergeMode::ZEROING>
                        (vreg_exp_sum, vreg_exp_sum, preg_all_b32);
                MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                        (tmpExpSumUb, vreg_exp_sum, ureg_exp_sum, 1);
        }
        MicroAPI::DataCopyUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (tmpExpSumUb, ureg_exp_sum, 0);
        mem_bar(VST_VLD);

        // x_sum = sum(exp_max * in_sum + x_sum)
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vreg_in_exp_sum_even, vreg_in_exp_sum_odd, inExpSumUb);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>
                (vreg_exp_sum_brc_even, vreg_exp_sum_brc_odd, tmpExpSumUbStart);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_even, vreg_exp_max_even, vreg_in_exp_sum_even, preg_all_b32);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_odd, vreg_exp_max_odd, vreg_in_exp_sum_odd, preg_all_b32);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_even, vreg_exp_sum_update_even, vreg_exp_sum_brc_even, preg_all_b32);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>
                (vreg_exp_sum_update_odd, vreg_exp_sum_update_odd, vreg_exp_sum_brc_odd, preg_all_b32);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>
                (expSumUb, vreg_exp_sum_update_even, vreg_exp_sum_update_odd, preg_all_b32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510Update8(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510UpdateImpl128<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510UpdateImpl256<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

/*
 * @ingroup SoftmaxFlashV510
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] pseTensor, reserved
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] originN, input origin colums, support range for sInner is: 0 < sInner <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isBmm2Concat, reserved
 * @param [in] isUpdate, enable flash mode
 * @param [in] mode
 *  mode 0: 64 < originN <= 128, and originN is not 8 aligned
 *  mode 1: 64 < originN <= 128, and originN is 8 aligned
 *  mode 2: 0 < originN <= 64
 * @param [in] hasAtten, indicates whether there is atten_mask
 * @param [in] hasPse, indicates whether there is pse_shift
 */
template <typename T, typename T2, bool isUpdate = false, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0>
__aicore__ inline void SoftmaxFlashV510_VF(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
    const LocalTensor<T>& inPseTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t originN, const T scale, const T minValue, float quantScaleP)
{
    constexpr uint32_t blockU8 = 32;
    uint32_t blockN = 0;
    if constexpr (IsSameType<T2, int8_t>::value) {
        blockN = 32;
    } else {
        blockN = 16;
    }
    uint16_t blockStride = sOuter  >> 1 | 0x1;

    if constexpr (!isUpdate) {
        SoftmaxFlashV510NoUpdate8<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510Update8<T, T2, mode, sOuter, sInner>(dstTensor,
            expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor, inPseTensor,
            sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}
} // namespace

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H