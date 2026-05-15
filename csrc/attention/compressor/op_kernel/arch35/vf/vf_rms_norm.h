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
 * \file vf_rms_norm.h
 * \brief
 */

#ifndef VF_RMS_NORM_H
#define VF_RMS_NORM_H
#include "kernel_tensor.h"

//repeatTimes——D轴的分块数
template <typename T, typename GammaType>
__simd_vf__ void RmsNormVFImpl(__ubuf__ T * inputBuf, __ubuf__ GammaType * gammaBuf, __ubuf__ T * outputBuf, 
                               uint32_t repeatTimes, float reciprocal, float epsilon)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSumReduce;
    MicroAPI::RegTensor<T> vregDiv;
    MicroAPI::RegTensor<T> vregSquareRoot;

    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();

    static constexpr MicroAPI::CastTrait castTraitB162B32 = {MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    MicroAPI::Duplicate<T,T>(vregSum, 0.0f);

    for(uint32_t i = 0; i < repeatTimes; ++i){
        MicroAPI::RegTensor<T> vregX;
        MicroAPI::RegTensor<T> vregXSquare;
        uint64_t loopOffset = i * FLOAT_REP_SIZE;

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregX, inputBuf + loopOffset);
        MicroAPI::Mul(vregXSquare, vregX, vregX, maskAll);
        MicroAPI::Add(vregSum, vregXSquare, vregSum, maskAll);
    }

    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, T, T, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSum, maskAll);
    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, reciprocal, maskFirst);
    MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, epsilon, maskFirst);
    MicroAPI::Sqrt(vregSquareRoot, vregSumReduce, maskFirst);
    MicroAPI::Duplicate<T, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(vregDiv, vregSquareRoot, maskAll);

    for(uint32_t i = 0; i < repeatTimes; ++i){
        MicroAPI::RegTensor<T> vregX;
        MicroAPI::RegTensor<GammaType> vregGamma;
        MicroAPI::RegTensor<T> vregGammaCast;
        uint16_t loopOffset = i * FLOAT_REP_SIZE;

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregX, inputBuf + loopOffset);
        MicroAPI::LoadAlign<GammaType, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGamma, gammaBuf + loopOffset);
        MicroAPI::Cast<T, GammaType, castTraitB162B32>(vregGammaCast, vregGamma, maskAll);

        MicroAPI::Div(vregX, vregX, vregDiv, maskAll);
        MicroAPI::Mul(vregX, vregX, vregGammaCast, maskAll);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregX, maskAll);
    }
}

/**
 * @brief RmsNormVF 对一行进行rmsnorm
 * @param outputLocal 输出tensor [row, col]，row目前均为1
 * @param inputLocal 输入tensor [row, col]
 * @param gammaLocal gamma参数tensor [row, col]
 * @param rmsNormParams rmsNrom计算所需系数，包括
          row 行数  1
          col 列数，对应headSizeCq或headSizeCkv
          reciprocal ，1/N
          epsilon，防止除零极小数
 */
template <typename T, typename GammaType>
__aicore__ inline void RmsNormVF(LocalTensor<T> outputLocal, const LocalTensor<T> inputLocal, const LocalTensor<GammaType> gammaLocal,
    float reciprocal, float epsilon, uint32_t row, uint32_t col) 
{
    uint32_t cnt = row * col;
    uint32_t repeatTimes = (cnt + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;

    __ubuf__ T * inputBuf = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ GammaType * gammaBuf = (__ubuf__ GammaType *)gammaLocal.GetPhyAddr();
    __ubuf__ T * outputBuf = (__ubuf__ T *)outputLocal.GetPhyAddr();
    
    RmsNormVFImpl<T, GammaType>(inputBuf, gammaBuf, outputBuf, repeatTimes, reciprocal, epsilon);
}


#endif