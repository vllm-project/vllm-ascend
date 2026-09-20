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

namespace MlaProlog {
constexpr uint64_t FLOAT_REP_SIZE = 64;

template <typename InType, typename GammaType, typename C, typename OutType>
__simd_vf__ void RmsNormVFImpl(__ubuf__ InType *inputBuf, __ubuf__ GammaType *gammaBuf, __ubuf__ OutType *outputBuf,
                               uint32_t cnt, uint32_t repeatTimes, const RmsNormParam rmsNormParams)
{
    MicroAPI::RegTensor<C> vregSum;
    MicroAPI::RegTensor<C> vregSumReduce;
    MicroAPI::RegTensor<C> vregDiv;
    MicroAPI::RegTensor<C> vregSquareRoot;

    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<C, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregFirst = MicroAPI::CreateMask<C, MicroAPI::MaskPattern::VL1>();

    static constexpr MicroAPI::CastTrait castTraitB162B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    MicroAPI::Duplicate<C, C>(vregSum, 0.0);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        MicroAPI::RegTensor<C> vregXCast;
        MicroAPI::RegTensor<C> vregXSquare;
        uint64_t loopOffset = i * FLOAT_REP_SIZE;

        MicroAPI::LoadAlign<C, MicroAPI::LoadDist::DIST_NORM>(vregXCast, inputBuf + loopOffset);
        MicroAPI::Mul<C, MicroAPI::MaskMergeMode::ZEROING>(vregXSquare, vregXCast, vregXCast, pregAll);
        MicroAPI::Add<C, MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregXSquare, vregSum, pregAll);
    }

    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, C, C, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSum,
                                                                                        pregAll);
    MicroAPI::Muls<C, C, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, rmsNormParams.reciprocal,
                                                           pregFirst);
    MicroAPI::Adds<C, C, MicroAPI::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, rmsNormParams.epsilon,
                                                           pregFirst);
    MicroAPI::Sqrt<C, MicroAPI::MaskMergeMode::ZEROING>(vregSquareRoot, vregSumReduce, pregFirst);
    MicroAPI::Duplicate<C, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(vregDiv, vregSquareRoot,
                                                                                            pregAll);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        MicroAPI::RegTensor<C> vregXCast;
        MicroAPI::RegTensor<GammaType> vregGamma;
        MicroAPI::RegTensor<C> vregGammaCast;
        uint16_t loopOffset = i * FLOAT_REP_SIZE;

        MicroAPI::LoadAlign<C, MicroAPI::LoadDist::DIST_NORM>(vregXCast, inputBuf + loopOffset);
        MicroAPI::LoadAlign<GammaType, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGamma, gammaBuf + loopOffset);
        MicroAPI::Cast<C, GammaType, castTraitB162B32>(vregGammaCast, vregGamma, pregAll);

        MicroAPI::Div<C, MicroAPI::MaskMergeMode::ZEROING>(vregXCast, vregXCast, vregDiv, pregAll);
        MicroAPI::Mul<C, MicroAPI::MaskMergeMode::ZEROING>(vregXCast, vregXCast, vregGammaCast, pregAll);

        MicroAPI::StoreAlign<OutType, MicroAPI::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregXCast, pregAll);
    }
}

/**
 * @brief RmsNormVF 对一行进行rmsnorm
 * @param outputLocal 输出tensor [row, col]，row目前均为1
 * @param inputLocal 输入tensor [row, col]
 * @param gammaLocal gamma参数tensor [row, col]
 * @param rmsNormParams rmsNrom计算所需系数，包括
          row 行数
          col 列数，对应headSizeCq或headSizeCkv
          reciprocal ，1/N
          epsilon，防止除零极小数
 */
template <typename InType, typename GammaType, typename C, typename OutType>
__aicore__ inline void RmsNormVF(const LocalTensor<OutType> &outputLocal, const LocalTensor<InType> &inputLocal,
                                 const LocalTensor<GammaType> &gammaLocal, const RmsNormParam rmsNormParams)
{
    uint32_t cnt = rmsNormParams.row * rmsNormParams.col;
    uint32_t repeatTimes = (cnt + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;

    __ubuf__ InType *inputBuf = (__ubuf__ InType *)inputLocal.GetPhyAddr();
    __ubuf__ GammaType *gammaBuf = (__ubuf__ GammaType *)gammaLocal.GetPhyAddr();
    __ubuf__ OutType *outputBuf = (__ubuf__ OutType *)outputLocal.GetPhyAddr();

    RmsNormVFImpl<InType, GammaType, C, OutType>(inputBuf, gammaBuf, outputBuf, cnt, repeatTimes, rmsNormParams);

    if (unlikely(rmsNormParams.isScaleEnable)) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(outputLocal, outputLocal, rmsNormParams.scale, cnt);
    }
}
} // namespace MlaProlog
#endif