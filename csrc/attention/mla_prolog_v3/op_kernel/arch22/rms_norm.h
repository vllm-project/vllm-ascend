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
 * \file rms_norm.h
 * \brief
 */

#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "mla_prolog_comm.h"
namespace MlaProlog {
/**
 * @brief RmsNorm 对一行进行rmsnorm
 * @param outLocal 输出tensor [1 * cnt]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [1 * cnt]
 * @param gammaLocal 系数gamma [1 * cnt]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [cnt * sizeof(float) + ALIGN_BLOCK_SIZE]
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 */
template <typename GammaType>
__aicore__ inline void RmsNorm(const LocalTensor<float> &outLocal, const LocalTensor<float> &inputLocal,
                               const LocalTensor<GammaType> &gammaLocal, const LocalTensor<uint8_t> &shareTmpUb,
                               const RmsNormParam &rmsNormParams)
{
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> xSquareLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> xSumLocal = xSquareLocal[cnt];
    Mul(xSquareLocal, inputLocal, inputLocal, cnt);
    AscendC::PipeBarrier<PIPE_V>();

    // calcNum >> 6 : calcNum / 64(FP32_REPEAT_ELEMENT_NUM)
    uint64_t repeatTimesAdd = static_cast<uint64_t>(cnt) >> 6;
    // dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride
    BinaryRepeatParams addParams = {1, 1, 1, 0, 8, 0};
    Add(xSquareLocal, xSquareLocal[FP32_REPEAT_ELEMENT_NUM], xSquareLocal, FP32_REPEAT_ELEMENT_NUM, repeatTimesAdd - 1,
        addParams);
    AscendC::PipeBarrier<PIPE_V>();
    WholeReduceSum(xSumLocal, xSquareLocal, FP32_REPEAT_ELEMENT_NUM, 1, 8, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSum = xSum * reciprocal
    Muls<float>(xSumLocal, xSumLocal, rmsNormParams.reciprocal, 1);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSum = xSum + epsilon
    Adds<float>(xSumLocal, xSumLocal, rmsNormParams.epsilon, 1);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSum = sqrt(xSum)
    Sqrt(xSumLocal, xSumLocal, 1);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSquare[1, 8] = brc(xSum[1,1])
    // dstBlkStride, dstRepStride
    BrcbRepeatParams repeatParams = {1, 1};
    Brcb(xSquareLocal, xSumLocal, 1, repeatParams);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: inputLocal = inputLocal / xSquareLocal
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    // dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride
    BinaryRepeatParams divParams = {1, 1, 0, 8, 8, 0};
    Div(inputLocal, inputLocal, xSquareLocal, mask, cnt / 64, divParams);

    AscendC::PipeBarrier<PIPE_V>();

    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, cnt);

    AscendC::PipeBarrier<PIPE_V>();

    Mul(outLocal, inputLocal, xSquareLocal, cnt);

    if (unlikely(rmsNormParams.isScaleEnable)) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(outLocal, outLocal, rmsNormParams.scale, cnt);
    }
}
} // namespace MlaProlog
#endif // MLA_PROLOG_RMS_NORM_H
