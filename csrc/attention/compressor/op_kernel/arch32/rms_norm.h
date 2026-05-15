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

#include "../compressor_comm.h"
#include "compressor_vector_comm.h"

namespace Compressor {
/**
 * @brief RmsNorm 对矩阵进行rmsnorm
 * @param dstLocal 输出tensor [row, col]，支持和srcLocal是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param gammaLocal 系数gamma [1, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [(row * col + row) * sizeof(float)]
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数
          col 列数
 */
template <typename GammaType>
__aicore__ inline void RmsNorm(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                               const LocalTensor<GammaType> &gammaLocal, const LocalTensor<float> &shareTmpUb,
                               const RmsNormParam &rmsNormParams)
{
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> temp1Local = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> temp2Local = temp1Local[cnt];

    // temp1Local = srcLocal ^ 2
    Mul(temp1Local, srcLocal, srcLocal, cnt);
    PipeBarrier<PIPE_V>();
    
    MatRpeatParam repeatParams = {
        rmsNormParams.row,                                // row
        rmsNormParams.col,                                // col
        FP32_REPEAT_ELEMENT_NUM,                          // dtypeMask
        rmsNormParams.col / FP32_REPEAT_ELEMENT_NUM,                    // loopTimes
        rmsNormParams.col % FP32_REPEAT_ELEMENT_NUM,                    // colsRemain
        static_cast<uint8_t>(rmsNormParams.col / FP32_BLOCK_ELEMENT_NUM),       // repeatStride
    };

    // temp2Local[row] = Sum(temp1Local)
    RowSum(temp2Local, temp1Local, temp1Local, repeatParams);
    PipeBarrier<PIPE_V>();


    // temp2Local[row] = temp2Local[row] * reciprocal(1/N)
    Muls(temp2Local, temp2Local, rmsNormParams.reciprocal, rmsNormParams.row);
    PipeBarrier<PIPE_V>();

    // temp2Local[row] = temp2Local[row] + epsilon
    Adds(temp2Local, temp2Local, rmsNormParams.epsilon, rmsNormParams.row);
    PipeBarrier<PIPE_V>();

    // temp2Local[row] = Sqrt(temp2Local[row])
    Sqrt(temp2Local, temp2Local, rmsNormParams.row);
    PipeBarrier<PIPE_V>();

    // temp1Local[row, 8] = brc(temp2Local[row, 1])
    Brcb(temp1Local, temp2Local, CeilDivT(rmsNormParams.row, BRCB_NUM), {1, 8});   
    PipeBarrier<PIPE_V>();

    // dstLocal = srcLocal / temp1Local(sum)
    RowDivs(dstLocal, srcLocal, temp1Local, repeatParams);
    PipeBarrier<PIPE_V>();

    // dstLocal = dstLocal * gammaLocal
    MatMulVec(dstLocal, dstLocal, gammaLocal, repeatParams);
}
} // namespace Compressor
#endif // MLA_PROLOG_RMS_NORM_H