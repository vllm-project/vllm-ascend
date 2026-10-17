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
 * \file soft_max.h
 * \brief
 */

#ifndef SOFT_MAX_H
#define SOFT_MAX_H

#include "compressor_comm.h"
#include "compressor_vector_comm.h"

namespace Compressor {
/**
 * @brief ColumnSoftMax 对矩阵按列进行SoftMax
 * @param dstLocal 输出tensor [row, col]，支持和srcLocal是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [floor(row / 2) * col * sizeof(float)]
 * @param row 行数
 * @param col 列数
 */
__aicore__ inline void ColumnSoftMax(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                     const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = col / dtypeMask;
    uint32_t dRemain = col % dtypeMask;
    uint8_t repeatStride = col / FP32_BLOCK_ELEMENT_NUM;
    ColumnMax(shareTmpUb, srcLocal, shareTmpUb, row, col);
    PipeBarrier<PIPE_V>();
    MatSubVec(dstLocal, srcLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
    PipeBarrier<PIPE_V>();
    Exp(dstLocal, dstLocal, row * col);
    PipeBarrier<PIPE_V>();
    ColumnSum(shareTmpUb, dstLocal, shareTmpUb, row, col);
    PipeBarrier<PIPE_V>();
    MatDivVec(dstLocal, dstLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
}

} // namespace Compressor

#endif
