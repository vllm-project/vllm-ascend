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
 * \file service_rope.h
 * \brief
 */

#ifndef SERVICE_ROPE_H
#define SERVICE_ROPE_H

namespace MlaProlog {
/**
 * @brief RotaryPosEmb, 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param outputLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param cosLocal cos系数tensor [(row - 1) * sinCosRepStride + col]
 * @param sinLocal sin系数tensor [(row - 1) * sinCosRepStride + col] - 1 应已在sin中
 * @param shareTmpUb 临时buffer 内部需要的空间为 [2 * row * col * sizeof(C)]
 * @param row 待处理的行数
 * @param col 待处理的列数  col <= 512 / sizeof(C)
 * @param sinCosRepStride 行与行之间sin/cos系数的偏移，单位为元素个数。
 */
template <typename C>
__aicore__ inline void RotaryPosEmb(const LocalTensor<C> &outputLocal, const LocalTensor<C> &inputLocal,
                                    const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal,
                                    const LocalTensor<uint8_t> &shareTmpUb, uint64_t row, uint64_t col,
                                    uint8_t sinCosRepStride)
{
    uint64_t cnt = row * col;
    uint64_t rsvdCnt = 0;
    LocalTensor<C> reArrLocal = shareTmpUb.ReinterpretCast<C>();
    LocalTensor<C> outputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[cnt];
    GatherMaskParams gatherMaskParams = {
        1, // repeatTimes
        1, // src0BlockStride
        0, // src0RepeatStride
        0  // src1RepeatStride
    };
    // 取奇数索引元素
    GatherMask(reArrLocal, inputLocal, 1, true, col * row, gatherMaskParams, rsvdCnt);
    // 取偶数索引元素
    GatherMask(reArrLocal[cnt >> 1], inputLocal, 2, true, col * row, gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;
    uint8_t blockNumSinCosRepStride = sinCosRepStride / (ALIGN_BLOCK_SIZE / sizeof(C));
    BinaryRepeatParams mulParams = {
        1,                      // dstBlkStrideIn
        1,                      // src0BlkStrideIn
        1,                      // src1BlkStrideIn
        blockNumPerRow,         // dstRepStrideIn
        blockNumPerRowHalf,     // src0RepStrideIn
        blockNumSinCosRepStride // src1RepStrideIn
    };
    Mul(outputLocal, reArrLocal, cosLocal, col >> 1, row, mulParams);
    Mul(outputLocal[col >> 1], reArrLocal[cnt >> 1], cosLocal[col >> 1], col >> 1, row, mulParams);
    Mul(outputLocalSinTmp, reArrLocal[cnt >> 1], sinLocal, col >> 1, row, mulParams);
    Mul(outputLocalSinTmp[col >> 1], reArrLocal, sinLocal[col >> 1], col >> 1, row, mulParams);
    AscendC::PipeBarrier<PIPE_V>();
    Add(outputLocal, outputLocal, outputLocalSinTmp, cnt);
}
} // namespace MlaProlog
#endif