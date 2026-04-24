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
 * \file rope.h
 * \brief
 */

#ifndef ROPE_H
#define ROPE_H

#include "../compressor_comm.h"
#include "compressor_vector_comm.h"

namespace Compressor {

/**
 * @brief SetGatherSrcOffset 计算用于interleave模式的offset
 * @param gatherOffsetLocal 输出tensor [count]，数据类型需要为int64_t，使用时要转换
 * @param count offset的元素个数，一般为列数
 */
template <typename T>
__aicore__ inline void SetGatherSrcOffset(const LocalTensor<int32_t> &gatherOffsetLocal, uint32_t count)
{
    for (uint32_t i = 0; i < 8; i++) {
        gatherOffsetLocal.SetValue(i, i ^ 1);
    }
    
    event_t eventId_S_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId_S_V);
    WaitFlag<HardEvent::S_V>(eventId_S_V);

    int32_t scalarValue = 8;
    while (scalarValue < count) {
        int32_t nextValue = scalarValue * 2;
        PipeBarrier<PIPE_V>();
        if (nextValue < count) {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, scalarValue);
        } else {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    PipeBarrier<PIPE_V>();
    Muls(gatherOffsetLocal, gatherOffsetLocal, static_cast<int32_t>(sizeof(T)), count);
}


/**
 * @brief RotaryPosEmb 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param dstLocal 输出tensor [row, actualCol]，支持和srcLocal是同一块空间
 * @param srcLocal 输入tensor [row, actualCol]
 * @param cosLocal cos系数tensor [row, col]
 * @param sinLocal sin系数tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * sizeof(float)]
 * @param gatherOffsetcastLocal 用于interleave模式的offset，数据类型需要为uint64_t
 * @param row 待处理的行数
 * @param col 待处理的列数
 * @param actualCol 实际列数
 * @param baseAddr 计算基地址
 */
template <ROTARY_MODE MODE>
__aicore__ inline void RotaryPosEmb(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                    const LocalTensor<float> &cosLocal, const LocalTensor<float> &sinLocal,
                                    const LocalTensor<float> &shareTmpUb,
                                    const LocalTensor<uint32_t> &gatherOffsetcastLocal, uint32_t row, uint32_t col,
                                    uint32_t actualCol, uint64_t baseAddr)
{
    uint64_t cnt = row * col;
    uint32_t half_col = col >> 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<float> reArrLocal = shareTmpUb.ReinterpretCast<float>();
    if constexpr (MODE == ROTARY_MODE::HALF) {
        DataCopy(reArrLocal, srcLocal[baseAddr + half_col],
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(half_col, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(actualCol - half_col, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(half_col, FP32_BLOCK_ELEMENT_NUM))});
        DataCopy(reArrLocal[half_col], srcLocal[baseAddr],
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(half_col, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(actualCol - half_col, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(half_col, FP32_BLOCK_ELEMENT_NUM))});
        PipeBarrier<PIPE_V>();
        Muls(reArrLocal, reArrLocal, float(-1), half_col, row,
             {1, 1, static_cast<uint8_t>(CeilDivT(static_cast<uint32_t>(col), FP32_BLOCK_ELEMENT_NUM)),
              static_cast<uint8_t>(CeilDivT(static_cast<uint32_t>(col), FP32_BLOCK_ELEMENT_NUM))});
    } else if constexpr (MODE == ROTARY_MODE::INTERLEAVE) {
        for (uint32_t i = 0; i < row; i++) {
            Gather(reArrLocal[i * col], srcLocal[i * actualCol + baseAddr], gatherOffsetcastLocal, 0, col);
        }
        PipeBarrier<PIPE_V>();
        uint32_t repeatTimes = cnt / FP32_REPEAT_ELEMENT_NUM;
        uint32_t remainer = cnt % FP32_REPEAT_ELEMENT_NUM;
        uint64_t fullMask = 0x5555555555555555;
        uint64_t partialMask = 0x55;
        SetVectorMask<float, MaskMode::NORMAL>(0, fullMask);
        Muls<float, false>(reArrLocal, reArrLocal, float(-1), MASK_PLACEHOLDER, repeatTimes,
                           {1, 1, FP32_BLOCK_ELEMENT_NUM, FP32_BLOCK_ELEMENT_NUM});

        if (unlikely(remainer > 0)) {
            SetVectorMask<float, MaskMode::NORMAL>(0, partialMask);
            Muls<float, false>(reArrLocal[repeatTimes * FP32_REPEAT_ELEMENT_NUM],
                               reArrLocal[repeatTimes * FP32_REPEAT_ELEMENT_NUM], float(-1), MASK_PLACEHOLDER,
                               remainer / FP32_BLOCK_ELEMENT_NUM, {1, 1, 1, 1});
        }
        ResetMask();
    }
   
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams computeParams{1,
                                     1,
                                     1,
                                     static_cast<uint8_t>(CeilDivT(actualCol, FP32_BLOCK_ELEMENT_NUM)),
                                     static_cast<uint8_t>(CeilDivT(actualCol, FP32_BLOCK_ELEMENT_NUM)),
                                     static_cast<uint8_t>(CeilDivT(col, FP32_BLOCK_ELEMENT_NUM))};
    Mul(dstLocal[baseAddr], srcLocal[baseAddr], cosLocal, col, row, computeParams);
    Mul(reArrLocal, reArrLocal, sinLocal, cnt);
    PipeBarrier<PIPE_V>();
    Add(dstLocal[baseAddr], dstLocal[baseAddr], reArrLocal, col, row, computeParams);
}
}

#endif
