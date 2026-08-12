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
 * \file mat_mul_nd2nz_util.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_ND2NZ_UTIL__
#define __OP_KERNEL_MATMUL_V3_ND2NZ_UTIL__

#include "mat_mul_v3_common.h"

using namespace AscendC;
using namespace matmul;

const uint32_t BLOCK_COUNT_MAX = 4095;

constexpr uint32_t VNCHW_SIZE = 16;
constexpr uint64_t BLOCK_SIZE_BYTE = 32;

constexpr uint64_t REPEAT_TIMES_MAX = 255;
constexpr uint64_t SINGLE_COPY_SIZE = 256;


constexpr uint32_t M_BLOCK_NUM_ELE_LIST[16] = {1, 16, 8, 16, 4, 16, 8, 16, 2, 16, 8, 16, 4, 16, 8, 16};
constexpr uint32_t GCD_LIST[16] = {16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
const TransDataTo5HDParams PARA_ONE(false, false, 1, 0, 0);

enum class ND2NZ_DB_TYPE : int32_t {
    IN_OUTPUT,
    OUTPUT,
    NO_DB_REUSE_OUTPUT
};

enum class Nd2NzMode : int32_t {
    MULTI_CORE = 0,
    SINGLE_CORE = 1
};

template <class T>
__aicore__ inline void Copy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, uint32_t count)
{
    constexpr uint32_t copyLen = SINGLE_COPY_SIZE / sizeof(T);
    const CopyRepeatParams para(1, 1, 8, 8); // vnchw parameters, which is used to decide stride, burstlength of Copy
    uint32_t repeatTimes = count / copyLen;
    uint32_t tail = count % copyLen;
    uint32_t offset = repeatTimes * copyLen;
    Copy(dstLocal, srcLocal, copyLen, repeatTimes, para);
    if (tail != 0) {
        Copy(dstLocal[offset], srcLocal[offset], tail, 1, para);
    }
}

__aicore__ inline int32_t Align2(uint32_t x, uint32_t divisor)
{
    uint32_t remainder = x & (divisor - 1); // 计算m与divisor的模数
    if (remainder == 0) {
        return x; // 如果m已经能被2^n整除，直接返回m
    }
    return (x + divisor - remainder); // 否则找到
}

template <class T>
__aicore__ inline void PadDMain(uint64_t progress, LocalTensor<T> &dstLocal, LocalTensor<T> &srcLocal,
                                LocalTensor<T> midBuf, LocalTensor<T> zeroBuf, int eventIn, int eventOut,
                                uint32_t totalWidth, uint64_t c0Size, uint32_t hBlockNum, uint32_t copyInRepeat,
                                uint32_t hBuffer, uint32_t wTail, bool hasFlag)
{
    uint32_t widthBlock = totalWidth / c0Size;
    uint64_t dstLocalList[VNCHW_SIZE];
    uint64_t srcLocalList[VNCHW_SIZE];

    for (uint32_t i = 0; i < VNCHW_SIZE; i++) {
        srcLocalList[i] = (i * hBlockNum * totalWidth * sizeof(T) + srcLocal.GetPhyAddr());
        dstLocalList[i] = (i * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr());
    }

    if (copyInRepeat == 1) {
        TransDataTo5HD<T>(dstLocalList, srcLocalList, PARA_ONE);
    } else {
        TransDataTo5HDParams para(false, false, copyInRepeat, VNCHW_SIZE, 1);
        TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
    }

    if (hasFlag) {
        SetFlag<HardEvent::V_MTE2>(eventIn);
        WaitFlag<HardEvent::MTE3_V>(eventOut);
    }

    for (uint32_t i = 0; i < widthBlock; i++) {
        if constexpr (sizeof(T) == sizeof(half)) {
            for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = (j * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr() + i * VNCHW_SIZE * BLOCK_SIZE_BYTE);
                dstLocalList[j] =
                    (j * BLOCK_SIZE_BYTE * hBlockNum + dstLocal.GetPhyAddr() + hBuffer * i * BLOCK_SIZE_BYTE);
            }
        }
        if constexpr (sizeof(T) == sizeof(float)) {
            for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
                // for fp32, one block copy twice, which needs copy 8 * 16 and jump 8 * 16 when moving.
                srcLocalList[j] =
                    ((2 * (j % 8) + j / 8) * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr() + i * VNCHW_SIZE * BLOCK_SIZE_BYTE);
                dstLocalList[j] = ((j / 2 + 8 * (j % 2)) * BLOCK_SIZE_BYTE * hBlockNum + dstLocal.GetPhyAddr() +
                                   hBuffer * i * BLOCK_SIZE_BYTE);
            }
        }
        if (hBlockNum == 1) {
            TransDataTo5HD<T>(dstLocalList, srcLocalList, PARA_ONE);
        } else {
            uint64_t srcRepStride = totalWidth * sizeof(T) / 2;
            TransDataTo5HDParams para(false, false, hBlockNum, 1, srcRepStride);
            TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
        }
    }

    constexpr uint32_t copyLen = SINGLE_COPY_SIZE / sizeof(T);

    if (totalWidth > c0Size && wTail != 0) {
        uint16_t dstBlockStride = sizeof(T) * totalWidth / 2;

        if (8 * dstBlockStride > UINT8_MAX) {
            for (int i = 0; i < hBlockNum / 8; i++) {
                Duplicate(midBuf[8 * dstBlockStride * c0Size * i], T(0), copyLen, 1, dstBlockStride,
                          8 * dstBlockStride);
            }
        } else {
            Duplicate(midBuf, T(0), copyLen, hBlockNum / 8, dstBlockStride, 8 * dstBlockStride);
        }

        Duplicate(midBuf[c0Size * dstBlockStride * (hBlockNum / 8) * 8], T(0), (copyLen / 8) * (hBlockNum % 8), 1,
                  dstBlockStride, 0);
    }

    if constexpr (sizeof(T) == sizeof(half)) {
        for (uint32_t j = 0; j < wTail; j++) {
            srcLocalList[j] = (j * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
        }
        if (totalWidth > c0Size) {
            for (uint32_t j = wTail; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = midBuf.GetPhyAddr();
            }
        } else {
            for (uint32_t j = wTail; j < VNCHW_SIZE; j++) {
                srcLocalList[j] = zeroBuf.GetPhyAddr();
            }
        }
        for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
            dstLocalList[j] =
                (j * BLOCK_SIZE_BYTE * hBlockNum + dstLocal.GetPhyAddr() + hBuffer * widthBlock * BLOCK_SIZE_BYTE);
        }
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        for (uint32_t j = 0; j < wTail; j++) {
            srcLocalList[j] =
                ((2 * j) * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
            srcLocalList[j + 8] =
                ((2 * j + 1) * BLOCK_SIZE_BYTE + midBuf.GetPhyAddr() + widthBlock * VNCHW_SIZE * BLOCK_SIZE_BYTE);
        }
        if (totalWidth > c0Size) {
            for (uint32_t j = wTail; j < 8; j++) {
                srcLocalList[j] = midBuf.GetPhyAddr();
                srcLocalList[j + 8] = midBuf.GetPhyAddr();
            }
        } else {
            for (uint32_t j = wTail; j < 8; j++) {
                srcLocalList[j] = zeroBuf.GetPhyAddr();
                srcLocalList[j + 8] = zeroBuf.GetPhyAddr();
            }
        }

        for (uint32_t j = 0; j < VNCHW_SIZE; j++) {
            dstLocalList[j] = ((j / 2 + 8 * (j % 2)) * BLOCK_SIZE_BYTE * hBlockNum + dstLocal.GetPhyAddr() +
                               hBuffer * widthBlock * BLOCK_SIZE_BYTE);
        }
    }
    uint16_t dstBlockStride = sizeof(T) * totalWidth / 2;
    TransDataTo5HDParams para(false, false, hBlockNum, 1, dstBlockStride);
    TransDataTo5HD<T>(dstLocalList, srcLocalList, para);
}

template <class T>
__aicore__ inline void PadDAligned(uint64_t progress, LocalTensor<T> &dstLocal, LocalTensor<T> &srcLocal, int eventIn,
                                   int eventOut, uint32_t totalWidth, uint64_t c0Size, uint32_t hBlockNum, bool hasFlag)
{
    if (hasFlag) {
        WaitFlag<HardEvent::MTE3_V>(eventOut);
    }
    uint32_t repeatTimes = hBlockNum * 2;
    int nLoop = repeatTimes / REPEAT_TIMES_MAX;
    int loopTail = repeatTimes % REPEAT_TIMES_MAX;
    for (int i = 0; i < totalWidth / c0Size; i++) {
        for (int j = 0; j < nLoop; j++) {
            Copy(dstLocal[hBlockNum * c0Size * ALIGNED_H * i + REPEAT_TIMES_MAX * j * 8 * c0Size],
                 srcLocal[c0Size * i + REPEAT_TIMES_MAX * j * 8 * totalWidth], SINGLE_COPY_SIZE / sizeof(T),
                 REPEAT_TIMES_MAX, {1, uint16_t(totalWidth / c0Size), 8, uint16_t(8 * totalWidth / c0Size)});
        }
        Copy(dstLocal[hBlockNum * c0Size * ALIGNED_H * i + REPEAT_TIMES_MAX * nLoop * 8 * c0Size],
             srcLocal[c0Size * i + REPEAT_TIMES_MAX * nLoop * 8 * totalWidth], SINGLE_COPY_SIZE / sizeof(T), loopTail,
             {1, uint16_t(totalWidth / c0Size), 8, uint16_t(8 * totalWidth / c0Size)});
    }
    if (hasFlag) {
        SetFlag<HardEvent::V_MTE2>(eventIn);
    }
}

#endif
