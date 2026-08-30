/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vector_common.h
 * \brief
 */
#ifndef VECTOR_COMMON_H
#define VECTOR_COMMON_H

#include "kernel_operator.h"

using namespace AiInfraInferenceAttentionCommon;
using namespace AscendC;
using AscendC::LocalTensor;

namespace AiInfraInferenceCommonFaBaseVector {

// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;
constexpr int64_t HALF_NUM = 2;
constexpr int64_t STRIDE_LENGTH = 8;
constexpr int64_t MAX_VALID_LENGTH = 1024;
enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

__aicore__ inline bool IsExistInvalidRows(
    int64_t nextTokensPerBatch, int64_t preTokensPerBatch, uint32_t mode, bool attenMaskFlag, bool isRowInvalid)
{
    if (mode == RIGHT_DOWN_CAUSAL) { // sparse = 3
        return (nextTokensPerBatch < 0);
    } else if (mode == BAND) { // sparse = 4
        return (nextTokensPerBatch < 0 || preTokensPerBatch < 0);
    } else if (mode == DEFAULT_MASK || mode == ALL_MASK) { // sparse = 0 || sparse = 1
        if (nextTokensPerBatch < 0 || preTokensPerBatch < 0) {
            return true;
        } else {
            return attenMaskFlag && isRowInvalid;
        }
    }
    return false;
}

__aicore__ inline void GetSafeActToken(int64_t actSeqLensQ,
                                       int64_t actSeqLensKv,
                                       int64_t &safePreToken,
                                       int64_t &safeNextToken,
                                       uint32_t mode)
{
    if (mode == DEFAULT_MASK) {
        safePreToken = Max(-actSeqLensKv, safePreToken);
        safePreToken = Min(safePreToken, actSeqLensQ);
        safeNextToken = Max(-actSeqLensQ, safeNextToken);
        safeNextToken = Min(safeNextToken, actSeqLensKv);
    } else if (mode == BAND) {
        safePreToken = Max(-actSeqLensQ, safePreToken);
        safePreToken = Min(safePreToken, actSeqLensKv);
        safeNextToken = Max(-actSeqLensKv, safeNextToken);
        safeNextToken = Min(safeNextToken, actSeqLensQ);
    }
}

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb,
                                 LocalTensor<float> src0Ub,
                                 LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount,
                                 uint32_t columnCount,
                                 uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void MatDivVec(LocalTensor<float> dstUb,
                                 LocalTensor<float> src0Ub,
                                 LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount,
                                 uint32_t columnCount,
                                 uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src1Ub[i, j] / src0Ub[j],
    // src0Ub:[dealRowCount, actualColumnCount] src1Ub:[1, columnCount] dstUb:[dealRowCount, columnCount]
    // restraint: dealRowCount < 256
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src1RepStride = 0;
        repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void VecMulBlkMat(LocalTensor<float> dstUb,
                                    LocalTensor<float> src0Ub,
                                    LocalTensor<float> src1Ub,
                                    uint32_t dealRowCount,
                                    uint32_t columnCount,
                                    uint32_t actualColumnCount)
{
    // vec mul by row
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, 8] dstUb:[dealRowCount, columnCount]
    BinaryRepeatParams repeatParams;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        // [1, columnCount] * [dealRowCount, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < loopCount; i++) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, remainCount, dealRowCount, repeatParams);
        }
    } else {
        // [1, columnCount] * [1, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = STRIDE_LENGTH;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = STRIDE_LENGTH;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub, src1Ub[i * FP32_BLOCK_ELEMENT_NUM], mask, loopCount, repeatParams);
            if (remainCount > 0) {
                Mul(dstUb[i * columnCount + loopCount * mask],
                    src0Ub[loopCount * mask],
                    src1Ub[i * FP32_BLOCK_ELEMENT_NUM],
                    remainCount,
                    1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void VecAddMat(LocalTensor<float> dstUb,
                                 LocalTensor<float> src0Ub,
                                 LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount,
                                 uint32_t columnCount,
                                 uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;

        uint64_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Add(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

template <typename T>
__aicore__ inline void RowDivs(LocalTensor<T> dstUb,
                               LocalTensor<T> src0Ub,
                               LocalTensor<T> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t repeatNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockNum = FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        repeatNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint32_t dLoop = actualColumnCount / repeatNum;
    uint32_t dRemain = actualColumnCount % repeatNum;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / blockNum;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / blockNum;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, repeatNum, dealRowCount, repeatParamsDiv);
            offset += repeatNum;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * blockNum], repeatNum, columnRepeatCount, columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * repeatNum], src0Ub[dLoop * repeatNum], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb,
                               LocalTensor<T> src0Ub,
                               LocalTensor<T> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2; // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount],
                    src0Ub[i * columnCount],
                    src1Ub[i * blockElementNum],
                    repeatElementNum,
                    dLoop,
                    columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum],
                src0Ub[dLoop * repeatElementNum],
                src1Ub,
                dRemain,
                dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount],
                src0Ub[i * columnCount],
                src1Ub[i * blockElementNum],
                repeatElementNum,
                dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum],
                    dRemain,
                    1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void RowSum(LocalTensor<float> &dstUb,
                              LocalTensor<float> srcUb,
                              uint32_t dealRowCount,
                              uint32_t columnCount,
                              uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Add(srcUb[j * dtypeMask],
                srcUb[j * dtypeMask],
                srcUb[(j + blockCount) * dtypeMask],
                dtypeMask,
                dealRowCount,
                repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstUb,
                   srcUb,
                   (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask,
                   dealRowCount,
                   1,
                   1,
                   columnCount / (BYTE_BLOCK / sizeof(float)));
}

__aicore__ inline uint32_t GetMinPowerTwo(uint32_t cap)
{
    uint32_t i = 1;
    while (i < cap) {
        i = i << 1;
    }
    return i;
}

__aicore__ inline void RowSumForLongColumnCount(LocalTensor<float> &dstUb,
                                                LocalTensor<float> srcUb,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    // columnCount要求32元素对齐
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Add(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > MAX_VALID_LENGTH) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Add(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            AscendC::PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            AscendC::PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowSum(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

__aicore__ inline void RowMax(LocalTensor<float> &dstUb,
                              LocalTensor<float> &srcUb,
                              uint32_t dealRowCount,
                              uint32_t columnCount,
                              uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask],
                srcUb[j * dtypeMask],
                srcUb[(j + blockCount) * dtypeMask],
                dtypeMask,
                dealRowCount,
                repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb,
                   srcUb,
                   (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask,
                   dealRowCount,
                   1,
                   1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM,
                   ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void RowMaxForLongColumnCount(LocalTensor<float> &dstUb,
                                                LocalTensor<float> srcUb,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Max(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > MAX_VALID_LENGTH) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Max(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            AscendC::PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            AscendC::PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowMax(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

__aicore__ inline void MatDivsVec(LocalTensor<float> dstUb,
                                  LocalTensor<float> src0Ub,
                                  LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount,
                                  uint32_t columnCount,
                                  uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask],
            src0Ub[dLoop * dtypeMask],
            src1Ub[dLoop * dtypeMask],
            dRemain,
            dealRowCount,
            repeatParamsDiv);
    }
}

__aicore__ inline void RowSub(LocalTensor<float> dstUb,
                              LocalTensor<float> src0Ub,
                              LocalTensor<float> src1Ub,
                              uint32_t dealRowCount,
                              uint32_t columnCount,
                              uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask],
            src0Ub[dLoop * dtypeMask],
            src1Ub[dLoop * dtypeMask],
            dRemain,
            dealRowCount,
            repeatParamsSub);
    }
}

__aicore__ inline void ColMax(LocalTensor<float> dstUb,
                              LocalTensor<float> src0Ub,
                              LocalTensor<float> src1Ub,
                              uint32_t dealRowCount,
                              uint32_t columnCount,
                              uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask],
            src0Ub[dLoop * dtypeMask],
            src1Ub[dLoop * dtypeMask],
            dRemain,
            dealRowCount,
            repeatParamsMax);
    }
}

__aicore__ inline void ColAdd(LocalTensor<float> dstUb,
                              LocalTensor<float> src0Ub,
                              LocalTensor<float> src1Ub,
                              uint32_t dealRowCount,
                              uint32_t columnCount,
                              uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask],
            src0Ub[dLoop * dtypeMask],
            src1Ub[dLoop * dtypeMask],
            dRemain,
            dealRowCount,
            repeatParamsAdd);
    }
}

template <typename T>
__aicore__ inline void ComputeSoftMaxLse(LocalTensor<T> &softmaxlseUb,
                                         LocalTensor<T> &softmaxSumUb,
                                         LocalTensor<T> &softmaxMaxUb,
                                         uint32_t dealRowCount)
{
    uint32_t blockNum = AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    if (std::is_same<T, half>::value) {
        blockNum = AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint64_t dealRowCountAlign = dealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    Log(softmaxlseUb, softmaxSumUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxlseUb, softmaxlseUb, softmaxMaxUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
}
static constexpr uint64_t headDim = 512ULL;
template <FIA_LAYOUT LAYOUT_T, typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDMTiling(LocalTensor<OUT_T> &attenOutUb,
                                                  const FusedTransposeInfo &transInfo,
                                                  const AiInfraInferenceAttentionCommon::ConstInfo &constInfo,
                                                  GlobalTensor<uint64_t> &actualSeqLengthsGmQ,
                                                  GlobalTensor<OUT_T> &attentionOutGm)
{
    uint32_t tSize = constInfo.batchSize * constInfo.qSeqSize;
    uint32_t tBase = transInfo.bIdx * constInfo.qSeqSize;
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
        tBase = transInfo.bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(transInfo.bIdx - 1);
    }

    uint32_t s1Idx = transInfo.s1StartIdx;
    uint32_t attenOutUbOffset = 0;
    for (int i = 0; i < transInfo.s1Count; i++) {
        uint32_t gIdx = 0; // 中间块
        uint32_t gCountOneS1 = constInfo.gSize;
        if (i == 0) { // 首块
            gIdx = transInfo.gStartIdx;
            gCountOneS1 = (constInfo.gSize - transInfo.gStartIdx) < transInfo.gCount
                              ? (constInfo.gSize - transInfo.gStartIdx)
                              : transInfo.gCount; // min(info.gSize - gStartIdx, gCount);
        } else if (i == transInfo.s1Count - 1) { // 尾块
            gIdx = 0;
            gCountOneS1 = transInfo.gEndIdx + 1;
        }
        uint64_t attenOutOffset = transInfo.n2Idx * constInfo.gSize * tSize * headDim + // N2轴的偏移
                                  gIdx * tSize * headDim + // G轴的偏移
                                  tBase * headDim + // B轴的偏移
                                  s1Idx * headDim; // S1轴的偏移
        bool dstStrideFlag = ((tSize - 1) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX ? 1 : 0;
        if (dstStrideFlag) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T); // 一个D的大小
            dataCopyParams.srcStride = 0; // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
            DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T) / 32U; // 一个D的大小
            dataCopyParams.srcStride = 0; // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U; // 跳写
            DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        }
        s1Idx++;
        attenOutUbOffset += gCountOneS1 * headDim;
    }
}

template <typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDGTiling(LocalTensor<OUT_T> &attenOutUb,
                                                  const FusedTransposeInfo &transInfo,
                                                  const AiInfraInferenceAttentionCommon::ConstInfo &constInfo,
                                                  GlobalTensor<OUT_T> &attentionOutGm)
{
    bool hasHeadBlock = transInfo.s1StartIdx != 0;
    bool hasTailBlock = (transInfo.s1EndIdx + 1) != constInfo.qSeqSize;
    uint32_t attenOutUbOffset = 0;
    if (hasHeadBlock) { // 头块单独一条DataCopy指令
        DataCopyParams dataCopyParamsHead;
        dataCopyParamsHead.blockCount = 1;
        dataCopyParamsHead.blockLen = (constInfo.qSeqSize - transInfo.s1StartIdx) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsHead.srcStride = 0;
        dataCopyParamsHead.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
            transInfo.gStartIdx * constInfo.batchSize * constInfo.qSeqSize * headDim + // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * headDim + // B轴的偏移
            transInfo.s1StartIdx * headDim; // S1轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParamsHead);
        attenOutUbOffset += (constInfo.qSeqSize - transInfo.s1StartIdx) * headDim;
    }
    // 中间块DataCopy指令
    uint64_t attenOutOffset =
        transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
        (transInfo.gStartIdx + static_cast<uint32_t>(hasHeadBlock)) * constInfo.batchSize * constInfo.qSeqSize *
            headDim + // G轴的偏移
        transInfo.bIdx * constInfo.qSeqSize * headDim; // B轴的偏移
    bool dstStrideFlag =
        ((constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX
            ? 1
            : 0;
    if (dstStrideFlag) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * headDim * sizeof(OUT_T); // 一个S1*D的大小
        dataCopyParams.srcStride = 0; // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T); // 跳写
        DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * headDim);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * headDim * sizeof(OUT_T) / 32U; // 一个S1*D的大小
        dataCopyParams.srcStride = 0; // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T) / 32U; // 跳写
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * headDim);
    }
    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = (transInfo.s1EndIdx + 1) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
            (transInfo.gStartIdx + transInfo.gCount - 1) * constInfo.batchSize * constInfo.qSeqSize *
                headDim + // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * headDim; // B轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
    }
}

enum LAYOUT_Q {
    GS,
    SG,
    S1_EQUAL1,
};

enum MaskDataType : uint8_t {
    MASK_BOOL,
    MASK_INT8,
    MASK_UINT8,
    MASK_FP16,
};

struct MaskInfo {
    uint32_t gs1StartIdx;
    uint32_t gs1dealNum;
    uint32_t s1Size;
    uint32_t gSize;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;

    int64_t preToken = 0;
    int64_t nextToken = 0;
    int64_t sinkNumber = 0; // AiInfraFusedInferAttentionSink自定义算子新增参数
    bool isEnableSinkNumber = false; // AiInfraFusedInferAttentionSink自定义算子新增参数

    // for bss & bs
    uint32_t batchIdx;
    uint32_t batchOffset;
    uint32_t attenMaskStride;

    LAYOUT_Q layout;
    MaskDataType attenMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;
};

__aicore__ inline uint64_t ComputeAttenMaskOffsetNoCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    uint64_t bOffset = static_cast<uint64_t>(info.batchIdx) * static_cast<uint64_t>(info.batchOffset);
    uint64_t s1Offset = s1StartIdx % info.s1Size * info.attenMaskStride;
    uint64_t s2Offset = info.s2StartIdx;
    return bOffset + s1Offset + s2Offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.sparseMode == BAND) { // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    } else if (info.isEnableSinkNumber &&
               info.sparseMode == LEFT_UP_CAUSAL) { // AiInfraFusedInferAttentionSink自定义算子新增部分
        nextToken = info.sinkNumber;
        nextToken = nextToken < info.s2Size ? nextToken : info.s2Size;
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskStride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskInfo &info, uint32_t s1StartIdx)
{
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算token
    int64_t delta = -preToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskStride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfo &info, uint32_t s1StartIdx = 0, bool isPre = false)
{
    if (isPre) {
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    } else {
        if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
            return ComputeAttenMaskOffsetNoCompress(info, s1StartIdx);
        } else {
            return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
        }
    }
}

// AiInfraFusedInferAttentionSink自定义算子新增函数：sparseMode3/4时，需要对sink部分的掩码进行赋值操作
template <typename T>
__aicore__ inline void AttentionmaskSinkNumberProcess(
    LocalTensor<T> &attenMaskUb, MaskInfo &info, uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
{
    uint32_t s2StartIdx = info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (info.isEnableSinkNumber &&
        (info.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND ||
         info.sparseMode == AiInfraInferenceCommonFaBaseVector::RIGHT_DOWN_CAUSAL) &&
        s2StartIdx < info.sinkNumber) {
        uint32_t s2EndIdx = s2StartIdx + info.s2dealNum;
        s2EndIdx = s2EndIdx <= info.sinkNumber ? s2EndIdx : info.sinkNumber;
        uint32_t processS2Len = (s2EndIdx - s2StartIdx);
        uint32_t s1SinkStartIdx = 0;
        uint32_t s1SinkEndIdx = s1EndIdx - s1StartIdx;
        uint32_t s1NextEnd = s1StartIdx;
        uint32_t s1PreStart = s1EndIdx;
        int64_t nextToken = 0; // sparse2 本身原点就是左上角
        if (info.sparseMode == AiInfraInferenceCommonFaBaseVector::RIGHT_DOWN_CAUSAL) {
            nextToken =
                static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
        } else if (info.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) { // 4
            nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> attenMaskUbDst = attenMaskUb.template ReinterpretCast<int16_t>();

        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
        if (!isPre and nextToken < info.sinkNumber) {
            s1NextEnd = info.sinkNumber - nextToken;
            if (s1StartIdx < s1NextEnd) {
                s1SinkStartIdx = s1NextEnd - s1StartIdx;
                s1SinkStartIdx = s1SinkStartIdx < s1EndIdx - s1StartIdx ? s1SinkStartIdx : s1EndIdx - s1StartIdx;
                for (uint32_t i = 0; i < s1SinkStartIdx; i++) {
                    uint32_t offset = i * attenMaskSizeAlign / 2;
                    Duplicate(attenMaskUbDst[offset], static_cast<int16_t>(257), processS2Len / 2);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
        if (isPre) {
            s1PreStart = info.preToken < 0 ? info.s1Size + info.preToken : info.s1Size;
            if (s1StartIdx >= s1PreStart) {
                return;
            }
            s1SinkEndIdx = s1PreStart < s1EndIdx ? s1PreStart - s1StartIdx : s1EndIdx - s1StartIdx;
            for (uint32_t i = s1SinkStartIdx; i < s1SinkEndIdx; i++) {
                uint32_t offset = i * attenMaskSizeAlign / 2;
                Duplicate(attenMaskUbDst[offset], static_cast<int16_t>(257), processS2Len / 2);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
    }
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T> &attenMaskUb,
                                             GlobalTensor<T> &srcGmAddr,
                                             MaskInfo &info,
                                             uint32_t s1StartIdx,
                                             uint32_t s1EndIdx,
                                             bool isPre = false)
{
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};

    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);

    AttentionmaskSinkNumberProcess(
        attenMaskUb,
        info,
        s1StartIdx,
        s1EndIdx,
        isPre); // AiInfraFusedInferAttentionSink自定义算子新增函数：sparseMode3/4时，需要对sink部分的掩码进行赋值操作
}

template <typename T, typename U>
__aicore__ inline void AttentionmaskCopyInForGsLayout(
    LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (info.gs1dealNum <= info.s1Size) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskSizeAlign];
            AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
    } else {
        AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, 0, info.s1Size, isPre);
        LocalTensor<T> attenMaskUbDst = tmpBuf.template ReinterpretCast<T>();
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

        uint32_t headS1Count = 0;
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            headS1Count = info.s1Size - s1StartIdx;
        } else {
            headS1Count = info.gs1dealNum;
        }

        // head
        DataCopy(attenMaskUbDst, attenMaskUb[s1StartIdx * attenMaskSizeAlign], headS1Count * attenMaskSizeAlign);
        // mid
        uint32_t reminRowCount = info.gs1dealNum - headS1Count;
        uint32_t midGCount = reminRowCount / info.s1Size;
        uint32_t tailS1Size = reminRowCount % info.s1Size;
        for (uint32_t i = 0; i < midGCount; i++) {
            DataCopy(attenMaskUbDst[(headS1Count + i * info.s1Size) * attenMaskSizeAlign],
                     attenMaskUb,
                     info.s1Size * attenMaskSizeAlign);
        }
        // tail
        if (tailS1Size > 0) {
            DataCopy(attenMaskUbDst[(headS1Count + midGCount * info.s1Size) * attenMaskSizeAlign],
                     attenMaskUb,
                     tailS1Size * attenMaskSizeAlign);
        }
        attenMaskUb = attenMaskUbDst;
    }
}

template <typename T, typename U>
__aicore__ inline void AttentionmaskCopyInForSgLayout(
    LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);

    AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx + 1, isPre);

    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(enQueEvtID);
    WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

    LocalTensor<int16_t> attenMaskUbDst = tmpBuf.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    uint32_t headGCount = s1Count > 1 ? (info.gSize - info.gs1StartIdx % info.gSize) : info.gs1dealNum;
    uint32_t dstMaskOffset = 0;
    uint32_t srcMaskBaseOffset = 0;
    // head
    SetMaskCount();
    SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset],
                         mask16[srcMaskBaseOffset],
                         AscendC::MASK_PLACEHOLDER,
                         headGCount,
                         {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    dstMaskOffset += headGCount * attenMaskSizeAlign / sizeof(int16_t);
    srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);

    // mid
    uint32_t reminRowCount = info.gs1dealNum - headGCount;
    uint32_t midS1Count = reminRowCount / info.gSize;
    uint32_t tailGSize = reminRowCount % info.gSize;

    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset],
                             mask16[srcMaskBaseOffset],
                             AscendC::MASK_PLACEHOLDER,
                             info.gSize,
                             {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
        dstMaskOffset += info.gSize * attenMaskSizeAlign / sizeof(int16_t);
        srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);
    }
    // tail
    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset],
                             mask16[srcMaskBaseOffset],
                             AscendC::MASK_PLACEHOLDER,
                             tailGSize,
                             {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo &info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t nextToken = 0; // sparse2 本身远点就在左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为远点计算Token
    } else if (info.sparseMode == BAND) { // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    } else if (info.sparseMode == LEFT_UP_CAUSAL &&
               info.sinkNumber > 0) { // AiInfraFusedInferAttentionSink自定义算子新增部分
        nextToken = info.sinkNumber;
    }

    if (static_cast<int64_t>(info.s2StartIdx) < static_cast<int64_t>(info.sinkNumber)) {
        return false;
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsSkipAttentionmaskForPre(MaskInfo &info)
{
    if (info.sparseMode != BAND) {
        return true;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t preToken = info.preToken + static_cast<uint64_t>(info.s1Size) -
                       static_cast<uint64_t>(info.s2Size); // 统一以左上角为原点计算Token
    int32_t s1EndIdx =
        info.layout == GS ? s1StartIdx + info.gs1dealNum : (info.gs1StartIdx + info.gs1dealNum) / info.gSize;

    if (static_cast<int64_t>(info.s2StartIdx) + preToken >= static_cast<int64_t>(s1EndIdx)) {
        return true;
    }
    return false;
}

template <typename T, typename U>
__aicore__ inline void AttentionmaskCopyIn(
    LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    if (info.layout == GS) {
        AttentionmaskCopyInForGsLayout(attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else if (info.layout == SG) { // sg
        AttentionmaskCopyInForSgLayout(attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else if (info.layout == S1_EQUAL1) {
        uint64_t maskOffset = ComputeAttenMaskOffset(info, 0, isPre);
        uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
        DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);

        // AiInfraFusedInferAttentionSink自定义算子新增：sparseMode3/4时，需要对sink部分的掩码进行赋值操作
        if (info.isEnableSinkNumber &&
            info.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND &&
            info.s2StartIdx < info.sinkNumber &&
            isPre) {
            uint32_t s2EndIdx = info.s2StartIdx + info.s2dealNum;
            s2EndIdx = s2EndIdx < info.sinkNumber ? s2EndIdx : info.sinkNumber;
            uint32_t processS2Len = (s2EndIdx - info.s2StartIdx);

            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<int16_t> attenMaskUbDst = attenMaskUb.template ReinterpretCast<int16_t>();

            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(enQueEvtID);
            WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
            uint32_t offset = attenMaskSizeAlign / 2;
            Duplicate(attenMaskUbDst, static_cast<int16_t>(257), processS2Len / 2);
            AscendC::PipeBarrier<PIPE_V>();
            attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
        }

        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
        for (uint32_t i = 1; i < info.gs1dealNum; i++) {
            uint32_t offset = i * attenMaskSizeAlign;
            DataCopy(attenMaskUb[offset], attenMaskUb, attenMaskSizeAlign);
        }
    }
}

template <typename T, typename M, typename U>
__aicore__ inline void AttentionmaskCompute(LocalTensor<T> &dstUb,
                                            LocalTensor<T> &srcUb,
                                            LocalTensor<M> &attenMaskUb,
                                            LocalTensor<U> &tmpBuf,
                                            MaskInfo &info,
                                            bool isPre = false)
{
    uint32_t dealRowCount = info.gs1dealNum;
    uint32_t columnCount = Align(info.s2dealNum, 32U);
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (info.attenMaskType != MASK_FP16) {
        // int8 & uint8 is ok
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        srcUb.SetSize(dealRowCount * columnCount); // Select接口要求src size与参数匹配
        if (isPre) {
            SelectWithBytesMask(
                dstUb, *((T *)&info.maskValue), srcUb, attenMaskUb, tmpBuf, selectWithBytesMaskShapeInfo);
        } else {
            SelectWithBytesMask(
                dstUb, srcUb, *((T *)&info.maskValue), attenMaskUb, tmpBuf, selectWithBytesMaskShapeInfo);
        }
        srcUb.SetSize(AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K /
                      sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
    }
}

enum class UbInputFormat {
    GS1 = 0,
    S1G = 1
};

struct InvalidRowParams {
    uint64_t actS1Size;
    uint64_t gSize;
    uint32_t gS1Idx;
    uint32_t dealRowCount;
    uint32_t columnCount;
    int64_t preTokensPerBatch;
    int64_t nextTokensPerBatch;
};

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD) || (LAYOUT_T == FIA_LAYOUT::BSND),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::BSND) {
        return UbInputFormat::S1G;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbInputFormat::GS1;
    }
}

template <typename T, UbInputFormat UB_INPUTFORMAT> class InvalidRows {
  public:
    __aicore__ inline void operator()(LocalTensor<T> &attenOutUb, InvalidRowParams &params)
    {
        if (params.preTokensPerBatch < 0) { // 下方存在行无效
            DealInvalidRowsBelow(attenOutUb, params);
        }

        if (params.nextTokensPerBatch < 0) { // 上方存在行无效
            AscendC::PipeBarrier<PIPE_V>();
            DealInvalidRowsAbove(attenOutUb, params);
        }
    }

  private:
    __aicore__ inline void DealInvalidRowsAbove(LocalTensor<T> &attenOutUb, InvalidRowParams &params);
    __aicore__ inline void DealInvalidRowsBelow(LocalTensor<T> &attenOutUb, InvalidRowParams &params);
};

template <typename T, UbInputFormat UB_INPUTFORMAT>
__aicore__ inline void InvalidRows<T, UB_INPUTFORMAT>::DealInvalidRowsBelow(LocalTensor<T> &attenOutUb,
                                                                            InvalidRowParams &params)
{
    if constexpr (UB_INPUTFORMAT == UbInputFormat::GS1) {
        int32_t s1BottomPos = params.actS1Size + params.preTokensPerBatch - 1;
        int32_t s1End = (params.gS1Idx + params.dealRowCount - 1) % params.actS1Size;

        for (int32_t s1RealEnd = params.dealRowCount - 1; s1RealEnd >= 0;) {
            if (s1End > s1BottomPos) {
                int32_t s1Num = s1End - s1BottomPos;
                if (s1RealEnd - s1Num < 0) {
                    s1Num = s1RealEnd + 1;
                }
                int32_t s1RealStart = s1RealEnd - s1Num + 1;
                Duplicate(attenOutUb[s1RealStart * params.columnCount],
                          static_cast<T>(AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO),
                          params.columnCount * s1Num);
                AscendC::PipeBarrier<PIPE_V>();
            }
            s1RealEnd -= s1End + 1;
            s1End = params.actS1Size - 1;
        }
    } else if constexpr (UB_INPUTFORMAT == UbInputFormat::S1G) {
        int32_t s1BottomTok = params.actS1Size + params.preTokensPerBatch;
        uint32_t s1 = params.gS1Idx / params.gSize;
        uint32_t gIdx = params.gS1Idx % params.gSize;
        uint32_t dealRowOffset = 0;
        if (s1 < s1BottomTok) {
            // 如果s1 < 行无效开始行，偏移s1到行无效开始行
            s1 = s1BottomTok;
            dealRowOffset = s1BottomTok * params.gSize - params.gS1Idx;
            gIdx = 0;
        }
        while (s1 >= s1BottomTok && s1 < params.actS1Size && dealRowOffset < params.dealRowCount) {
            uint32_t gNum = params.gSize - gIdx;
            if (dealRowOffset + gNum > params.dealRowCount) {
                gNum = params.dealRowCount - dealRowOffset;
            }
            Duplicate(attenOutUb[dealRowOffset * params.columnCount],
                      static_cast<T>(AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO),
                      params.columnCount * gNum);
            AscendC::PipeBarrier<PIPE_V>();
            dealRowOffset += gNum;
            s1++;
            gIdx = 0;
        }
    }
}

template <typename T, UbInputFormat UB_INPUTFORMAT>
__aicore__ inline void InvalidRows<T, UB_INPUTFORMAT>::DealInvalidRowsAbove(LocalTensor<T> &attenOutUb,
                                                                            InvalidRowParams &params)
{
    uint32_t s1Tok = -params.nextTokensPerBatch;
    if constexpr (UB_INPUTFORMAT == UbInputFormat::GS1) {
        uint32_t s1 = params.gS1Idx % params.actS1Size;
        for (uint32_t i = 0; i < params.dealRowCount;) {
            if (s1 < s1Tok) {
                uint32_t s1Num = s1Tok - s1;
                if (i + s1Num > params.dealRowCount) {
                    s1Num = params.dealRowCount - i;
                }
                Duplicate(attenOutUb[i * params.columnCount],
                          static_cast<T>(AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO),
                          params.columnCount * s1Num);
                AscendC::PipeBarrier<PIPE_V>();
            }
            i += params.actS1Size - s1;
            s1 = 0;
        }
    } else if constexpr (UB_INPUTFORMAT == UbInputFormat::S1G) {
        uint32_t s1 = params.gS1Idx / params.gSize;
        uint32_t gIdx = params.gS1Idx % params.gSize;
        for (uint32_t i = 0; i < params.dealRowCount;) {
            if (s1 < s1Tok) {
                uint32_t gNum = params.gSize - gIdx;
                if (i + gNum > params.dealRowCount) {
                    gNum = params.dealRowCount - i;
                }
                Duplicate(attenOutUb[i * params.columnCount],
                          static_cast<T>(AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO),
                          params.columnCount * gNum);
                AscendC::PipeBarrier<PIPE_V>();
                i += gNum;
                s1++;
                gIdx = 0;
                continue;
            }
            break;
        }
    }
}

template <typename OUT_T, typename SOFTMAX_T, const bool SOFTMAX_WITH_BRC>
__aicore__ inline void InvalidMaskRows(uint32_t softmaxOutOffset,
                                       uint32_t dealRowCount,
                                       uint32_t columnCount,
                                       LocalTensor<SOFTMAX_T> &softmaxMaxUb,
                                       uint32_t softmaxMinSaclar,
                                       LocalTensor<OUT_T> &bmm2ResUb)
{
    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(dealRowCount),
                                      static_cast<uint32_t>(columnCount),
                                      static_cast<uint32_t>(dealRowCount),
                                      static_cast<uint32_t>(columnCount)};

    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (SOFTMAX_WITH_BRC) {
        AdjustSoftMaxRes<OUT_T, SOFTMAX_T>(bmm2ResUb,
                                           softmaxMaxUb[softmaxOutOffset],
                                           softmaxMinSaclar,
                                           (OUT_T)AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO,
                                           softmaxShapeInfo);
    } else {
        AdjustSoftMaxRes<OUT_T, SOFTMAX_T, false, 1>(bmm2ResUb,
                                                     softmaxMaxUb[softmaxOutOffset],
                                                     softmaxMinSaclar,
                                                     (OUT_T)AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO,
                                                     softmaxShapeInfo);
    }
}

} // namespace AiInfraInferenceCommonFaBaseVector
#endif