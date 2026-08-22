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
 * \file vector_common.h
 * \brief
 */
#ifndef VECTOR_COMMON_H
#define VECTOR_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "const_def.h"

using namespace AttentionCommon;
using namespace AscendC;
using AscendC::LocalTensor;

namespace fa_base_vector {

// BLOCK和REPEAT的字节数
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;
// 最大repeat次数
constexpr uint32_t MAX_REPEAT_TIMES = 255;
constexpr int64_t HALF_NUM = 2;
constexpr int64_t STRIDE_LENGTH = 8;
constexpr int64_t MAX_VALID_LENGTH = 1024;
enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    TREE = 9,
};

__aicore__ inline bool IsExistInvalidRows(int64_t nextTokensPerBatch, int64_t preTokensPerBatch, uint32_t mode,
                                          bool attenMaskFlag, bool isRowInvalid)
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

__aicore__ inline void GetSafeActToken(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &safePreToken,
                                       int64_t &safeNextToken, uint32_t mode)
{
    if (mode == DEFAULT_MASK) {
        safePreToken = AttentionCommon::Max(-actSeqLensKv, safePreToken);
        safePreToken = AttentionCommon::Min(safePreToken, actSeqLensQ);
        safeNextToken = AttentionCommon::Max(-actSeqLensQ, safeNextToken);
        safeNextToken = AttentionCommon::Min(safeNextToken, actSeqLensKv);
    } else if (mode == BAND) {
        safePreToken = AttentionCommon::Max(-actSeqLensQ, safePreToken);
        safePreToken = AttentionCommon::Min(safePreToken, actSeqLensKv);
        safeNextToken = AttentionCommon::Max(-actSeqLensKv, safeNextToken);
        safeNextToken = AttentionCommon::Min(safeNextToken, actSeqLensQ);
    }
}

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    // dstRepStride为0~255,columnCount需要小于2048
    if (columnCount < REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
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

__aicore__ inline void VecMulMatForBigRowCount(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                               LocalTensor<float> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    uint32_t repeatTimes = MAX_REPEAT_TIMES;
    for (uint32_t i = 0; i < dealRowCount; i += MAX_REPEAT_TIMES) {
        if (i + MAX_REPEAT_TIMES > dealRowCount) {
            repeatTimes = dealRowCount - i;
        }
        VecMulMat(dstUb[i * columnCount], src0Ub, src1Ub[i * columnCount], repeatTimes, columnCount, actualColumnCount);
    }
}

__aicore__ inline void MatDivVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src1Ub[i, j] / src0Ub[j],
    // src0Ub:[dealRowCount, actualColumnCount] src1Ub:[1, columnCount] dstUb:[dealRowCount, columnCount]
    // restraint: dealRowCount < 256
    // dstRepStride为0~255,columnCount需要小于2048
    if (columnCount < REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src1RepStride = 0;
        repeatParams.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
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

__aicore__ inline void VecMulBlkMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, 8] dstUb:[dealRowCount, columnCount]
    BinaryRepeatParams repeatParams;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;
    // dstRepStride为0~255,columnCount需要小于2048
    if (columnCount < REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
        // [1, columnCount] * [dealRowCount, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
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
            Mul(dstUb[i * columnCount], src0Ub, src1Ub[i * AttentionCommon::FP32_BLOCK_ELEMENT_NUM], mask, loopCount,
                repeatParams);
            if (remainCount > 0) {
                Mul(dstUb[i * columnCount + loopCount * mask], src0Ub[loopCount * mask],
                    src1Ub[i * AttentionCommon::FP32_BLOCK_ELEMENT_NUM], remainCount, 1, repeatParams);
            }
        }
    }
}

__aicore__ inline void VecAddMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    // dstRepStride为0~255,columnCount需要小于2048
    if (columnCount < REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
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

__aicore__ inline void VecAddMatForBigRowCount(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                               LocalTensor<float> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    uint32_t repeatTimes = MAX_REPEAT_TIMES;
    for (uint32_t i = 0; i < dealRowCount; i += MAX_REPEAT_TIMES) {
        if (i + MAX_REPEAT_TIMES > dealRowCount) {
            repeatTimes = dealRowCount - i;
        }
        VecAddMat(dstUb[i * columnCount], src0Ub, src1Ub[i * columnCount], repeatTimes, columnCount, actualColumnCount);
    }
}

template <typename T>
__aicore__ inline void RowDivs(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, AttentionCommon::FP32_BLOCK_ELEMENT_NUM]
    // dstUb:[dealRowCount, columnCount]
    uint32_t repeatNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        repeatNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM * 2;
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
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, AttentionCommon::FP32_BLOCK_ELEMENT_NUM]
    // dstUb:[dealRowCount, columnCount] dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;                // 256/4 * 2=128
        blockElementNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM * 2; // 32/4 * 2 = 16
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
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
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
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void RowSum(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
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
    repeatParamsMax.src0RepStride = columnCount / (AttentionCommon::BYTE_BLOCK / sizeof(float));
    repeatParamsMax.src1RepStride = columnCount / (AttentionCommon::BYTE_BLOCK / sizeof(float));
    repeatParamsMax.dstRepStride = columnCount / (AttentionCommon::BYTE_BLOCK / sizeof(float));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Add(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / (AttentionCommon::BYTE_BLOCK / sizeof(float)));
}

__aicore__ inline uint32_t GetMinPowerTwo(uint32_t cap)
{
    uint32_t i = 1;
    while (i < cap) {
        i = i << 1;
    }
    return i;
}

__aicore__ inline void RowSumForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    // columnCount要求32元素对齐
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
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

__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
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
    repeatParamsMax.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void RowMaxForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * AttentionCommon::FP32_BLOCK_ELEMENT_NUM) {
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

__aicore__ inline void MatDivsVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsDiv);
    }
}

__aicore__ inline void RowSub(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsSub);
    }
}

__aicore__ inline void ColMax(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsMax);
    }
}

__aicore__ inline void ColAdd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsAdd);
    }
}

template <typename T>
__aicore__ inline void ComputeSoftMaxLse(LocalTensor<T> &softmaxlseUb, LocalTensor<T> &softmaxSumUb,
                                         LocalTensor<T> &softmaxMaxUb, uint32_t dealRowCount)
{
    uint32_t blockNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        blockNum = AttentionCommon::FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint64_t dealRowCountAlign = dealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    Log(softmaxlseUb, softmaxSumUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxlseUb, softmaxlseUb, softmaxMaxUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
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

    // for bss & bs
    uint32_t batchIdx;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskStride;

    LAYOUT_Q layout;
    MaskDataType attenMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;

    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
};

__aicore__ inline uint64_t ComputeAttenMaskOffsetNoCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    uint64_t bOffset = static_cast<uint64_t>(info.batchIdx) * static_cast<uint64_t>(info.attenMaskBatchStride);
    uint64_t s1Offset = (info.s1LeftPaddingSize + s1StartIdx % info.s1Size) * info.attenMaskStride;
    uint64_t s2Offset = info.s2LeftPaddingSize + info.s2StartIdx;
    return bOffset + s1Offset + s2Offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfo &info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.sparseMode == BAND) {                                          // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
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
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskStride; // min(delta, s2inner)
    }
    return offset;
}

template <bool ENABLE_TREE = false>
__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfo &info, uint32_t s1StartIdx = 0, uint64_t treeMaskStart = 0,
                                                  bool isPre = false)
{
    if (isPre) {
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    } else {
        if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
            return ComputeAttenMaskOffsetNoCompress(info, s1StartIdx);
        }
        if constexpr (ENABLE_TREE) {
            uint64_t bOffset = info.attenMaskBatchStride;
            uint64_t s1Offset = (s1StartIdx % info.s1Size) * info.attenMaskStride;
            uint64_t s2Offset = info.s2StartIdx > treeMaskStart ? info.s2StartIdx - treeMaskStart : 0;
            return bOffset + s1Offset + s2Offset;
        }
        return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
    }
}

template <typename T, bool ENABLE_TREE = false>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
{
    uint32_t treeMaskStart = 0;
    if constexpr (ENABLE_TREE) {
        treeMaskStart = info.s2Size - info.s1Size;
        uint32_t curS2EnsPos = info.s2StartIdx + info.s2dealNum;
        // 只有info.s2StartIdx + info.s2dealNum > treeMaskStart时，才会进入此流程；
        // 当info.s2StartIdx > treeMaskStart，mask拷贝也是全量拷贝，和其余sparse过程相同
        if (info.s2StartIdx < treeMaskStart) {
            if (curS2EnsPos <= treeMaskStart) {
                return; // 整个 tile 在零区，UB 已初始化为 0，无需拷贝
            }
            // 由于sparse9 mask只有一部分，不能合并处理
            uint32_t attenMaskSize = curS2EnsPos - treeMaskStart;
            uint32_t attenMaskSizeAlign = AttentionCommon::Align(static_cast<uint32_t>(attenMaskSize + treeMaskStart % 32), 32U);
            uint64_t maskOffset = ComputeAttenMaskOffset<ENABLE_TREE>(info, s1StartIdx, treeMaskStart, isPre);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
            dataCopyParams.blockLen = curS2EnsPos - treeMaskStart;
            dataCopyParams.srcStride = info.attenMaskStride - (curS2EnsPos - treeMaskStart);
            dataCopyParams.dstStride =
                (treeMaskStart - info.s2StartIdx) / 32; // dst在UB上，单位为32字节，同时左侧不对齐场景会进行左padding
            DataCopyPadExtParams<bool> padParams;
            padParams.isPad = true;
            padParams.leftPadding = static_cast<uint8_t>(treeMaskStart % 32);
            padParams.rightPadding = static_cast<uint8_t>(attenMaskSizeAlign - (attenMaskSize + treeMaskStart % 32));
            padParams.paddingValue = 0;
            DataCopyPad(attenMaskUb[(treeMaskStart - info.s2StartIdx) / 32 * 32], srcGmAddr[maskOffset], dataCopyParams,
                        padParams);
            return;
        }
    }
    // 标准拷贝路径（非TREE模式，或TREE模式但s2StartIdx >= treeMaskStart时）
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset<ENABLE_TREE>(info, s1StartIdx, treeMaskStart, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};

    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, typename U, bool ENABLE_TREE = false>
__aicore__ inline void AttentionmaskCopyInForGsLayout(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    if (info.gs1dealNum <= info.s1Size) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy<T, ENABLE_TREE>(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskSizeAlign];
            AttentionmaskDataCopy<T, ENABLE_TREE>(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy<T, ENABLE_TREE>(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
    } else {
        AttentionmaskDataCopy<T, ENABLE_TREE>(attenMaskUb, srcGmAddr, info, 0, info.s1Size, isPre);
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
            DataCopy(attenMaskUbDst[(headS1Count + i * info.s1Size) * attenMaskSizeAlign], attenMaskUb,
                     info.s1Size * attenMaskSizeAlign);
        }
        // tail
        if (tailS1Size > 0) {
            DataCopy(attenMaskUbDst[(headS1Count + midGCount * info.s1Size) * attenMaskSizeAlign], attenMaskUb,
                     tailS1Size * attenMaskSizeAlign);
        }
        attenMaskUb = attenMaskUbDst;
    }
}

template <typename T, typename U, bool ENABLE_TREE = false>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);

    AttentionmaskDataCopy<T, ENABLE_TREE>(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx + 1, isPre);

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
    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                         headGCount, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    dstMaskOffset += headGCount * attenMaskSizeAlign / sizeof(int16_t);
    srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);

    // mid
    uint32_t reminRowCount = info.gs1dealNum - headGCount;
    uint32_t midS1Count = reminRowCount / info.gSize;
    uint32_t tailGSize = reminRowCount % info.gSize;

    for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             info.gSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
        dstMaskOffset += info.gSize * attenMaskSizeAlign / sizeof(int16_t);
        srcMaskBaseOffset += attenMaskSizeAlign / sizeof(int16_t);
    }
    // tail
    if (tailGSize > 0) {
        Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset], AscendC::MASK_PLACEHOLDER,
                             tailGSize, {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
    }
    SetMaskNorm();
    ResetMask();
    attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
}

template <bool ENABLE_TREE = false>
__aicore__ inline bool IsSkipAttentionmask(MaskInfo &info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }

    // 增加sparse = 9的处理
    if constexpr (ENABLE_TREE) {
        // 由于分核时按照Batch进行划分，sparse9在每个batch的所有 S 跳过的范围固定，所以不区分跨g轴的情况
        if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) >
            static_cast<int64_t>(info.s2Size - info.s1Size)) {
            return false;
        } else {
            return true;
        }
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t nextToken = 0; // sparse2 本身远点就在左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为远点计算Token
    } else if (info.sparseMode == BAND) {                                          // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
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

template <typename T, typename U, bool ENABLE_TREE = false>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                           LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    if (info.layout == GS) {
        AttentionmaskCopyInForGsLayout<T, U, ENABLE_TREE>(attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else if (info.layout == SG) { // sg
        AttentionmaskCopyInForSgLayout<T, U, ENABLE_TREE>(attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else if (info.layout == S1_EQUAL1) {
        uint32_t treeMaskStart = 0;
        if constexpr (ENABLE_TREE) {
            treeMaskStart = info.s2Size - info.s1Size;
            uint32_t curS2EndPos = info.s2StartIdx + info.s2dealNum;
            if (info.s2StartIdx >= treeMaskStart) {
                // TREE 但 s2StartIdx >= treeMaskStart：全量拷贝
                uint64_t maskOffset = ComputeAttenMaskOffset<ENABLE_TREE>(info, 0, treeMaskStart, isPre);
                uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = info.s2dealNum;
                dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
                dataCopyParams.dstStride = 0;
                DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum),
                                                     0};
                DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
            } else if (curS2EndPos > treeMaskStart) {
                // 部分拷贝：只拷贝 [treeMaskStart, curS2EndPos) 区域
                uint32_t attenMaskSize = curS2EndPos - treeMaskStart;
                uint32_t attenMaskSizeAlign = AttentionCommon::Align(static_cast<uint32_t>(attenMaskSize + treeMaskStart % 32), 32U);
                uint64_t maskOffset = ComputeAttenMaskOffset<ENABLE_TREE>(info, 0, treeMaskStart, isPre);
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = curS2EndPos - treeMaskStart;
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPadExtParams<bool> padParams;
                padParams.isPad = true;
                padParams.leftPadding = static_cast<uint8_t>(treeMaskStart % 32);
                padParams.rightPadding =
                    static_cast<uint8_t>(attenMaskSizeAlign - (attenMaskSize + treeMaskStart % 32));
                padParams.paddingValue = 0;
                DataCopyPad(attenMaskUb[(treeMaskStart - info.s2StartIdx) / 32 * 32], srcGmAddr[maskOffset],
                            dataCopyParams, padParams);
            }
            // else: curS2EndPos <= treeMaskStart，整个 tile 在零区，UB 已初始化为 0，无需拷贝
        } else {
            // 非 TREE：全量拷贝
            uint64_t maskOffset = ComputeAttenMaskOffset<ENABLE_TREE>(info, 0, 0, isPre);
            uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = info.s2dealNum;
            dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
            DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
        }

        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(enQueEvtID);
        WaitFlag<HardEvent::MTE2_V>(enQueEvtID);
        for (uint32_t i = 1; i < info.gs1dealNum; i++) {
            uint32_t offset = i * AttentionCommon::Align(info.s2dealNum, 32U);
            DataCopy(attenMaskUb[offset], attenMaskUb, AttentionCommon::Align(info.s2dealNum, 32U));
        }
    }
}

template <typename T, typename M, typename U>
__aicore__ inline void AttentionMaskCompute(LocalTensor<T> &dstUb, LocalTensor<T> &srcUb, LocalTensor<M> &attenMaskUb,
                                            LocalTensor<U> &tmpBuf, MaskInfo &info, bool isPre = false)
{
    uint32_t dealRowCount = info.gs1dealNum;
    uint32_t columnCount = AttentionCommon::Align(info.s2dealNum, 32U);
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    if (info.attenMaskType != MASK_FP16) {
        // int8 & uint8 is ok
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        srcUb.SetSize(dealRowCount * columnCount);              // Select接口要求src size与参数匹配
        if (isPre) {
            SelectWithBytesMask(dstUb, *((T *)&info.maskValue), srcUb, attenMaskUb, tmpBuf,
                                selectWithBytesMaskShapeInfo);
        } else {
            SelectWithBytesMask(dstUb, srcUb, *((T *)&info.maskValue), attenMaskUb, tmpBuf,
                                selectWithBytesMaskShapeInfo);
        }
        constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
        srcUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
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

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbInputFormat GeInputUbFormat()
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

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return UbInputFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return UbInputFormat::GS1;
    }
}

template <typename T, UbInputFormat UB_INPUTFORMAT>
class InvalidRows {
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
                Duplicate(attenOutUb[s1RealStart * params.columnCount], static_cast<T>(FLOAT_ZERO),
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
        uint64_t dealRowOffset = 0;
        if (s1 < s1BottomTok) {
            // 如果s1 < 行无效开始行，偏移s1到行无效开始行
            s1 = s1BottomTok;
            dealRowOffset = s1BottomTok * params.gSize - params.gS1Idx;
            gIdx = 0;
        }
        while (s1 < params.actS1Size && dealRowOffset < params.dealRowCount) {
            uint32_t gNum = params.gSize - gIdx;
            if (dealRowOffset + gNum > params.dealRowCount) {
                gNum = params.dealRowCount - dealRowOffset;
            }
            Duplicate(attenOutUb[dealRowOffset * params.columnCount], static_cast<T>(FLOAT_ZERO),
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
                Duplicate(attenOutUb[i * params.columnCount], static_cast<T>(FLOAT_ZERO), params.columnCount * s1Num);
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
                Duplicate(attenOutUb[i * params.columnCount], static_cast<T>(FLOAT_ZERO), params.columnCount * gNum);
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
__aicore__ inline void InvalidMaskRows(uint32_t softmaxOutOffset, uint32_t dealRowCount, uint32_t columnCount,
                                       LocalTensor<SOFTMAX_T> &softmaxMaxUb, uint32_t softmaxMinSaclar,
                                       LocalTensor<OUT_T> &bmm2ResUb)
{
    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(dealRowCount), static_cast<uint32_t>(columnCount),
                                      static_cast<uint32_t>(dealRowCount), static_cast<uint32_t>(columnCount)};

    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (SOFTMAX_WITH_BRC) {
        AdjustSoftMaxRes<OUT_T, SOFTMAX_T>(bmm2ResUb, softmaxMaxUb[softmaxOutOffset], softmaxMinSaclar,
                                           (OUT_T)FLOAT_ZERO, softmaxShapeInfo);
    } else {
        AdjustSoftMaxRes<OUT_T, SOFTMAX_T, false, 1>(bmm2ResUb, softmaxMaxUb[softmaxOutOffset], softmaxMinSaclar,
                                                     (OUT_T)FLOAT_ZERO, softmaxShapeInfo);
    }
}

} // namespace fa_base_vector
#endif
