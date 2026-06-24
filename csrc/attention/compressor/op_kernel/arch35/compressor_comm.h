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
 * \file compressor_comm.h
 * \brief
 */

#ifndef COMPRESSOR_COMM_H
#define COMPRESSOR_COMM_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace Compressor {
template <typename T>
__aicore__ inline T CeilDivT(T num1, T num2)
{
    if (num2 == 0) {
        return static_cast<T>(0);
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T>
__aicore__ inline T Trunc(T num, T rnd)
{
    return ((rnd) == 0) ? 0 : (((num) / (rnd) * (rnd)));
}

template <typename T>
__aicore__ inline T FloorPow2(T num)
{
    if (num == 0)
        return 1;
    for (uint32_t i = 1; i < sizeof(T) * 8; i <<= 1) {
        num |= (num >> i);
    }
    return num - (num >> 1);
}

template <typename T>
__aicore__ inline T CeilPow2(T num)
{
    if (num <= 1)
        return 1;
    num--;
    for (uint32_t i = 1; i < sizeof(T) * 8; i <<= 1) {
        num |= (num >> i);
    }
    num++;
    return num;
}

enum class X_LAYOUT : std::uint8_t {
    BSH = static_cast<std::uint8_t>(0),
    TH = static_cast<std::uint8_t>(1)
};

enum class X_DTYPE : std::uint8_t {
    BF16 = static_cast<std::uint8_t>(0),
    FP16 = static_cast<std::uint8_t>(1)
};

enum class COFF : std::uint8_t {
    DISABLE = static_cast<std::uint8_t>(1),
    OVERLAP = static_cast<std::uint8_t>(2)
};

enum class ROTARY_MODE : std::uint8_t {
    HALF = static_cast<std::uint8_t>(1),
    INTERLEAVE = static_cast<std::uint8_t>(2)
};

enum class CACHE_MODE : std::uint8_t {
    CONTINUOUS = static_cast<std::uint8_t>(1),
    CYCLE = static_cast<std::uint8_t>(2)
};

enum class TEMPLATE_ID : uint8_t {
    NORMAL = 0,
    EMPTY_X = 1,
    FULL_LOAD = 2
};

template <X_LAYOUT X_L, X_DTYPE X_T, COFF C, ROTARY_MODE Rotary_Mode, CACHE_MODE Cache_Mode, typename... Args>
struct COMPType {
    static constexpr X_LAYOUT xLayout = X_L;
    static constexpr X_DTYPE xDtype = X_T;
    static constexpr COFF coff = C;
    static constexpr ROTARY_MODE rotaryMode = Rotary_Mode;
    static constexpr CACHE_MODE cacheMode = Cache_Mode;
};

struct CmpBlockInfo {
    __aicore__ inline CmpBlockInfo(){};
    __aicore__ inline CmpBlockInfo(uint32_t bIdx, uint32_t sIdx, bool needReset = false)
        : bIdx(bIdx), sIdx(sIdx), needReset(needReset){};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;
    bool needReset = false;
    bool isFirst = true;

    uint32_t headSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailSeqCnt = 0U;
    bool isCompress = 0U;
};

struct BasicBlockInfo {
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
    uint32_t compressedTcNum = 0;
    uint32_t dealSeqCnt = 0;
    uint32_t dealTcNum = 0;
};

struct BatchInfo {
    uint32_t tcNum = 0;
    uint32_t compressedTcNum = 0;
    uint32_t remSeqCnt = 0;
    uint32_t seqCnt = 0;
    uint32_t seqUsedCnt = 0;
    uint32_t headHolderSeq = 0;
    uint32_t bStartPos = 0;
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
};

struct ConstInfo {
    // 整个AICORE的任务信息, 左闭右开区间[ (bStart, s2Start), (bEnd, s2End) )
    uint32_t bStart = 0U;
    uint32_t sStart = 0U;
    uint32_t bEnd = 0U;
    uint32_t sEnd = 0U;

    // 分核相关
    uint32_t usedCoreNum = 0;
    uint32_t dBaseSize = 0;
    uint32_t mBaseSize = 0;
    uint32_t kBaseSize = 0;
    uint32_t kBaseNum = 0;
    uint32_t tcSize = 0;
    uint32_t tcBaseSize = 0;
    uint32_t tcBasicBlockNum = 0;
    uint32_t dBasicBlockNum = 0;
    uint32_t coreGroupNum = 0;
    uint32_t singleCoreDealTcBasicNum = 0;
    uint32_t dIdx = 0;
    uint32_t mStart = 0;
    uint32_t mEnd = 0;
    uint32_t nStart = 0;
    uint32_t nEnd = 0;
    uint32_t kStart = 0;
    uint32_t kEnd = 0;
    uint32_t mLoopNum = 0;
    uint32_t bIdxOfLastTc = 0;
    uint32_t sIdxOfLastTc = 0;
    uint32_t mGroupNum = 0;
    uint32_t mCurGroupIdx = 0;

    // shape及参数
    uint32_t batchSize = 0;
    uint32_t hSize = 0;
    uint32_t sSize = 0;
    uint32_t headDim = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t cmpRatio = 0;
    float normEps = 1e-6;
    float reciprocalD = 0;
    uint64_t stateCacheStrideDim0 = 0;

    uint32_t curGroupIdx = 0;
    uint32_t tailGroupIdx = 0;
    uint32_t tailBasicBlockNum = 0;
    uint32_t realDealBasicBlockNum = 0;

    // pageAttention
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    // workSpace
    uint32_t dbWorkspaceRatio = 1;
    uint32_t mm1KvResSize = 0;
    uint32_t mm1ScoreResSize = 0;
    uint32_t vec1TailCacheSize = 0;
    uint32_t vec1ResSize = 0;
    uint32_t mm1ResSize = 0; // 所有cube输出kv/score结果的总大小

    uint32_t aiCoreIdx = 0;
    uint32_t nSize = 0;

    uint32_t dbSize = 0;
};

struct RunInfo {
    bool isValid = false;
    uint32_t cubeDbIdx = 0; // kernel主循环索引

    // 增加字段
    uint32_t dealTcNum = 0;
    // 右边相关信息
    uint32_t bStart = 0;
    uint32_t sStart = 0;
    uint32_t dealSeqCnt = 0;
    // 左边相关信息
    uint32_t preBStart = 0;
    uint32_t preSStart = 0;
    uint32_t preDealSeqCnt = 0;  // 左边需要处理的s大小
    uint32_t preFirstSeqCnt = 0; // 左边首块大小

    uint32_t kStartIdx = 0;
    uint32_t dealKSize = 0;
    uint32_t hStart = 0;

    uint32_t bEnd = 0;
    uint32_t sEnd = 0;
    uint32_t bStartSeqIdx = 0;
    uint32_t bEndSeqIdx = 0;

    // v2分核信息 sc是左闭右开
    uint32_t scStart = 0;
    uint32_t scEnd = 0;
    uint32_t dealScSize = 0;

    // vec1Res offset
    uint64_t vec1ResOffset = 0;
};

struct Vec1RunInfo {
    // vec相关信息，一次syncAll需处理数据的起始索引
    bool resetResFlag = false; // v1积攒N轮 是否是N轮的起始轮
    uint32_t c1v1DbIdx = 0;    // vec1 doubleBuffer索引
    uint32_t v1v2DbIdx = 0;    // v1v2 doubleBuffer索引
    uint32_t bStart = 0;
    uint32_t sStart = 0;
    uint32_t dealTcNum = 0;
    uint32_t dealScSize = 0;
};

struct Vec2RunInfo {
    // uint32_t bStart = 0;
    uint32_t v2DbIdx = 0; // v2 doubleBuffer索引
    uint32_t sStart = 0;
    uint32_t bEnd = 0;
    uint32_t sEnd = 0;
    // v2分核信息 sc是左闭右开
    uint32_t scStart = 0;
    uint32_t scEnd = 0;

    // 增加字段
    uint32_t bStart = 0;
    uint32_t compressedId = 0;
    uint32_t bCompressedId = 0;
    uint32_t dealScSize = 0;
};

struct MSplitInfo {
    uint32_t vecStartB = 0U;
    uint32_t vecStartS = 0U;
    uint32_t vecEndB = 0U;
    uint32_t vecEndS = 0U;
    uint32_t dealTcNum = 0U;
    // vec1Res offset
    uint64_t vec1StartOffset = 0;
    uint64_t vec1ResOffset = 0;
};

struct BlockInfo {
    __aicore__ inline BlockInfo(uint32_t bIdx, uint32_t sIdx, uint32_t dealSeqSize)
        : bIdx(bIdx), sIdx(sIdx), dealSeqSize(dealSeqSize){};
    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t dealSeqSize = 0;

    uint32_t isFirst = true;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;
    uint32_t headHolderSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailHolderSeqCnt = 0U;
    uint32_t dealTcSize = 0U;
    uint32_t tailValidSeqCnt = 0U;
    uint32_t compressTcSize = 0U;
};

struct LoopInfo {
    uint32_t groupSize = 0U;
    uint32_t groupNum = 0U;
    uint32_t coreRowIdx = 0U;
    uint32_t coreColIdx = 0U;
    uint32_t dLoopIdx = 0U;
    bool isCoreRowFirst = false;
    bool isCoreRowLast = false;
    bool isCoreLoopFirst = false;
    bool isCoreLoopLast = false;
};

struct Vec1SplitInfo {
    uint32_t dealSeqStartIdx = 0;
    uint32_t dealSeqCnt = 0;
    uint32_t dBaseSize = 0;
    uint32_t vec1GroupSize = 0;
    uint32_t vec1GroupNum = 0;
    uint32_t dealTcSize = 0;
    uint32_t dealTcNum = 0;
    uint32_t dealBatchNum = 0;
    uint32_t preDealTcSize = 0;
    uint32_t preDealBatchNum = 0;
    uint32_t curBStart = 0;
    uint32_t curSStart = 0;
    uint32_t curCompressedCnt = 0;
    uint32_t preCompressedCnt = 0;
    uint32_t totalCompressedCnt = 0;
    uint32_t tcSplitSize = 0;
    uint32_t dSplitSize = 0;
    uint32_t dLoopCount = 0;
};

struct Vec2SplitInfo {
    uint32_t dealedScCnt = 0;
    uint32_t preScCnt = 0;
    uint32_t dealScNum = 0;
    uint32_t curBStart = 0;
    uint32_t curScStart = 0;
};

// BUFFER的字节数
inline constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
inline constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
inline constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
inline constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
inline constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
inline constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
inline constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
inline constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
inline constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
inline constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
inline constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;

// BLOCK和REPEAT的字节数
inline constexpr uint64_t BYTE_BLOCK = 32UL;
inline constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;

template <typename T>
__aicore__ inline constexpr T BlockElementNum()
{
    return BYTE_BLOCK / sizeof(T);
}

template <typename T>
__aicore__ inline constexpr T RepeatElementNum()
{
    return REPEAT_BLOCK_BYTE / sizeof(T);
}
// BLOCK和REPEAT的FP32元素数
inline constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float); // 8
inline constexpr uint32_t FP16_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(bfloat16_t);    // 16
inline constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float); // 64
inline constexpr uint32_t REPEAT_STRIDE_NUM = REPEAT_BLOCK_BYTE / BYTE_BLOCK; // 8
inline constexpr uint32_t REPEAT_MAX_NUM = 255;
inline constexpr uint32_t BRCB_NUM = 8;
inline constexpr uint32_t MAX_R = 256;

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor, uint32_t nValue,
                                              uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; // nd矩阵的行数
    if constexpr (IsSameType<T, int4b_t>::value) {
        constexpr uint32_t HALF_SIZE_DIVISOR = 2;
        nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
        nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
    } else {
        nd2nzPara.dValue = dValue;       // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
    }
    nd2nzPara.dstNzC0Stride = dstNzC0Stride;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}
template <typename T>
__aicore__ inline void DumpTensorForDim2(GlobalTensor<T> tensor, uint32_t desc, uint32_t dumpSize, uint32_t row,
                                         uint32_t col)
{
    uint32_t array2[] = {static_cast<uint32_t>(row), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(LocalTensor<T> tensor, uint32_t desc, uint32_t dumpSize, uint32_t row,
                                         uint32_t col)
{
    uint32_t array2[] = {static_cast<uint32_t>(row), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(LocalTensor<T> tensor, uint32_t desc, uint32_t dumpSize)
{
    uint32_t col = 32 / sizeof(T);
    uint32_t array2[] = {static_cast<uint32_t>(dumpSize / col), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

template <typename T>
__aicore__ inline void DumpTensorForDim2(GlobalTensor<T> tensor, uint32_t desc, uint32_t dumpSize)
{
    uint32_t col = 32 / sizeof(T);
    uint32_t array2[] = {static_cast<uint32_t>(dumpSize / col), static_cast<uint32_t>(col)};
    AscendC::ShapeInfo shapeInfo(2, array2);
    // AscendC::DumpTensor(tensor, desc, dumpSize, shapeInfo);
}

} // namespace Compressor
#endif
