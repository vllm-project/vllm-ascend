/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_common.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_COMMON_H
#define SPARSE_FLASH_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SFA_LAYOUT
{
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
	  SFA_LAYOUT LAYOUT_T = SFA_LAYOUT::BSND, SFA_LAYOUT KV_LAYOUT_T = SFA_LAYOUT::BSND,
          const int TEMPLATE_MODE = C_TEMPLATE, typename... Args>
struct SFAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SFA_LAYOUT layout = LAYOUT_T;
    static constexpr SFA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr int templateMode = TEMPLATE_MODE;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SFA_LAYOUT::PA_BSND);
};

// ================================Util functions==================================
template <typename T> __aicore__ inline T SFAAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

struct RunInfo {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t gIdx;
    uint32_t s1Idx;
    uint32_t s2Idx;
    uint32_t bn2IdxInCurCore;
    uint32_t curSInnerLoopTimes;
    uint64_t tndBIdxOffsetForQ;
    uint64_t tndBIdxOffsetForKV;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorARopeOffset;
    uint64_t tensorBRopeOffset;
    uint64_t attenOutOffset;
    uint64_t attenMaskOffset;
    uint64_t topKBaseOffset;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    bool isFirstSInnerLoop;
    bool isChangeBatch;
    uint32_t s2BatchOffset;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t mSize;
    uint32_t mSizeV;
    uint32_t mSizeVStart;
    uint32_t tndIsS2SplitCore;
    uint32_t tndCoreStartKVSplitPos;
    bool isBmm2Output;
    bool isValid = false;

    static constexpr uint32_t n2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t curActualSeqLenOri = 0ULL;

    uint32_t gS1Idx;
    uint64_t actS2Size = 1;
    uint32_t actMBaseSize;
    bool isLastS2Loop;
    int32_t nextTokensPerBatch = 0;
    int64_t threshold;
    uint32_t curTopKIdx = 0;
    uint64_t curOffsetInSparseBlock = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t SFA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    uint32_t mmResUbSize = 0U;        // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U;      // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U;      // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t headDimRope;
    uint64_t kvSeqSize = 0ULL;        // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    int64_t kvCacheBlockSize = 0;    // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    SFA_LAYOUT outputLayout;          // 输出的Transpose格式
    uint32_t sparseMode = 0;
    bool needInit = false;

    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // FlashDecoding场景, S2在核间切分的最大份数
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint32_t actualLenDimsQ = 0U;        // query的actualSeqLength 的维度
    uint32_t actualLenDimsKV = 0U;       // KV 的actualSeqLength 的维度

    // TND
    uint32_t s2Start = 0U;               // TND场景下，S2的起始位置
    uint32_t s2End = 0U;                 // 单核TND场景下S2循环index上限

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;       // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U;   // TNDFlashDecoding kv起始位置

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // sparse attr
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};

#endif // SPARSE_FLASH_ATTENTION_COMMON_H