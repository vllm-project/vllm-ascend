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
 * \file quant_lightning_indexer_v2_common_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_H

namespace QLIV2Common {

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
    const bool PAGE_ATTENTION = false,
          LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename... Args>
struct QLIV2Type {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isValid = false;
    bool isNeedLD = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
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
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;              // topK选取大小
    uint64_t kSeqSize = 0ULL;          // kv最大S长度
    uint64_t qSeqSize = 1ULL;          // q最大S长度
    uint32_t kCacheBlockSize = 0;      // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0;  // PA场景的最大单batch block number
    uint32_t keyStride0 = 0;           // PA 0-axis stride of key (elements)
    uint32_t keyDequantScaleStride0 = 0; // PA 0-axis stride of key scale (elements)
    LI_LAYOUT outputLayout;            // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;             // 压缩率

    uint32_t actualLenQDims = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;   // KV 的actualSeqLength 的维度
    uint32_t cmpResiduaKLenDims = 0U;   // cmpResidualK的维度
    bool isAccumSeqS1 = false;     // 是否累加模式
    bool isAccumSeqS2 = false;     // 是否累加模式

    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    uint32_t coreEnable = 0U;
};

struct LdSplitCoreInfo {
        bool isLdCoreEnable = false;     // 当前核是否参与规约任务
        uint32_t saveWorkSpaceIdx = 0U;  // 存放LD参数的地址
        uint32_t bn2Idx = 0U;            // 归约任务
        uint32_t bIdx = 0U;
        uint32_t n2Idx = 0U;
        uint32_t mIdx = 0U;
        uint32_t workspaceIdx = 0U;      // 当前AIV核上规约任务的索引
        uint32_t workspaceNum = 0U;      // 当前AIV核上规约任务的S2切分数量
        uint32_t mStart = 0U;
        uint32_t mNum = 0U;
        uint64_t indiceOutCoreOffset = 0U;  // 最终输出索引搬出Topk的初始偏移地址
    };

template <typename T1, typename T2>
__aicore__ inline T1 Align(T1 num, T2 rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
}  // namespace QLIV2Common

#endif  // QUANT_LIGHTNING_INDEXER_V2_COMMON_H