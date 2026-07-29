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
 * \file lightning_indexer_decode_update_common.h
 * \brief
 */
#ifndef lightning_indexer_decode_update_COMMON_H
#define lightning_indexer_decode_update_COMMON_H

namespace LICommon {
constexpr int32_t ROW_MODE_PAD = 0;
constexpr int32_t ROW_MODE_DENSE = 1;
constexpr int32_t ROW_MODE_SPARSE = 2;

template <typename T>
struct LIType {
    using queryType = T;
    using keyType = T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t s2Idx;
    uint32_t actS2Size;
    uint32_t originalSeqLen;
    uint32_t cacheRowIdx;
    int32_t cacheMetadata;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isSparse;
    bool isFirstFill;
};

struct ConstInfo {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr int INVALID_IDX = -1;
    static constexpr uint32_t mBaseSize = 64;
    static constexpr uint32_t s2BaseSize = 512;
    static constexpr uint64_t headDim = 128;
    static constexpr uint64_t sparseCount = 2048;
    static constexpr uint64_t outputCapacity = 16384;

    static constexpr uint32_t syncC1V1 = 0;
    static constexpr uint32_t syncV1C1 = 0;

    uint64_t batchSize = 0ULL;
    uint64_t kSeqSize = 0ULL;
    uint32_t kCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t qHeadNum = 0;
    uint32_t cacheRowWidth = 0;
    uint32_t metadataIndex = 0;
    uint32_t tokenCapacity = 0;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
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
} // namespace LICommon

#endif // lightning_indexer_decode_update_COMMON_H
