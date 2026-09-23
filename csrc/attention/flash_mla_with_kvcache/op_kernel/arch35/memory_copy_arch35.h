/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEMORY_COPY_ARCH35_H
#define MEMORY_COPY_ARCH35_H

#include "../../../a5_mla_common/op_kernel/vector_common.h"
#include "../../../a5_mla_common/op_kernel/memcopy/gm_layout.h"
#include "../../../a5_mla_common/op_kernel/memcopy/parser.h"
#include "../../../a5_mla_common/op_kernel/memcopy/offset_calculator_v2.h"
#include "../../../a5_mla_common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../a5_mla_common/op_kernel/memcopy/fa_l1_tensor.h"
#include "../../../a5_mla_common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../a5_mla_common/op_kernel/memcopy/gm_coord.h"
#include "../../../a5_mla_common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../../a5_mla_common/op_kernel/memcopy/attn_copy_gm_to_ub.h"
#include "../../../a5_mla_common/op_kernel/memcopy/copy_ub_to_gm.h"

// ----------------------------------------------Copy LSE UB To Gm arch35--------------------------------
template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseBSNDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                    uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                    const CONST_INFO_T &constInfo, uint64_t s1LeftPaddingSize = 0)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.s1Size + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseBNSDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                    uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                    const CONST_INFO_T &constInfo, uint64_t qActSeqLens,
                                                    uint64_t s1LeftPaddingSize = 0)
{
    uint64_t gOffset = mOffset / qActSeqLens * constInfo.s1Size;
    uint64_t seqOffset = mOffset % qActSeqLens;
    uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
    uint64_t ubOffset = 0;

    // dealCount ≤ 当前actQs剩余部分，则直接搬运全部dealCount
    if ((qActSeqLens - seqOffset) >= dealCount) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        return;
    }
    // dealCount > 当前actQs剩余部分，分块搬运dealCount
    // dealCount首块
    uint64_t headActSeq = qActSeqLens - seqOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = headActSeq;
    dataCopyParams.blockLen = sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    outOffset += constInfo.s1Size - qActSeqLens + headActSeq;
    // ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    // dealCount中间块
    uint64_t pendingCount = dealCount - headActSeq;
    while (pendingCount > qActSeqLens) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = qActSeqLens;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += constInfo.s1Size;
        ubOffset += qActSeqLens * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        pendingCount -= qActSeqLens;
    }
    // dealCount尾块
    if (pendingCount > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = pendingCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    }
}

// 将 TND 的 LSE 写为 (N,T) 头主序；prefixBS1 为当前 batch 的累计 token 偏移。
template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDtoNTArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                       uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                       uint32_t prefixBS1, const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize + prefixBS1;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize + prefixBS1;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.t1Size + s1Idx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseNTDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                   uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                   const CONST_INFO_T &constInfo, uint32_t s1Size)
{
    uint32_t startS1Idx = mOffset % s1Size;
    uint32_t startGIdx = mOffset / s1Size;
    uint32_t endS1Idx = (mOffset + dealCount - 1) % s1Size;
    uint32_t endGIdx = (mOffset + dealCount - 1) / s1Size;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
        outOffset = bN2Offset + startS1Idx * constInfo.n2Size * constInfo.gSize + gIdx;
        if (gIdx != endGIdx) {
            curDealRowCount = s1Size - startS1Idx;
        } else {
            curDealRowCount = endS1Idx + 1 - startS1Idx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.gSize * constInfo.n2Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startS1Idx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}
#endif
