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

#if __has_include("../../../common/op_kernel/vector_common.h")
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memcopy/gm_layout.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/offset_calculator_v2.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_l1_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../common/op_kernel/memcopy/gm_coord.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_ub.h"
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"
#else
#include "../../common/vector_common.h"
#include "../../common/memcopy/gm_layout.h"
#include "../../common/memcopy/parser.h"
#include "../../common/memcopy/offset_calculator_v2.h"
#include "../../common/memcopy/fa_gm_tensor.h"
#include "../../common/memcopy/fa_l1_tensor.h"
#include "../../common/memcopy/fa_ub_tensor.h"
#include "../../common/memcopy/gm_coord.h"
#include "../../common/memcopy/attn_copy_gm_to_l1.h"
#include "../../common/memcopy/attn_copy_gm_to_ub.h"
#include "../../common/memcopy/copy_ub_to_gm.h"
#endif

// post quant 拷贝
struct PostQuantInfo_V2 {
    uint32_t gSize;
    uint32_t dSize;
    uint32_t s1Size; // actualS1
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t gS1DealSize;
    uint32_t colCount;
};

template <typename PARAM_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
__aicore__ void CopyParamsGmToUb(LocalTensor<PARAM_T> &dstUb, FaGmTensor<PARAM_T, GM_FORMAT> &srcTensor,
                                 PostQuantInfo_V2 &postQuantInfo)
{
    OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;

    if constexpr (UB_FORMAT == UbFormat::S1G) {
        uint32_t s1IdxStart = postQuantInfo.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = postQuantInfo.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % offsetCalculator.GetDimG();

        if (s1IdxEnd - s1IdxStart > 1) {
            // 存在完整中间段, 拷贝完整G
            uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, 0, 0);
            uint32_t blockCount = offsetCalculator.GetDimG();
            CopySingleMatrixNDToND<PARAM_T>(dstUb, srcTensor.gmTensor[offset], offsetCalculator.GetDimG(),
                                            offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(),
                                            postQuantInfo.colCount);
        } else {
            // 处理第一段S1
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }
            uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, gIdxStart, 0);
            CopySingleMatrixNDToND<PARAM_T>(dstUb, srcTensor.gmTensor[offset], headSize, offsetCalculator.GetDimD(),
                                            offsetCalculator.GetStrideG(), postQuantInfo.colCount);

            // 处理第二段S1
            if ((s1IdxEnd - s1IdxStart == 1) && (gIdxEnd > 0)) {
                offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, 0, 0);
                uint32_t ubOffset = headSize * postQuantInfo.colCount;

                CopySingleMatrixNDToND<PARAM_T>(dstUb[ubOffset], srcTensor.gmTensor[offset], gIdxEnd,
                                                offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(),
                                                postQuantInfo.colCount);
            }
        }
    } else {
        uint32_t gIdxStart = postQuantInfo.gS1Idx / postQuantInfo.s1Size;
        uint32_t s1IdxStart = postQuantInfo.gS1Idx % postQuantInfo.s1Size;

        uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, gIdxStart, 0);
        // postQuantInfo.gS1DealSize + s1IdxStart是将第一个G的S1部分补齐后的总GS1行数
        CopySingleMatrixNDToND<PARAM_T>(
            dstUb, srcTensor.gmTensor[offset],
            ((postQuantInfo.gS1DealSize + s1IdxStart) + (postQuantInfo.s1Size - 1)) / postQuantInfo.s1Size,
            offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(), postQuantInfo.colCount);
    }
}

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

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                   uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                   const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + s1Idx * constInfo.n2Size * constInfo.gSize + startGIdx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDArch35NoGS1Merge(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                             uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                             const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.realGSize;
    uint32_t startGIdx = mOffset % constInfo.realGSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.realGSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.realGSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + s1Idx * constInfo.realN2Size * constInfo.realGSize + startGIdx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.realGSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

// flash_mla_with_kvcache LSE 输出契约为 (N,T) 头主序（learnings LSE 定论：aclnn/l0/csrc/python/golden 四点一致）。
// FIA 派生的 DataCopySoftmaxLseTNDArch35 / DataCopySoftmaxLseNTDArch35 均为 token 主序（FIA 原生 {t,n,1}），
// 与 flash_mla_with_kvcache (N,T) 输出不符 —— flash_attn 用 DataCopySoftmaxLseTNDtoNTArch35 做 TND→(N,T) 转置写出，
// 本函数为该范式的 flash_mla_with_kvcache 移植（bN2Offset 基址由调用方给到 head 块起址，本函数负责写入连续 (N,T)）。
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
