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
 * \file kernel_common_sink.h
 * \brief
 */
#ifndef KERNEL_COMMON_SINK_H
#define KERNEL_COMMON_SINK_H

#include "kernel_operator.h"

using namespace AscendC;
using AscendC::GlobalTensor;

namespace FusedInferAttentionScoreV2SinkBaseKernel {

__aicore__ inline uint64_t
GetActualQSeqLength(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx, const ConstInfo &constInfo)
{
    if (constInfo.actualLenQDims == 0) {
        return constInfo.qSeqSize;
    }
    if (constInfo.actualLenQDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumQSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline uint64_t GetActualKVSeqLength(GlobalTensor<uint64_t> &actualSeqLengths,
                                                uint32_t bIdx,
                                                const ConstInfo &constInfo,
                                                __gm__ uint8_t *keyPtr)
{
    if (constInfo.actualLenDims == 0) {
        if (!constInfo.batchContinuous) {
            return SeqLenFromTensorList<LAYOUT_T>(keyPtr, bIdx);
        }
        return constInfo.kvSeqSize;
    }
    if (constInfo.actualLenDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumKVSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline bool IsSkipCal(uint64_t gS1Idx,
                                 uint64_t actMSize,
                                 uint64_t s2Start,
                                 uint64_t s2DealSize,
                                 const RunInfo &info,
                                 const ConstInfo &constInfo)
{
    // 处理sinkTensor的循环，sink全部参与运算，不允许跳过
    if (unlikely(info.isSinkTensor)) {
        return false;
    }

    // 计算有效 s1Start、s1End
    int64_t s1GFirstToken = gS1Idx;
    int64_t s1GLastToken = s1GFirstToken + actMSize;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if constexpr (unlikely(LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NBSD ||
                           LAYOUT_T == FIA_LAYOUT::NTD)) {
        if (s1GFirstToken / info.actS1Size == s1GLastToken / info.actS1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % info.actS1Size;
            s1LastToken = s1GLastToken % info.actS1Size;
        } else {
            // start and end locate in tow or more G, but working same as crossing one complete block
            s1FirstToken = 0;
            s1LastToken = info.actS1Size;
        }
    } else {
        s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
        s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
    }

    // 计算有效的S2边界
    int64_t validS2FirstToken = s1FirstToken - info.preTokenLeftUp - 1;
    int64_t validS2LastToken = s1LastToken + info.nextTokenLeftUp;
    int64_t actS2LastToken = static_cast<int64_t>(s2Start + s2DealSize - 1);

    // 稀疏边界检查：如果当前S2块完全在有效S2边界外，则跳过计算
    if (unlikely((actS2LastToken <= validS2FirstToken) || (validS2LastToken <= s2Start))) {
        return true; // 需要跳过
    }
    return false; // 不需要跳过，需要计算
}

} // namespace FusedInferAttentionScoreV2SinkBaseKernel
#endif