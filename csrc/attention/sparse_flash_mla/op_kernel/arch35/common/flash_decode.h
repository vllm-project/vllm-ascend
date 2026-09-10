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
 * \file flash_decode.h
 * \brief S2-split Flash Decode data-plane: staging layout, Vec1 LSE staging,
 *        Vec2 partial-O staging, and FD chunk reduction.
 *
 * GM staging layout (three contiguous regions):
 *   [partial O slots][max slots][sum slots]
 *
 * Per-slot sizes:
 *   partial O : stagingM * dAlign * sizeof(float)
 *   max       : stagingM * broadcastElems * sizeof(float)   (each row is broadcastElems identical floats)
 *   sum       : stagingM * broadcastElems * sizeof(float)
 *
 * Numerics:
 *   Vec2 must write normalized p_i = sum(exp(score-m_i)*V) / s_i (after final division).
 *   FD Reduce computes M = max(m_i), G = sum(exp(m_i-M)*s_i), w_i = exp(m_i-M)*s_i/G, O = sum(w_i*p_i).
 *
 * Synchronization:
 *   - All participating cores must enter SyncAll() after FA completes before FD reads staging.
 *   - SyncAll() is owned by the operator kernel, not this file.
 *   - Event IDs (V_MTE3, MTE3_V, V_MTE2, MTE2_V) are passed in by the caller.
 *
 * workspaceNum limit: currently maxSplits = 2. Caller must ensure 1 <= workspaceNum <= maxSplits.
 */
#ifndef SPARSE_FLASH_MLA_FLASH_DECODE_H
#define SPARSE_FLASH_MLA_FLASH_DECODE_H

#include <stdint.h>

#if __has_include("../../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#elif __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#else
#include "../common/arch35/vf/vf_flash_decode.h"
#endif

namespace AttentionCommon {

constexpr int64_t FD_MAX_S2_SPLIT_NUM = 2U;
constexpr int64_t FD_BROADCAST_ELEMS_PER_ROW = 8U;
constexpr int64_t FD_REDUCE_CHUNK_ROWS = 16U;
static constexpr int64_t FD_BUFFER_SIZE_BYTE_32B = 32;

struct FdRunInfo {
    bool coreEnable = false;
    int64_t bn2Idx = 0;
    int64_t mIdx = 0;
    int64_t workspaceIdx = 0;
    int64_t workspaceNum = 0;
    int64_t mStartIdx = 0;
    int64_t mNum = 0;
};

template <typename BufferType>
struct FdBuffers {
    BufferType accumOut;
    BufferType blockMax;
    BufferType blockSum;
    BufferType lseExp;
    BufferType partialO;
};

template <typename T, int64_t D_ALIGN, typename PipeType, typename BufferType>
__aicore__ inline void InitFDBuffers(const FdRunInfo &fdRunInfo, PipeType *tPipe, FdBuffers<BufferType> &buffers)
{
    tPipe->Reset();
    int64_t maxSumTotal = static_cast<uint32_t>(fdRunInfo.workspaceNum) * FD_REDUCE_CHUNK_ROWS *
        FD_BROADCAST_ELEMS_PER_ROW * sizeof(float);
    int64_t lseExpSize = FD_REDUCE_CHUNK_ROWS * FD_BROADCAST_ELEMS_PER_ROW * sizeof(float);
    int64_t accumOutSize = static_cast<uint32_t>(fdRunInfo.mNum) * D_ALIGN * sizeof(T);
    int64_t partialOSize = FD_REDUCE_CHUNK_ROWS * D_ALIGN * sizeof(T);
    tPipe->InitBuffer(buffers.accumOut, accumOutSize);
    tPipe->InitBuffer(buffers.blockMax, maxSumTotal);
    tPipe->InitBuffer(buffers.blockSum, maxSumTotal);
    tPipe->InitBuffer(buffers.lseExp, lseExpSize);
    tPipe->InitBuffer(buffers.partialO, partialOSize);
}

// The three regions are contiguous: partial O, max, then sum.
// slotCount = maxSplits * physicalCoreSlots (already includes maxSplits).
// broadcastElems: each max/sum row is stored as broadcastElems identical floats (currently 8).
// chunkRows: FD reduction chunk width (currently 16).
struct S2SplitFdStagingLayout {
    int64_t stagingM;
    int64_t dAlign;
    int64_t slotCount;
    int64_t broadcastElems;
    int64_t chunkRows;

    __aicore__ inline int64_t StagingAttenOutElems() const
    {
        return stagingM * dAlign;
    }

    __aicore__ inline int64_t StagingMaxSumBytes() const
    {
        return stagingM * broadcastElems * sizeof(float);
    }

    __aicore__ inline __gm__ uint8_t *AttenOutRegion(__gm__ uint8_t *base) const
    {
        return base;
    }

    __aicore__ inline __gm__ uint8_t *MaxRegion(__gm__ uint8_t *base) const
    {
        return AttenOutRegion(base) + slotCount * StagingAttenOutElems() * sizeof(float);
    }

    __aicore__ inline __gm__ uint8_t *SumRegion(__gm__ uint8_t *base) const
    {
        return MaxRegion(base) + slotCount * StagingMaxSumBytes();
    }
};

// Stage Vec1 max/sum to GM staging.
// tmpUb must hold at least 2 * stagingM * broadcastElems floats (max + sum broadcast blocks).
__aicore__ inline void StageVec1Lse(const S2SplitFdStagingLayout &layout,
    __gm__ uint8_t *stagingBase, int64_t workspaceIdx, int64_t stagingMOffset,
    int64_t validRows, LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
    LocalTensor<float> &tmpUb, uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    __gm__ uint8_t *maxRegion = layout.MaxRegion(stagingBase);
    __gm__ uint8_t *sumRegion = layout.SumRegion(stagingBase);
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)maxRegion);
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)sumRegion);
    LocalTensor<float> tmpMaxBlockUb = tmpUb;
    LocalTensor<float> tmpSumBlockUb = tmpUb[1024 / sizeof(float)];
    int64_t mSizeAlign8 = (validRows + layout.broadcastElems - 1) /
        layout.broadcastElems * layout.broadcastElems;
    int64_t brcbRepeat = mSizeAlign8 / layout.broadcastElems;
    Brcb(tmpMaxBlockUb, maxUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});
    Brcb(tmpSumBlockUb, sumUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
    int64_t maxSumBytes = layout.StagingMaxSumBytes();
    int64_t floatOffset = workspaceIdx * (maxSumBytes / sizeof(float)) +
        stagingMOffset * layout.broadcastElems;
    DataCopyExtParams lseOutParams{static_cast<uint16_t>(validRows),
        static_cast<uint32_t>(layout.broadcastElems * sizeof(float)), 0, 0, 0};
    DataCopyPad(sumGm[floatOffset], tmpSumBlockUb, lseOutParams);
    DataCopyPad(maxGm[floatOffset], tmpMaxBlockUb, lseOutParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
}

// Stage Vec2 normalized partial O to GM staging.
// vec2ResUb must contain FP32 partial O after final division.
// stagingOut points to the base of the atten-out staging region; workspaceIdx offset is folded into the element offset.
template <typename T>
__aicore__ inline void StageVec2PartialO(const S2SplitFdStagingLayout &layout,
    GlobalTensor<float> &stagingOut, int64_t workspaceIdx, int64_t stagingMOffset,
    int64_t validRows, int64_t dValid, LocalTensor<T> &vec2ResUb,
    uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    int64_t offset = workspaceIdx * layout.StagingAttenOutElems() +
        stagingMOffset * dValid;
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
    DataCopyExtParams outParams;
    outParams.blockLen = dValid * sizeof(float);
    outParams.srcStride = static_cast<uint16_t>((layout.dAlign - dValid) >> 3);
    outParams.dstStride = 0;
    outParams.blockCount = validRows;
    DataCopyPad(stagingOut[offset], vec2ResUb, outParams);
}

// FD chunk reduction: read all splits from staging, compute weights, reduce partial O.
// D_ALIGN is a compile-time constant (e.g. 512) used by ReduceFinalRes_const_VF.
// workspaceNum must not exceed maxSplits (currently 2).
template <typename T, int64_t D_ALIGN>
__aicore__ inline void Reduce(const S2SplitFdStagingLayout &layout,
    __gm__ uint8_t *stagingBase, int64_t workspaceIdx, int64_t workspaceNum,
    int64_t fdMOffset, int64_t mNum, int64_t dValid,
    LocalTensor<T> &accumulatedO, LocalTensor<float> &lseExpUb,
    LocalTensor<float> &blockMaxUb, LocalTensor<float> &blockSumUb,
    LocalTensor<T> &partialOFp32,
    uint8_t vToMte2Id0, uint8_t vToMte2Id1, uint8_t mte2ToVId)
{
    constexpr int64_t OUTPUT_ELEMS_PER_BLOCK = FD_BUFFER_SIZE_BYTE_32B / sizeof(float);
    int64_t attenOutElems = layout.StagingAttenOutElems();
    int64_t maxSumBytes = layout.StagingMaxSumBytes();
    __gm__ uint8_t *maxRegion = layout.MaxRegion(stagingBase);
    __gm__ uint8_t *sumRegion = layout.SumRegion(stagingBase);

    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)(maxRegion + workspaceIdx * maxSumBytes));
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)(sumRegion + workspaceIdx * maxSumBytes));
    int64_t splitStride = maxSumBytes / sizeof(float);
    GlobalTensor<float> stagingOutGm;
    stagingOutGm.SetGlobalBuffer((__gm__ float *)(layout.AttenOutRegion(stagingBase) +
        workspaceIdx * attenOutElems * sizeof(float)));
    int64_t outSplitStride = attenOutElems;
    int64_t mChunks = (mNum + layout.chunkRows - 1) / layout.chunkRows;
    int64_t startRow = 0;
    for (int64_t chunkIdx = 0; chunkIdx < mChunks; chunkIdx++) {
        int64_t dealRowCount = layout.chunkRows;
        if (startRow + dealRowCount > mNum) {
            dealRowCount = mNum - startRow;
        }
        WaitFlag<HardEvent::V_MTE2>(vToMte2Id0);
        int64_t dealRowsElems = dealRowCount * layout.broadcastElems;
        int64_t srcOffset = (fdMOffset + startRow) * layout.broadcastElems;
        int64_t dstOffset = 0;
        for (int64_t splitIdx = 0; splitIdx < workspaceNum; splitIdx++) {
            DataCopy(blockMaxUb[dstOffset], maxGm[srcOffset], dealRowsElems);
            DataCopy(blockSumUb[dstOffset], sumGm[srcOffset], dealRowsElems);
            srcOffset += splitStride;
            dstOffset += dealRowsElems;
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVId);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVId);

        LocalTensor<T> sinkUb;
        FaVectorApi::ComputeScaleValue_VF<T, T>(sinkUb, blockMaxUb, blockSumUb, lseExpUb,
                                   dealRowCount, workspaceNum, false, false);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> chunkAccumO = accumulatedO[startRow * D_ALIGN];
        int64_t outSrcOffset = (fdMOffset + startRow) * dValid;
        for (int64_t splitIdx = 0; splitIdx < workspaceNum; splitIdx++) {
            DataCopyExtParams inParams;
            inParams.blockLen = dValid * sizeof(float);
            inParams.srcStride = 0;
            inParams.dstStride = static_cast<uint16_t>((D_ALIGN - dValid) / OUTPUT_ELEMS_PER_BLOCK);
            inParams.blockCount = dealRowCount;
            DataCopyPadExtParams<float> padParams{true, 0,
                static_cast<uint8_t>((D_ALIGN - dValid) % OUTPUT_ELEMS_PER_BLOCK), 0};
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id1);
            DataCopyPad(partialOFp32, stagingOutGm[outSrcOffset], inParams, padParams);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId);
            FaVectorApi::ReduceFinalRes_const_VF<T, D_ALIGN>(chunkAccumO, blockSumUb, partialOFp32,
                                                         dealRowCount, splitIdx);
            SetFlag<HardEvent::V_MTE2>(vToMte2Id1);
            outSrcOffset += outSplitStride;
        }

        SetFlag<HardEvent::V_MTE2>(vToMte2Id0);
        startRow += layout.chunkRows;
    }
}

} // namespace AttentionCommon

#endif  // SPARSE_FLASH_MLA_FLASH_DECODE_H
