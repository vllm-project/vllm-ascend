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
 * FD_MAX_S2_SPLIT_NUM is the ordinary per-task staging limit. Batch-consistency paths may use a different
 * number of staging slots; their host allocation and metadata must agree with workspaceNum.
 */
#ifndef SPARSE_FLASH_MLA_FLASH_DECODE_H
#define SPARSE_FLASH_MLA_FLASH_DECODE_H

#include <stdint.h>

#if __has_include("../../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h")
#include "../../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#elif __has_include("../../../common/arch35/vf/vf_flash_decode_arch35.h")
#include "../../../common/arch35/vf/vf_flash_decode_arch35.h"
#endif
#include "static_buffer.h"

namespace AttentionCommon {

constexpr int64_t FD_MAX_S2_SPLIT_NUM = 2U;
constexpr int64_t FD_INCREMENTAL_MERGE_INPUT_NUM = 2U;
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

// 静态 tensor 版本的 FD buffer 初始化：不调用 tPipe->Reset()，从 ubBaseAddr 顺序排布。
// FD 在主流程 SyncAll() 之后独立执行，可复用主流程 UB 地址空间。
template <typename T, int64_t D_ALIGN>
__aicore__ inline void InitFDBuffersStatic(const FdRunInfo &fdRunInfo, uint32_t ubBaseAddr,
                                           FdBuffers<fa_base_matmul::StaticBuffer<uint8_t>> &buffers)
{
    uint32_t ubAddr = ubBaseAddr;
    int64_t maxSumTotal = static_cast<uint32_t>(fdRunInfo.workspaceNum) * FD_REDUCE_CHUNK_ROWS *
                          FD_BROADCAST_ELEMS_PER_ROW * sizeof(float);
    int64_t lseExpSize = FD_REDUCE_CHUNK_ROWS * FD_BROADCAST_ELEMS_PER_ROW * sizeof(float);
    int64_t accumOutSize = static_cast<uint32_t>(fdRunInfo.mNum) * D_ALIGN * sizeof(T);
    int64_t partialOSize = FD_REDUCE_CHUNK_ROWS * D_ALIGN * sizeof(T);

    buffers.accumOut = {LocalTensor<uint8_t>(TPosition::VECIN, ubAddr, accumOutSize), 0};
    ubAddr += accumOutSize;
    buffers.blockMax = {LocalTensor<uint8_t>(TPosition::VECIN, ubAddr, maxSumTotal), 0};
    ubAddr += maxSumTotal;
    buffers.blockSum = {LocalTensor<uint8_t>(TPosition::VECIN, ubAddr, maxSumTotal), 0};
    ubAddr += maxSumTotal;
    buffers.lseExp = {LocalTensor<uint8_t>(TPosition::VECIN, ubAddr, lseExpSize), 0};
    ubAddr += lseExpSize;
    buffers.partialO = {LocalTensor<uint8_t>(TPosition::VECIN, ubAddr, partialOSize), 0};
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

// Stage max/sum that are already stored as one broadcast block per row.
__aicore__ inline void StageBroadcastMaxSum(const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase,
                                            int64_t workspaceIdx, int64_t stagingMOffset, int64_t validRows,
                                            LocalTensor<float> &maxBroadcastUb, LocalTensor<float> &sumBroadcastUb,
                                            uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    __gm__ uint8_t *maxRegion = layout.MaxRegion(stagingBase);
    __gm__ uint8_t *sumRegion = layout.SumRegion(stagingBase);
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)maxRegion);
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)sumRegion);
    int64_t maxSumBytes = layout.StagingMaxSumBytes();
    int64_t floatOffset = workspaceIdx * (maxSumBytes / sizeof(float)) + stagingMOffset * layout.broadcastElems;
    DataCopyExtParams copyParams{static_cast<uint16_t>(validRows),
                                 static_cast<uint32_t>(layout.broadcastElems * sizeof(float)), 0, 0, 0};
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
    DataCopyPad(sumGm[floatOffset], sumBroadcastUb, copyParams);
    DataCopyPad(maxGm[floatOffset], maxBroadcastUb, copyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
}

// Stage Vec1 max/sum to GM staging.
// tmpUb must hold at least 2 * stagingM * broadcastElems floats (max + sum broadcast blocks).
__aicore__ inline void StageVec1Lse(const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase,
                                    int64_t workspaceIdx, int64_t stagingMOffset, int64_t validRows,
                                    LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, LocalTensor<float> &tmpUb,
                                    uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    LocalTensor<float> tmpMaxBlockUb = tmpUb;
    LocalTensor<float> tmpSumBlockUb = tmpUb[1024 / sizeof(float)];
    int64_t mSizeAlign8 = (validRows + layout.broadcastElems - 1) / layout.broadcastElems * layout.broadcastElems;
    int64_t brcbRepeat = mSizeAlign8 / layout.broadcastElems;
    Brcb(tmpMaxBlockUb, maxUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});
    Brcb(tmpSumBlockUb, sumUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});
    StageBroadcastMaxSum(layout, stagingBase, workspaceIdx, stagingMOffset, validRows, tmpMaxBlockUb, tmpSumBlockUb,
                         vToMte3Id, mte3ToVId);
}

// Stage Vec2 normalized partial O to GM staging.
// vec2ResUb must contain FP32 partial O after final division.
// stagingOut points to the base of the atten-out staging region; workspaceIdx offset is folded into the element offset.
template <typename T>
__aicore__ inline void StageVec2PartialO(const S2SplitFdStagingLayout &layout, GlobalTensor<float> &stagingOut,
                                         int64_t workspaceIdx, int64_t stagingMOffset, int64_t validRows,
                                         int64_t dValid, LocalTensor<T> &vec2ResUb, uint8_t vToMte3Id,
                                         uint8_t mte3ToVId)
{
    int64_t offset = workspaceIdx * layout.StagingAttenOutElems() + stagingMOffset * dValid;
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
    DataCopyExtParams outParams;
    outParams.blockLen = dValid * sizeof(float);
    outParams.srcStride = static_cast<uint16_t>((layout.dAlign - dValid) >> 3);
    outParams.dstStride = 0;
    outParams.blockCount = validRows;
    DataCopyPad(stagingOut[offset], vec2ResUb, outParams);
}

// Stage normalized partial O and wait for MTE3 completion before the source UB can be reused.
template <typename T>
__aicore__ inline void StageVec2PartialOAndWait(const S2SplitFdStagingLayout &layout, GlobalTensor<float> &stagingOut,
                                                int64_t workspaceIdx, int64_t stagingMOffset, int64_t validRows,
                                                int64_t dValid, LocalTensor<T> &vec2ResUb, uint8_t vToMte3Id,
                                                uint8_t mte3ToVId)
{
    StageVec2PartialO<T>(layout, stagingOut, workspaceIdx, stagingMOffset, validRows, dValid, vec2ResUb, vToMte3Id,
                         mte3ToVId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
}

// FD chunk reduction: read all splits from staging, compute weights, reduce partial O.
// D_ALIGN is a compile-time constant (e.g. 512) used by ReduceFinalRes_const_VF.
// workspaceNum must match the number of contiguous slots allocated by the host and emitted by metadata.
template <typename T, int64_t D_ALIGN>
__aicore__ inline void ReduceWithLse(const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase,
                                     int64_t workspaceIdx, int64_t workspaceNum, int64_t fdMOffset, int64_t mNum,
                                     int64_t dValid, LocalTensor<T> &accumulatedO, LocalTensor<float> &lseExpUb,
                                     LocalTensor<float> &blockMaxUb, LocalTensor<float> &blockSumUb,
                                     LocalTensor<T> &partialOFp32, bool softmaxLseFlag, GlobalTensor<T> &softmaxLseGm,
                                     int64_t softmaxLseOffset, uint8_t vToMte2Id0, uint8_t vToMte2Id1,
                                     uint8_t mte2ToVId, uint8_t vToMte3LseOutId, uint8_t mte3ToVLseOutId)
{
    constexpr int64_t outputElemsPerBlock = FD_BUFFER_SIZE_BYTE_32B / sizeof(float);
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
    stagingOutGm.SetGlobalBuffer(
        (__gm__ float *)(layout.AttenOutRegion(stagingBase) + workspaceIdx * attenOutElems * sizeof(float)));
    int64_t outSplitStride = attenOutElems;
    LocalTensor<T> sinkUb;
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

        FaVectorApi::ComputeScaleValue_VF<T, T>(sinkUb, blockMaxUb, blockSumUb, lseExpUb, dealRowCount, workspaceNum,
                                                softmaxLseFlag, false);
        PipeBarrier<PIPE_V>();
        if (softmaxLseFlag) {
            DataCopyExtParams lseParams;
            lseParams.blockCount = static_cast<uint16_t>(dealRowCount);
            lseParams.blockLen = sizeof(float);
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
            SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            DataCopyPad(softmaxLseGm[softmaxLseOffset + startRow], lseExpUb, lseParams);
            SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        }

        LocalTensor<T> chunkAccumO = accumulatedO[startRow * D_ALIGN];
        int64_t outSrcOffset = (fdMOffset + startRow) * dValid;
        for (int64_t splitIdx = 0; splitIdx < workspaceNum; splitIdx++) {
            DataCopyExtParams inParams;
            inParams.blockLen = dValid * sizeof(float);
            inParams.srcStride = 0;
            inParams.dstStride = static_cast<uint16_t>((D_ALIGN - dValid) / outputElemsPerBlock);
            inParams.blockCount = dealRowCount;
            DataCopyPadExtParams<float> padParams{true, 0,
                                                  static_cast<uint8_t>((D_ALIGN - dValid) % outputElemsPerBlock), 0};
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id1);
            DataCopyPad(partialOFp32, stagingOutGm[outSrcOffset], inParams, padParams);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId);
            FaVectorApi::ReduceFinalRes_const_VF<T, D_ALIGN>(chunkAccumO, blockSumUb, partialOFp32, dealRowCount,
                                                             splitIdx);
            SetFlag<HardEvent::V_MTE2>(vToMte2Id1);
            outSrcOffset += outSplitStride;
        }

        SetFlag<HardEvent::V_MTE2>(vToMte2Id0);
        startRow += layout.chunkRows;
    }
}

template <typename T, int64_t D_ALIGN>
__aicore__ inline void Reduce(const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase, int64_t workspaceIdx,
                              int64_t workspaceNum, int64_t fdMOffset, int64_t mNum, int64_t dValid,
                              LocalTensor<T> &accumulatedO, LocalTensor<float> &lseExpUb,
                              LocalTensor<float> &blockMaxUb, LocalTensor<float> &blockSumUb,
                              LocalTensor<T> &partialOFp32, uint8_t vToMte2Id0, uint8_t vToMte2Id1, uint8_t mte2ToVId)
{
    GlobalTensor<T> unusedSoftmaxLseGm;
    ReduceWithLse<T, D_ALIGN>(layout, stagingBase, workspaceIdx, workspaceNum, fdMOffset, mNum, dValid, accumulatedO,
                              lseExpUb, blockMaxUb, blockSumUb, partialOFp32, false, unusedSoftmaxLseGm, 0, vToMte2Id0,
                              vToMte2Id1, mte2ToVId, 0, 0);
}

// Merge two normalized partial results. mergedO may alias leftO; rightO must remain
// readable until the second reduction finishes. The result is (LSE, 1, normalized O).
template <typename T, int64_t D_ALIGN>
__aicore__ inline void MergeTwoInputsWithLse(const S2SplitFdStagingLayout &layout, int64_t mNum,
                                             LocalTensor<T> &mergedO, LocalTensor<T> &leftO, LocalTensor<T> &rightO,
                                             LocalTensor<float> &mergedLseUb, LocalTensor<float> &mergedSumUb,
                                             LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                             LocalTensor<T> &sinkUb)
{
    FaVectorApi::ComputeScaleValue_VF<T, T>(sinkUb, maxUb, sumUb, mergedLseUb, mNum, FD_INCREMENTAL_MERGE_INPUT_NUM,
                                            true, false);
    FaVectorApi::ReduceFinalRes_const_VF<T, D_ALIGN>(mergedO, sumUb, leftO, mNum, 0);
    PipeBarrier<PIPE_V>();
    FaVectorApi::ReduceFinalRes_const_VF<T, D_ALIGN>(mergedO, sumUb, rightO, mNum, 1);
    PipeBarrier<PIPE_V>();
    Duplicate(mergedSumUb, static_cast<float>(1.0), mNum * layout.broadcastElems);
    PipeBarrier<PIPE_V>();
}

// Deterministic FD reduction: merge contiguous staging slots in a fixed left fold.
// Each step uses the same two-input UB reduction as the intra-core batch-consistency path:
//   state = ((slot0 merge slot1) merge slot2) ...
// accumulatedO holds the left-fold state; partialOUb streams one new input at a time.
template <typename T, int64_t D_ALIGN>
__aicore__ inline void ReducePairwiseWithLse(const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase,
                                             int64_t workspaceIdx, int64_t workspaceNum, int64_t fdMOffset,
                                             int64_t mNum, int64_t dValid, LocalTensor<T> &accumulatedO,
                                             LocalTensor<float> &lseExpUb, LocalTensor<float> &blockMaxUb,
                                             LocalTensor<float> &blockSumUb, LocalTensor<T> &partialOUb,
                                             bool softmaxLseFlag, GlobalTensor<T> &softmaxLseGm,
                                             int64_t softmaxLseOffset, uint8_t vToMte2Id0, uint8_t vToMte2Id1,
                                             uint8_t mte2ToVId, uint8_t vToMte3LseOutId, uint8_t mte3ToVLseOutId)
{
    if (workspaceNum <= FD_INCREMENTAL_MERGE_INPUT_NUM) {
        ReduceWithLse<T, D_ALIGN>(layout, stagingBase, workspaceIdx, workspaceNum, fdMOffset, mNum, dValid,
                                  accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOUb, softmaxLseFlag,
                                  softmaxLseGm, softmaxLseOffset, vToMte2Id0, vToMte2Id1, mte2ToVId, vToMte3LseOutId,
                                  mte3ToVLseOutId);
        return;
    }

    constexpr int64_t outputElemsPerBlock = FD_BUFFER_SIZE_BYTE_32B / sizeof(float);
    int64_t attenOutElems = layout.StagingAttenOutElems();
    int64_t maxSumBytes = layout.StagingMaxSumBytes();
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)(layout.MaxRegion(stagingBase) + workspaceIdx * maxSumBytes));
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)(layout.SumRegion(stagingBase) + workspaceIdx * maxSumBytes));
    GlobalTensor<float> stagingOutGm;
    stagingOutGm.SetGlobalBuffer(
        (__gm__ float *)(layout.AttenOutRegion(stagingBase) + workspaceIdx * attenOutElems * sizeof(float)));
    int64_t splitStride = maxSumBytes / sizeof(float);
    int64_t outSplitStride = attenOutElems;

    LocalTensor<T> sinkUb;
    int64_t mChunks = (mNum + layout.chunkRows - 1) / layout.chunkRows;
    int64_t startRow = 0;
    for (int64_t chunkIdx = 0; chunkIdx < mChunks; chunkIdx++) {
        int64_t dealRowCount = layout.chunkRows;
        if (startRow + dealRowCount > mNum) {
            dealRowCount = mNum - startRow;
        }
        int64_t dealRowsElems = dealRowCount * layout.broadcastElems;
        int64_t maxSumRowOffset = (fdMOffset + startRow) * layout.broadcastElems;
        int64_t outRowOffset = (fdMOffset + startRow) * dValid;
        DataCopyExtParams inParams;
        inParams.blockLen = dValid * sizeof(float);
        inParams.srcStride = 0;
        inParams.dstStride = static_cast<uint16_t>((D_ALIGN - dValid) / outputElemsPerBlock);
        inParams.blockCount = dealRowCount;
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>((D_ALIGN - dValid) % outputElemsPerBlock),
                                              0};

        // Seed the fold with slot0 and slot1.
        if (softmaxLseFlag) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        }
        WaitFlag<HardEvent::V_MTE2>(vToMte2Id0);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Id1);
        DataCopy(blockMaxUb, maxGm[maxSumRowOffset], dealRowsElems);
        DataCopy(blockSumUb, sumGm[maxSumRowOffset], dealRowsElems);
        DataCopy(blockMaxUb[dealRowsElems], maxGm[maxSumRowOffset + splitStride], dealRowsElems);
        DataCopy(blockSumUb[dealRowsElems], sumGm[maxSumRowOffset + splitStride], dealRowsElems);
        LocalTensor<T> chunkAccumO = accumulatedO[startRow * D_ALIGN];
        DataCopyPad(chunkAccumO, stagingOutGm[outRowOffset], inParams, padParams);
        DataCopyPad(partialOUb, stagingOutGm[outRowOffset + outSplitStride], inParams, padParams);
        SetFlag<HardEvent::MTE2_V>(mte2ToVId);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVId);

        MergeTwoInputsWithLse<T, D_ALIGN>(layout, dealRowCount, chunkAccumO, chunkAccumO, partialOUb, blockMaxUb,
                                          blockSumUb, blockMaxUb, blockSumUb, sinkUb);

        // Convert the accumulated state to (LSE, 1, normalized O), then merge one slot at a time.
        for (int64_t splitIdx = FD_INCREMENTAL_MERGE_INPUT_NUM; splitIdx < workspaceNum; splitIdx++) {
            SetFlag<HardEvent::V_MTE2>(vToMte2Id0);
            SetFlag<HardEvent::V_MTE2>(vToMte2Id1);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id0);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id1);

            int64_t maxSumOffset = maxSumRowOffset + splitIdx * splitStride;
            DataCopy(blockMaxUb[dealRowsElems], maxGm[maxSumOffset], dealRowsElems);
            DataCopy(blockSumUb[dealRowsElems], sumGm[maxSumOffset], dealRowsElems);
            DataCopyPad(partialOUb, stagingOutGm[outRowOffset + splitIdx * outSplitStride], inParams, padParams);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId);
            MergeTwoInputsWithLse<T, D_ALIGN>(layout, dealRowCount, chunkAccumO, chunkAccumO, partialOUb, blockMaxUb,
                                              blockSumUb, blockMaxUb, blockSumUb, sinkUb);
        }

        if (softmaxLseFlag) {
            DataCopyExtParams lseParams;
            lseParams.blockCount = static_cast<uint16_t>(dealRowCount);
            lseParams.blockLen = sizeof(float);
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            DataCopyPad(softmaxLseGm[softmaxLseOffset + startRow], blockMaxUb, lseParams);
            SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        }

        SetFlag<HardEvent::V_MTE2>(vToMte2Id0);
        SetFlag<HardEvent::V_MTE2>(vToMte2Id1);
        startRow += layout.chunkRows;
    }
}

// Merge one previously staged result with one current normalized result.
// dealRowCount must not exceed layout.chunkRows.
// currentMaxUb/currentSumUb contain one scalar per row; currentPartialO uses D_ALIGN elements per row.
// blockMaxUb/blockSumUb layout: [previous, current][dealRowCount][broadcastElems].
// partialOTmpUb holds the previous normalized O and then the merged O.
// The merged state is returned as normalized partial O plus (LSE, 1), so it can be staged and merged again.
// Both V_MTE2 events must be set before the first call. This function restores them before returning.
template <typename T, int64_t D_ALIGN>
__aicore__ inline void MergeStagedAndCurrentChunk(
    const S2SplitFdStagingLayout &layout, __gm__ uint8_t *stagingBase, int64_t workspaceIdx, int64_t stagingMOffset,
    int64_t dealRowCount, int64_t dValid, LocalTensor<float> &currentMaxUb, LocalTensor<float> &currentSumUb,
    LocalTensor<T> &currentPartialO, LocalTensor<float> &blockMaxUb, LocalTensor<float> &blockSumUb,
    LocalTensor<T> &partialOTmpUb, LocalTensor<float> &mergedLseBroadcastUb, LocalTensor<float> &mergedSumBroadcastUb,
    LocalTensor<T> &sinkUb, uint8_t maxSumVToMte2Id, uint8_t partialOVToMte2Id, uint8_t mte2ToVId)
{
    constexpr int64_t outputElemsPerBlock = FD_BUFFER_SIZE_BYTE_32B / sizeof(float);
    int64_t dealRowsElems = dealRowCount * layout.broadcastElems;
    int64_t brcbRepeat = (dealRowCount + layout.broadcastElems - 1) / layout.broadcastElems;
    Brcb(blockMaxUb[dealRowsElems], currentMaxUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});
    Brcb(blockSumUb[dealRowsElems], currentSumUb, brcbRepeat, {1, static_cast<uint16_t>(layout.broadcastElems)});

    int64_t maxSumBytes = layout.StagingMaxSumBytes();
    GlobalTensor<float> previousMaxGm;
    previousMaxGm.SetGlobalBuffer((__gm__ float *)(layout.MaxRegion(stagingBase) + workspaceIdx * maxSumBytes));
    GlobalTensor<float> previousSumGm;
    previousSumGm.SetGlobalBuffer((__gm__ float *)(layout.SumRegion(stagingBase) + workspaceIdx * maxSumBytes));
    int64_t maxSumOffset = stagingMOffset * layout.broadcastElems;
    WaitFlag<HardEvent::V_MTE2>(maxSumVToMte2Id);
    DataCopy(blockMaxUb, previousMaxGm[maxSumOffset], dealRowsElems);
    DataCopy(blockSumUb, previousSumGm[maxSumOffset], dealRowsElems);
    SetFlag<HardEvent::MTE2_V>(mte2ToVId);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVId);

    GlobalTensor<float> previousPartialOGm;
    previousPartialOGm.SetGlobalBuffer((__gm__ float *)(layout.AttenOutRegion(stagingBase) +
                                                        workspaceIdx * layout.StagingAttenOutElems() * sizeof(float)));
    int64_t partialOOffset = stagingMOffset * dValid;
    DataCopyExtParams inParams;
    inParams.blockLen = static_cast<uint32_t>(dValid * sizeof(float));
    inParams.srcStride = 0;
    inParams.dstStride = static_cast<uint16_t>((D_ALIGN - dValid) / outputElemsPerBlock);
    inParams.blockCount = static_cast<uint16_t>(dealRowCount);
    DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>((D_ALIGN - dValid) % outputElemsPerBlock), 0};
    WaitFlag<HardEvent::V_MTE2>(partialOVToMte2Id);
    DataCopyPad(partialOTmpUb, previousPartialOGm[partialOOffset], inParams, padParams);
    SetFlag<HardEvent::MTE2_V>(mte2ToVId);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVId);

    DataCopy(partialOTmpUb[dealRowCount * D_ALIGN], currentPartialO, dealRowCount * D_ALIGN);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> currentPartialOTmp = partialOTmpUb[dealRowCount * D_ALIGN];
    MergeTwoInputsWithLse<T, D_ALIGN>(layout, dealRowCount, currentPartialO, partialOTmpUb, currentPartialOTmp,
                                      mergedLseBroadcastUb, mergedSumBroadcastUb, blockMaxUb, blockSumUb, sinkUb);
    SetFlag<HardEvent::V_MTE2>(partialOVToMte2Id);
    SetFlag<HardEvent::V_MTE2>(maxSumVToMte2Id);
}

} // namespace AttentionCommon

#endif // SPARSE_FLASH_MLA_FLASH_DECODE_H
