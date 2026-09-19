/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d.h
 * \brief CausalConv1d kernel core class and shared device-side logic.
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/causal_conv1d_regbase.h"
#endif

namespace NsCausalConv1d {

using namespace AscendC;
using namespace NsCausalConv1dCommon;

#define CAUSAL_CONV1D_TEMPLATE_ARGS typename T, uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey
#define CAUSAL_CONV1D_CLASS CausalConv1d<T, runModeKey, widthKey, fnPlanKey>

enum SeqTaskWindowMode : int32_t {
    SEQ_TASK_WINDOW_MODE_VARLEN = 0,
    SEQ_TASK_WINDOW_MODE_BATCH = 1,
    SEQ_TASK_WINDOW_MODE_DECODE2D = 2,
};

inline constexpr int32_t INIT_STATE_SYNCALL_NEED_SIZE = 8;
inline constexpr int32_t INIT_STATE_SYNCALL_MAX_BLOCKS = 64;

struct SeqTaskWindow {
    bool valid = false;
    int32_t start = 0;
    int32_t len = 0;
};

__aicore__ inline int32_t GetSeqTaskWindowMode(int32_t inputMode)
{
    if (inputMode == 0) {
        return SEQ_TASK_WINDOW_MODE_VARLEN;
    }
    if (inputMode == 2) {
        return SEQ_TASK_WINDOW_MODE_DECODE2D;
    }
    return SEQ_TASK_WINDOW_MODE_BATCH;
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowVarlen(int32_t startVal, int32_t endVal)
{
    SeqTaskWindow window;
    window.start = startVal;
    window.len = endVal - startVal;
    window.valid = (window.len > 0);
    return window;
}

__aicore__ inline int32_t RetreatRingSlot(int32_t slot, int32_t delta)
{
    int32_t prev = slot - delta;
    return (prev >= 0) ? prev : (prev + RING_SLOTS);
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowBatch(int32_t seq, int32_t seqLen)
{
    SeqTaskWindow window;
    window.start = seq * seqLen;
    window.len = seqLen;
    window.valid = (window.len > 0);
    return window;
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowDecode2D(int32_t seq)
{
    SeqTaskWindow window;
    window.valid = true;
    window.start = seq;
    window.len = 1;
    return window;
}

__aicore__ inline constexpr int32_t DecodeWidthTplKey(uint32_t widthKey)
{
    switch (widthKey) {
        case CAUSAL_CONV1D_TPL_WIDTH_2:
            return 2;
        case CAUSAL_CONV1D_TPL_WIDTH_3:
            return 3;
        case CAUSAL_CONV1D_TPL_WIDTH_4:
            return 4;
        default:
            return 0;
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
class CausalConv1d {
public:
    __aicore__ inline CausalConv1d() = default;

protected:
    static constexpr bool kIsUpdateMode = (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE);
    static constexpr int32_t kTemplateWidth = DecodeWidthTplKey(widthKey);
    static constexpr bool kHasCompileTimeWidth =
        (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) && (kTemplateWidth >= 2) && (kTemplateWidth <= MAX_WIDTH);
    static constexpr FnExecutionPlan kFnExecutionPlan = static_cast<FnExecutionPlan>(fnPlanKey);

    __aicore__ inline void ResetRuntimeState(const CausalConv1dTilingData *tilingData);
    __aicore__ inline void InitSharedBuffersAndEvents();
    __aicore__ inline void LoadWeightAndBias(int32_t channelStart, int32_t baseDim);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                    int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void InitRingSeqSplit(int32_t seq, int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                            int32_t tileStart, int32_t tileLen, int32_t channelStart, int32_t baseDim,
                                            int32_t dim);
    __aicore__ inline void PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize,
                                                         int32_t ownerIndex, int32_t ownerCount);
    __aicore__ inline void RestoreFnLocalPartials(int32_t baseDim);
    __aicore__ inline void ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim);
    __aicore__ inline void AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim);
    __aicore__ inline void CopyFnRollingInputBatch(int32_t start, int32_t firstToken, int32_t tokenCount,
                                                   int32_t dstSlot, int32_t channelStart, int32_t baseDim,
                                                   int32_t dim);
    __aicore__ inline void ProcessFnRollingToken(int32_t start, int32_t len, int32_t t, int32_t slotCurr,
                                                 int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim,
                                           int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void WriteFnOutput(LocalTensor<T> &outSlotT, int32_t token, int32_t channelStart,
                                         int32_t baseDim, int32_t dim);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                          int32_t dim);
    __aicore__ inline void WriteBackFnState(int32_t seq, int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                            int32_t seqLen, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void WriteBackStateSpec(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                              int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void DrainTaskMte3();
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();
    __aicore__ inline int32_t FindVarlenSeqByToken(int32_t tokenIdx) const;
    __aicore__ inline bool ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq, int32_t &endSeq) const;
    __aicore__ inline bool ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen, int32_t &start,
                                                int32_t &len) const;
    template <int32_t kWindowMode>
    __aicore__ inline bool ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start, int32_t &len) const;
    __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices, int32_t &cacheIdx) const;
    __aicore__ inline bool ResolveSeqHasInit(int32_t seq, bool hasInitialState) const;
    __aicore__ inline int64_t ReadQueryStartLoc(int32_t index) const;
    __aicore__ inline int64_t ReadCacheIndex(int32_t index) const;
    __aicore__ inline int64_t ReadHasInitialState(int32_t index) const;
    __aicore__ inline int64_t ReadNumAcceptedTokens(int32_t index) const;
    __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                           int32_t seqLen, int32_t seq, int32_t cacheIdx,
                                                           bool hasInit, int32_t channelStart, int32_t baseDim,
                                                           int32_t dim);
    __aicore__ inline void ProcessDefault();
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode();
    __aicore__ inline void ProcessVarlenTokenTiled();
    __aicore__ inline void ProcessFnChunk(int32_t seq, int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                          int32_t seqLen, int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                          int32_t baseDim, int32_t dim);
    __aicore__ inline const CausalConv1dTilingData *GetTilingData() const;
    __aicore__ inline bool HasActivation() const;
    __aicore__ inline bool HasBias() const;
    __aicore__ inline bool HasConvStates() const;
    __aicore__ inline bool IsUpdateMode() const;
    __aicore__ inline bool IsFnRollingFastPathEnabled() const;
    __aicore__ inline bool HasExplicitFnTokenSeqRanges() const;
    __aicore__ inline bool IsUpdateSpecDecodingEnabled() const;

protected:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID weightBiasMte2ToVEvent_;
    TEventID stateMte2ToVEvent_;
    TEventID inputMte2ToVEvent_[RING_SLOTS];
    TEventID inputVToMte2Event_;
    TEventID rollingPairVToMte2Event_[2];
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];
    TEventID stateWritebackMte3ToVEvent_;
    TEventID stateWritebackMte3ToMte2Event_;
    TEventID stateShiftMte2ToMte3Event_;
    TEventID stateShiftVToMte3Event_;
    TEventID stateShiftMte3ToMte2Event_;
    TEventID initSnapshotMte2ToMte3Event_;
    TEventID initSnapshotMte3ToMte2Event_;
    TEventID initSyncVToMte3Event_;
    TEventID initSyncMte3ToVEvent_;
    TEventID specWritebackMte2ToMte3Event_[2];
    TEventID specWritebackMte3ToMte2Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int64_t> queryStartLocGm;
    GlobalTensor<int32_t> queryStartLocI32Gm;
    GlobalTensor<int64_t> cacheIndicesGm;
    GlobalTensor<int32_t> cacheIndicesI32Gm;
    GlobalTensor<int64_t> hasInitialStateGm;
    GlobalTensor<int32_t> hasInitialStateI32Gm;
    GlobalTensor<uint8_t> hasInitialStateBoolGm;
    GlobalTensor<int64_t> numAcceptedTokensGm;
    GlobalTensor<int32_t> numAcceptedTokensI32Gm;
    GlobalTensor<T> yGm;
    GlobalTensor<int32_t> initStateSyncGm_;
    GlobalTensor<T> initStateWorkspaceGm_;

    const CausalConv1dTilingData *tilingData_{nullptr};
};

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ResetRuntimeState(const CausalConv1dTilingData *tilingData)
{
    tilingData_ = tilingData;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::InitSharedBuffersAndEvents()
{
    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 4) * MAX_BLOCK_DIM * sizeof(float));
    AllocEvents();
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::AllocEvents()
{
    weightBiasMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    stateMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        inputMte2ToVEvent_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    }
    inputVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    rollingPairVToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    rollingPairVToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    stateShiftMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    stateShiftVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateShiftMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    initSnapshotMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    initSnapshotMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    initSyncVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    initSyncMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    specWritebackMte2ToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte2ToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte3ToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    specWritebackMte3ToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(rollingPairVToMte2Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(rollingPairVToMte2Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(initSyncVToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(initSyncMte3ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::LoadWeightAndBias(int32_t channelStart, int32_t baseDim)
{
    const int32_t dim = tilingData_->dim;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    const bool hasBias = HasBias();
    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &biasF = cl.biasF;
    LocalTensor<T> weightT;
    LocalTensor<T> biasT;

    if constexpr (!std::is_same<T, float>::value) {

        weightT = inBuf.Get<T>();
        biasT = weightT[MAX_WIDTH * MAX_BLOCK_DIM];
    }

    for (int32_t j = 0; j < jStart; ++j) {
        Duplicate(weightF[j * MAX_BLOCK_DIM], 0.0f, baseDim);
    }

    for (int32_t j = 0; j < width; ++j) {
        const int32_t jDst = jStart + j;
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + channelStart;

        if constexpr (std::is_same<T, float>::value) {
            DataCopy(weightF[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
        } else {
            DataCopy(weightT[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
        }
    }

    if (hasBias) {
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(biasF, biasGm[channelStart], baseDim);
        } else {
            DataCopy(biasT, biasGm[channelStart], baseDim);
        }
    }

    SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);

    if constexpr (!std::is_same<T, float>::value) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t jDst = jStart + j;
            Cast(weightF[jDst * MAX_BLOCK_DIM], weightT[jDst * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        }
        if (hasBias) {
            Cast(biasF, biasT, RoundMode::CAST_NONE, baseDim);
        }

        PipeBarrier<PIPE_V>();
    }

    if (!hasBias) {
        Duplicate(biasF, 0.0f, baseDim);
    }

    if constexpr (!std::is_same<T, float>::value) {

        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                     int32_t start, int32_t len, int32_t channelStart,
                                                     int32_t baseDim, int32_t dim)
{
    const int64_t convStateStride0 = tilingData_->convStateStride0;
    const int64_t convStateStride1 = tilingData_->convStateStride1;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t ringStart = MAX_WIDTH - width;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t i = 0; i < ringStart; ++i) {
        Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
    }
    if (ringStart > 0) {
        PipeBarrier<PIPE_V>();
    }

    if (hasInit) {
        for (int32_t i = 0; i < (width - 1); ++i) {
            const int32_t pos = stateTokenOffset + i;
            const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * convStateStride0 +
                                        static_cast<int64_t>(pos) * convStateStride1 + channelStart;
            DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], baseDim);
        }

        SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    } else {
        for (int32_t i = 0; i < (width - 1); ++i) {
            Duplicate(ring[(ringStart + i) * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
        }

        PipeBarrier<PIPE_V>();
    }

    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);

        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
    }

    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteFnOutput(LocalTensor<T> &outSlotT, int32_t token,
                                                          int32_t channelStart, int32_t baseDim, int32_t dim)
{
    const int32_t headNum = static_cast<int32_t>(tilingData_->headNum);
    if (headNum <= 0) {
        const int64_t outOffset = static_cast<int64_t>(token) * dim + channelStart;
        DataCopy(yGm[outOffset], outSlotT, baseDim);
        return;
    }

    const int32_t headDim = static_cast<int32_t>(tilingData_->headDim);
    const int32_t inputMode = static_cast<int32_t>(tilingData_->inputMode);
    const int32_t cuSeqlen = static_cast<int32_t>(tilingData_->cuSeqlen);
    const int32_t seqLen = static_cast<int32_t>(tilingData_->seqLen);
    int32_t srcOffset = 0;
    int32_t channel = channelStart;
    int32_t remaining = baseDim;
    while (remaining > 0) {
        const int32_t headIdx = channel / headDim;
        const int32_t innerOffset = channel - headIdx * headDim;
        const int32_t headRemaining = headDim - innerOffset;
        const int32_t copySize = (remaining < headRemaining) ? remaining : headRemaining;
        int64_t outOffset = 0;
        if (inputMode == 0) {

            outOffset = (static_cast<int64_t>(headIdx) * cuSeqlen + token) * headDim + innerOffset;
        } else {

            const int32_t batchIdx = token / seqLen;
            const int32_t seqIdx = token - batchIdx * seqLen;
            outOffset = ((static_cast<int64_t>(batchIdx) * headNum + headIdx) * seqLen + seqIdx) * headDim +
                        innerOffset;
        }
        DataCopy(yGm[outOffset], outSlotT[srcOffset], copySize);
        srcOffset += copySize;
        channel += copySize;
        remaining -= copySize;
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeq(int32_t start, int32_t len, int32_t channelStart,
                                                   int32_t baseDim, int32_t dim)
{
    if (IsFnRollingFastPathEnabled()) {
        RunSeqFnRolling(start, len, channelStart, baseDim, dim);
        return;
    }

    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &biasF = cl.biasF;
    LocalTensor<float> &accF = cl.accF;
    LocalTensor<float> &tmpF = cl.tmpF;
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasBias = HasBias();
    const bool hasActivation = HasActivation();
    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);

        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t);
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
            DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
        }

        bool accInitialized = false;
        if (hasBias) {

            Adds(accF, biasF, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();
            accInitialized = true;
        }

        for (int32_t j = jStart; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            if (!accInitialized) {
                Mul(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
                accInitialized = true;
            } else {
                MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
            }
        }

        PipeBarrier<PIPE_V>();

        if (hasActivation) {
            Silu(tmpF, accF, baseDim);
        }

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {

            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outSlotT, tmpF, baseDim);
            } else {
                DataCopy(outSlotT, accF, baseDim);
            }
        } else {
            if (hasActivation) {
                Cast(outSlotT, tmpF, RoundMode::CAST_RINT, baseDim);
            } else {
                Cast(outSlotT, accF, RoundMode::CAST_RINT, baseDim);
            }
        }

        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        WriteFnOutput(outSlotT, start + t, channelStart, baseDim, dim);
        if (t + 2 < len) {

            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        }
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::RestoreFnLocalPartials(int32_t baseDim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &state2F = cl.biasF;
    LocalTensor<float> &state1F = cl.accF;
    LocalTensor<float> &state0F = cl.tmpF;
    LocalTensor<float> &currF = cl.currF;
    LocalTensor<T> ring = inBuf.Get<T>();
    constexpr int32_t ringStart = MAX_WIDTH - kTemplateWidth;
    constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    RestoreFnLocalPartialsRegbase<T, kTemplateWidth>(ring[ringStart * MAX_BLOCK_DIM],
                                                     weightF[w0Idx * MAX_BLOCK_DIM], state0F, state1F, state2F,
                                                     baseDim, MAX_BLOCK_DIM);
#else
    if constexpr (kTemplateWidth == 2) {
        Duplicate(state2F, 0.0f, baseDim);
        Duplicate(state1F, 0.0f, baseDim);
        PipeBarrier<PIPE_V>();

        Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 3) {
        Duplicate(state2F, 0.0f, baseDim);
        PipeBarrier<PIPE_V>();

        Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();

        Cast(currF, ring[(ringStart + 1) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        MulAddDst(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 4) {
        Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();

        Cast(currF, ring[(ringStart + 1) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        MulAddDst(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();

        Cast(currF, ring[(ringStart + 2) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        PipeBarrier<PIPE_V>();
        Mul(state2F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        MulAddDst(state1F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        MulAddDst(state0F, currF, weightF[(w0Idx + 2) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    }
#endif
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &state0F = cl.tmpF;
    LocalTensor<float> &currF = cl.currF;
    LocalTensor<T> ring = inBuf.Get<T>();

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    const bool hasActivation = HasActivation();
    if (hasActivation) {
        ComputeFnRollingOutputRegbase<T, true>(ring[slotCurr * MAX_BLOCK_DIM], currF, state0F,
                                               weightF[3 * MAX_BLOCK_DIM], baseDim);
    } else {
        ComputeFnRollingOutputRegbase<T, false>(ring[slotCurr * MAX_BLOCK_DIM], currF, state0F,
                                                weightF[3 * MAX_BLOCK_DIM], baseDim);
    }
#else
    Cast(currF, ring[slotCurr * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
    PipeBarrier<PIPE_V>();
    MulAddDst(state0F, currF, weightF[3 * MAX_BLOCK_DIM], baseDim);
    PipeBarrier<PIPE_V>();

    const bool hasActivation = HasActivation();
    if (hasActivation) {

        PipeBarrier<PIPE_V>();
        Silu(currF, state0F, baseDim);
    }
#endif
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &state2F = cl.biasF;
    LocalTensor<float> &state1F = cl.accF;
    LocalTensor<float> &state0F = cl.tmpF;
    LocalTensor<float> &currF = cl.currF;
    LocalTensor<T> ring = inBuf.Get<T>();
    constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    AdvanceFnLocalPartialsRegbase<T, kTemplateWidth>(ring[slotCurr * MAX_BLOCK_DIM],
                                                     weightF[w0Idx * MAX_BLOCK_DIM], state0F, state1F, state2F,
                                                     baseDim, MAX_BLOCK_DIM);
#else
    Cast(currF, ring[slotCurr * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
    PipeBarrier<PIPE_V>();

    if constexpr (kTemplateWidth == 2) {
        Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 3) {
        Mul(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state0F, state0F, state1F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 4) {
        Mul(state0F, currF, weightF[(w0Idx + 2) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state0F, state0F, state1F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state1F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state1F, state1F, state2F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state2F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    }
#endif
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::CopyFnRollingInputBatch(int32_t start, int32_t firstToken,
                                                                   int32_t tokenCount, int32_t dstSlot,
                                                                   int32_t channelStart, int32_t baseDim, int32_t dim)
{

    const int32_t pairBuffer = dstSlot / 2;
    WaitFlag<HardEvent::V_MTE2>(rollingPairVToMte2Event_[pairBuffer]);
    LocalTensor<T> ring = inBuf.Get<T>();
    const int64_t xOffset = static_cast<int64_t>(start + firstToken) * dim + channelStart;
    const int32_t blockLen = baseDim * static_cast<int32_t>(sizeof(T)) / 32;
    const int32_t srcStride = (dim - baseDim) * static_cast<int32_t>(sizeof(T)) / 32;
    const int32_t dstStride = (MAX_BLOCK_DIM - baseDim) * static_cast<int32_t>(sizeof(T)) / 32;

    const bool canUseCompactBatch =
        tokenCount > 1 && blockLen <= 65535 && srcStride <= 65535 && dstStride <= 65535;
    if (canUseCompactBatch) {
        const DataCopyParams copyParams{static_cast<uint16_t>(tokenCount), static_cast<uint16_t>(blockLen),
                                        static_cast<uint16_t>(srcStride), static_cast<uint16_t>(dstStride)};
        DataCopy(ring[dstSlot * MAX_BLOCK_DIM], xGm[xOffset], copyParams);
    } else {
        for (int32_t i = 0; i < tokenCount; ++i) {
            const int64_t tokenOffset = static_cast<int64_t>(start + firstToken + i) * dim + channelStart;
            DataCopy(ring[(dstSlot + i) * MAX_BLOCK_DIM], xGm[tokenOffset], baseDim);
        }
    }
    SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[dstSlot]);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessFnRollingToken(int32_t start, int32_t len, int32_t t,
                                                                  int32_t slotCurr, int32_t channelStart,
                                                                  int32_t baseDim, int32_t dim)
{
    auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &state2F = cl.biasF;
    LocalTensor<float> &state1F = cl.accF;
    LocalTensor<float> &state0F = cl.tmpF;
    LocalTensor<float> &currF = cl.currF;
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasActivation = HasActivation();

    const int32_t outSlot = t & 1;
    LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
    if (t >= 2) {
        WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;
    if (hasActivation) {
        ComputeFnRollingTokenRegbase<T, kTemplateWidth, true>(
            ring[slotCurr * MAX_BLOCK_DIM], weightF[w0Idx * MAX_BLOCK_DIM], state0F, state1F, state2F, outSlotT,
            baseDim, MAX_BLOCK_DIM);
    } else {
        ComputeFnRollingTokenRegbase<T, kTemplateWidth, false>(
            ring[slotCurr * MAX_BLOCK_DIM], weightF[w0Idx * MAX_BLOCK_DIM], state0F, state1F, state2F, outSlotT,
            baseDim, MAX_BLOCK_DIM);
    }
    PipeBarrier<PIPE_V>();
#else
    ComputeFnRollingOutput(slotCurr, baseDim);
    if constexpr (IsSameType<T, float>::value) {
        if (hasActivation) {
            DataCopy(outSlotT, currF, baseDim);
        } else {
            DataCopy(outSlotT, state0F, baseDim);
        }
    } else {
        if (hasActivation) {
            Cast(outSlotT, currF, RoundMode::CAST_RINT, baseDim);
        } else {
            Cast(outSlotT, state0F, RoundMode::CAST_RINT, baseDim);
        }
    }

    AdvanceFnLocalPartials(slotCurr, baseDim);
    PipeBarrier<PIPE_V>();
#endif

    SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

    WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
    WriteFnOutput(outSlotT, start + t, channelStart, baseDim, dim);
    if (t + 2 < len) {
        SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart,
                                                            int32_t baseDim, int32_t dim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }
    if (len <= 0) {
        return;
    }

    constexpr int32_t kTokensPerCopy = 2;
    constexpr int32_t kPairSlot0 = 0;
    constexpr int32_t kPairSlot1 = 2;

    RestoreFnLocalPartials(baseDim);
    const int32_t slot0 = SlotCurr(0);
    WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);

    int32_t nextToken = 1;
    int32_t currentPairSlot = kPairSlot0;
    if (nextToken < len) {

        SetFlag<HardEvent::V_MTE2>(rollingPairVToMte2Event_[0]);
        const int32_t tokenCount = ((len - nextToken) < kTokensPerCopy) ? (len - nextToken) : kTokensPerCopy;
        CopyFnRollingInputBatch(start, nextToken, tokenCount, currentPairSlot, channelStart, baseDim, dim);
        nextToken += tokenCount;
    }

    ProcessFnRollingToken(start, len, 0, slot0, channelStart, baseDim, dim);
    if (nextToken < len) {

        SetFlag<HardEvent::V_MTE2>(rollingPairVToMte2Event_[1]);
    }

    for (int32_t t = 1; t < len;) {
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[currentPairSlot]);
        const int32_t tokenCount = ((len - t) < kTokensPerCopy) ? (len - t) : kTokensPerCopy;
        const int32_t nextPairSlot = (currentPairSlot == kPairSlot0) ? kPairSlot1 : kPairSlot0;

        if (nextToken < len) {
            const int32_t nextTokenCount =
                ((len - nextToken) < kTokensPerCopy) ? (len - nextToken) : kTokensPerCopy;
            CopyFnRollingInputBatch(start, nextToken, nextTokenCount, nextPairSlot, channelStart, baseDim, dim);
            nextToken += nextTokenCount;
        }

        for (int32_t i = 0; i < tokenCount; ++i) {
            ProcessFnRollingToken(start, len, t + i, currentPairSlot + i, channelStart, baseDim, dim);
        }

        if (nextToken < len) {

            const int32_t currentPairBuffer = currentPairSlot / 2;
            SetFlag<HardEvent::V_MTE2>(rollingPairVToMte2Event_[currentPairBuffer]);
        }

        t += tokenCount;
        currentPairSlot = nextPairSlot;
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::DrainTaskMte3()
{

    SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim)
{
    if (!HasConvStates()) {
        return;
    }
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int64_t convStateStride0 = tilingData_->convStateStride0;
    const int64_t convStateStride1 = tilingData_->convStateStride1;
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();
    const int32_t lastSlot = SlotCurr(lastT);
    const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * convStateStride0 + channelStart;

    for (int32_t pos = 0; pos < (width - 1); ++pos) {
        const int32_t tap = (width - 2) - pos;
        const int32_t slot = RetreatRingSlot(lastSlot, tap);
        const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * convStateStride1;
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], baseDim);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackFnState(int32_t seq, int32_t cacheIdx, bool hasInit,
                                                             int32_t seqStart, int32_t seqLen, int32_t channelStart,
                                                             int32_t baseDim, int32_t dim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }
    if (!HasConvStates() || seqLen <= 0) {
        return;
    }

    constexpr int32_t historyCount = kTemplateWidth - 1;
    LocalTensor<T> stateT = inBuf.Get<T>();
    bool hasGmCopy = false;
    bool hasVectorInit = false;

    if (seqLen >= historyCount) {
        const int32_t firstToken = seqStart + seqLen - historyCount;
        const int64_t xOffset = static_cast<int64_t>(firstToken) * dim + channelStart;
        const int32_t blockLen = baseDim * static_cast<int32_t>(sizeof(T)) / 32;
        const int32_t srcStride = (dim - baseDim) * static_cast<int32_t>(sizeof(T)) / 32;
        const int32_t dstStride = (MAX_BLOCK_DIM - baseDim) * static_cast<int32_t>(sizeof(T)) / 32;
        if (historyCount > 1 && blockLen <= 65535 && srcStride <= 65535 && dstStride <= 65535) {
            const DataCopyParams copyParams{static_cast<uint16_t>(historyCount), static_cast<uint16_t>(blockLen),
                                            static_cast<uint16_t>(srcStride), static_cast<uint16_t>(dstStride)};
            DataCopy(stateT, xGm[xOffset], copyParams);
        } else {
            for (int32_t pos = 0; pos < historyCount; ++pos) {
                const int64_t tokenOffset = static_cast<int64_t>(firstToken + pos) * dim + channelStart;
                DataCopy(stateT[pos * MAX_BLOCK_DIM], xGm[tokenOffset], baseDim);
            }
        }
        hasGmCopy = true;
    } else {
        for (int32_t pos = 0; pos < historyCount; ++pos) {
            const int32_t relativeToken = seqLen - historyCount + pos;
            LocalTensor<T> stateSlot = stateT[pos * MAX_BLOCK_DIM];
            if (relativeToken >= 0) {
                const int64_t xOffset = static_cast<int64_t>(seqStart + relativeToken) * dim + channelStart;
                DataCopy(stateSlot, xGm[xOffset], baseDim);
                hasGmCopy = true;
            } else if (hasInit) {
                const int32_t statePos = relativeToken + historyCount;
                if (tilingData_->hasInitStateWorkspace != 0) {
                    const int64_t snapshotOffset =
                        (static_cast<int64_t>(seq) * historyCount + statePos) * dim + channelStart;
                    DataCopy(stateSlot, initStateWorkspaceGm_[snapshotOffset], baseDim);
                } else {
                    const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * tilingData_->convStateStride0 +
                                                static_cast<int64_t>(statePos) * tilingData_->convStateStride1 +
                                                channelStart;
                    DataCopy(stateSlot, convStatesGm[stateOffset], baseDim);
                }
                hasGmCopy = true;
            } else {
                Duplicate(stateSlot, static_cast<T>(0), baseDim);
                hasVectorInit = true;
            }
        }
    }

    if (hasGmCopy) {
        SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
    }
    if (hasVectorInit) {
        SetFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
    }

    const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * tilingData_->convStateStride0 + channelStart;
    const int64_t dstGap = tilingData_->convStateStride1 - baseDim;
    const int32_t blockLen = baseDim * static_cast<int32_t>(sizeof(T)) / 32;
    const int32_t srcStride = (MAX_BLOCK_DIM - baseDim) * static_cast<int32_t>(sizeof(T)) / 32;
    constexpr int64_t elementsPerBlock = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t dstStride = dstGap / elementsPerBlock;
    bool canBatchStateWrite = historyCount > 1 && blockLen <= 65535 && srcStride <= 65535 && dstGap >= 0 &&
                              dstGap % elementsPerBlock == 0 && dstStride <= 65535;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

    canBatchStateWrite = false;
#endif
    if (canBatchStateWrite) {
        const DataCopyParams copyParams{static_cast<uint16_t>(historyCount), static_cast<uint16_t>(blockLen),
                                        static_cast<uint16_t>(srcStride), static_cast<uint16_t>(dstStride)};
        DataCopy(convStatesGm[stateBaseOffset], stateT, copyParams);
    } else {
        for (int32_t pos = 0; pos < historyCount; ++pos) {
            const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * tilingData_->convStateStride1;
            DataCopy(convStatesGm[stateOffset], stateT[pos * MAX_BLOCK_DIM], baseDim);
        }
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackStateSpec(int32_t cacheIdx, bool hasInit,
                                                               int32_t stateTokenOffset, int32_t start, int32_t len,
                                                               int32_t channelStart, int32_t baseDim, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);

    const int32_t stateLen = tilingData_->stateLen;
    const int64_t convStateStride0 = tilingData_->convStateStride0;
    const int64_t convStateStride1 = tilingData_->convStateStride1;
    if (len <= 0) {
        return;
    }

    if (width != 4) {
        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    constexpr int32_t keep = MAX_WIDTH - 2;
    const int32_t reqStateLen = keep + len;
    if (reqStateLen > stateLen) {

        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> buf0 = ring[0 * MAX_BLOCK_DIM];
    LocalTensor<T> buf1 = ring[1 * MAX_BLOCK_DIM];

    if (hasInit) {
        const int32_t srcPos0 = stateTokenOffset + 1;
        const int32_t srcPos1 = stateTokenOffset + 2;
        const int64_t srcOffset0 = static_cast<int64_t>(cacheIdx) * convStateStride0 +
                                   static_cast<int64_t>(srcPos0) * convStateStride1 + channelStart;
        const int64_t srcOffset1 = static_cast<int64_t>(cacheIdx) * convStateStride0 +
                                   static_cast<int64_t>(srcPos1) * convStateStride1 + channelStart;
        DataCopy(buf0, convStatesGm[srcOffset0], baseDim);
        DataCopy(buf1, convStatesGm[srcOffset1], baseDim);
        SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        const int64_t dstOffset0 = static_cast<int64_t>(cacheIdx) * convStateStride0 + channelStart;
        const int64_t dstOffset1 = static_cast<int64_t>(cacheIdx) * convStateStride0 + convStateStride1 + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf1, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    } else {
        Duplicate(buf0, static_cast<T>(0), baseDim);
        SetFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        const int64_t dstOffset0 = static_cast<int64_t>(cacheIdx) * convStateStride0 + channelStart;
        const int64_t dstOffset1 = static_cast<int64_t>(cacheIdx) * convStateStride0 + convStateStride1 + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf0, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    }

    const int64_t xOffset0 = static_cast<int64_t>(start) * dim + channelStart;
    DataCopy(buf0, xGm[xOffset0], baseDim);
    SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t curr = t & 1;
        const int32_t next = curr ^ 1;
        LocalTensor<T> currBuf = (curr == 0) ? buf0 : buf1;
        LocalTensor<T> nextBuf = (next == 0) ? buf0 : buf1;

        WaitFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[curr]);

        if (t + 1 < len) {
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            if (t > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[next]);
            }
            DataCopy(nextBuf, xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[next]);
        }

        const int64_t dstOffset = static_cast<int64_t>(cacheIdx) * convStateStride0 +
                                  static_cast<int64_t>(keep + t) * convStateStride1 + channelStart;
        DataCopy(convStatesGm[dstOffset], currBuf, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[curr]);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    if (len > 1) {
        WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen,
                                                                 int32_t &start, int32_t &len) const
{
    switch (GetSeqTaskWindowMode(inputMode)) {
        case SEQ_TASK_WINDOW_MODE_VARLEN:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_VARLEN>(seq, seqLen, start, len);
        case SEQ_TASK_WINDOW_MODE_DECODE2D:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_DECODE2D>(seq, seqLen, start, len);
        default:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_BATCH>(seq, seqLen, start, len);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
template <int32_t kWindowMode>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start,
                                                                       int32_t &len) const
{
    SeqTaskWindow window;
    if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_VARLEN) {
        const int64_t startValue = ReadQueryStartLoc(seq);
        const int64_t endValue = ReadQueryStartLoc(seq + 1);
        if (startValue < 0 || endValue < startValue || endValue > tilingData_->cuSeqlen) {
            return false;
        }
        const int32_t startVal = static_cast<int32_t>(startValue);
        const int32_t endVal = static_cast<int32_t>(endValue);
        window = BuildSeqTaskWindowVarlen(startVal, endVal);
    } else if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_DECODE2D) {
        window = BuildSeqTaskWindowDecode2D(seq);
    } else {
        window = BuildSeqTaskWindowBatch(seq, seqLen);
    }

    if (!window.valid) {
        return false;
    }
    start = window.start;
    len = window.len;
    return true;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices,
                                                                 int32_t &cacheIdx) const
{
    cacheIdx = seq;
    if (!hasCacheIndices) {
        return true;
    }

    const int64_t cacheIdx64 = ReadCacheIndex(seq);
    if (cacheIdx64 == tilingData_->padSlotId ||
        (tilingData_->hasNullBlock != 0 && cacheIdx64 == tilingData_->nullBlockId)) {
        return false;
    }
    if (!HasConvStates()) {

        cacheIdx = 0;
        return true;
    }
    if (cacheIdx64 < 0 || cacheIdx64 >= tilingData_->numCacheLines) {
        return false;
    }
    cacheIdx = static_cast<int32_t>(cacheIdx64);
    return true;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline int64_t CAUSAL_CONV1D_CLASS::ReadQueryStartLoc(int32_t index) const
{
    return (tilingData_->queryStartLocDtype == 1) ? static_cast<int64_t>(queryStartLocI32Gm.GetValue(index))
                                                  : queryStartLocGm.GetValue(index);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline int64_t CAUSAL_CONV1D_CLASS::ReadCacheIndex(int32_t index) const
{
    const int32_t offset = index * static_cast<int32_t>(tilingData_->cacheIndicesStride);
    return (tilingData_->cacheIndicesDtype == 1) ? static_cast<int64_t>(cacheIndicesI32Gm.GetValue(offset))
                                                 : cacheIndicesGm.GetValue(offset);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline int64_t CAUSAL_CONV1D_CLASS::ReadHasInitialState(int32_t index) const
{
    if (tilingData_->hasInitialStateDtype == 2) {
        return static_cast<int64_t>(hasInitialStateBoolGm.GetValue(index));
    }
    return (tilingData_->hasInitialStateDtype == 1) ? static_cast<int64_t>(hasInitialStateI32Gm.GetValue(index))
                                                    : hasInitialStateGm.GetValue(index);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline int64_t CAUSAL_CONV1D_CLASS::ReadNumAcceptedTokens(int32_t index) const
{
    return (tilingData_->numAcceptedTokensDtype == 1)
               ? static_cast<int64_t>(numAcceptedTokensI32Gm.GetValue(index))
               : numAcceptedTokensGm.GetValue(index);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqHasInit(int32_t seq, bool hasInitialState) const
{
    return HasConvStates() && hasInitialState && (ReadHasInitialState(seq) != 0);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefault()
{
    switch (GetSeqTaskWindowMode(tilingData_->inputMode)) {
        case SEQ_TASK_WINDOW_MODE_VARLEN:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
            return;
        case SEQ_TASK_WINDOW_MODE_DECODE2D:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_DECODE2D>();
            return;
        default:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
            return;
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
template <int32_t kWindowMode>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefaultByWindowMode()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
    const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);

    const bool hasInit = true;
    const bool isSpecDecodingGlobal = IsUpdateSpecDecodingEnabled();

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * baseDimCnt;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / baseDimCnt);
        const int32_t baseDimIdx = static_cast<int32_t>(task % baseDimCnt);
        const int32_t channelStart = baseDimIdx * baseDim;
        if (channelStart >= dim) {
            continue;
        }
        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        int32_t start = 0;
        int32_t len = 0;
        if (!ResolveSeqTaskWindowByMode<kWindowMode>(seq, seqLen, start, len)) {
            continue;
        }

        int32_t cacheIdx = 0;
        if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
            continue;
        }

        int32_t stateTokenOffset = 0;
        if (isSpecDecodingGlobal) {
            int32_t accepted = static_cast<int32_t>(ReadNumAcceptedTokens(seq));
            stateTokenOffset = accepted - 1;
            const int32_t maxOffset = static_cast<int32_t>(tilingData_->stateLen - (width - 1));
            if (stateTokenOffset < 0) {
                stateTokenOffset = 0;
            } else if (stateTokenOffset > maxOffset) {
                stateTokenOffset = maxOffset;
            }
        }

        LoadWeightAndBias(channelStart, curBaseDim);

        InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        RunSeq(start, len, channelStart, curBaseDim, dim);

        if (isSpecDecodingGlobal) {

            DrainTaskMte3();
            WriteBackStateSpec(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        } else {
            WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
        }

        DrainTaskMte3();
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline const CausalConv1dTilingData *CAUSAL_CONV1D_CLASS::GetTilingData() const
{
    return tilingData_;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::HasActivation() const
{
    return (tilingData_ != nullptr) && (tilingData_->activationMode != 0);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::HasBias() const
{
    return (tilingData_ != nullptr) && (tilingData_->hasBias != 0);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::HasConvStates() const
{
    return (tilingData_ != nullptr) && (tilingData_->numCacheLines > 0);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateMode() const
{
    return kIsUpdateMode;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::IsFnRollingFastPathEnabled() const
{
    return !kIsUpdateMode && (tilingData_ != nullptr) && (kFnExecutionPlan != FN_EXECUTION_PLAN_INVALID) &&
           (tilingData_->hasNumAcceptedTokens == 0) && !HasBias();
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::HasExplicitFnTokenSeqRanges() const
{
    return !kIsUpdateMode && (tilingData_ != nullptr) && (tilingData_->inputMode == 0) &&
           (tilingData_->hasExplicitTokenSeqRanges != 0) &&
           (tilingData_->explicitTokenSeqRangeCount >= tilingData_->tokenBlockCnt);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateSpecDecodingEnabled() const
{
    return kIsUpdateMode && (tilingData_->hasNumAcceptedTokens != 0) && (tilingData_->width == 4);
}

#include "causal_conv1d_fn_tasks.h"

#undef CAUSAL_CONV1D_CLASS
#undef CAUSAL_CONV1D_TEMPLATE_ARGS

}
#endif
