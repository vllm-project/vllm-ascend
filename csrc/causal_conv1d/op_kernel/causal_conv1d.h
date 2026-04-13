/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.h
 * \brief CausalConv1D (prefill/extend) AscendC kernel implementation.
 *
 * Design highlights:
 *   - width in [2, 4] (MAX_WIDTH = 4)
 *   - dim is generalized (require dim % 16 == 0 on host for fp16/bf16 alignment)
 *   - dim tiling is chosen by the current host planner and capped by MAX_BLOCK_DIM=4096
 *     - fn: dim <= MAX_BLOCK_DIM keeps dim intact; dim > MAX_BLOCK_DIM splits dim first
 *     - update: uses the canonical candidate set {4096, 2048, 1024, 512, 384, 192}
 *     - the last dim block may be partial (tail handled by kernel)
 *   - 5-slot ring buffer (FP16/BF16 IO + FP32 compute)
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_common.h"

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
    __aicore__ inline void InitRingSeqSplit(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t tileStart,
                                            int32_t tileLen, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void RestoreFnLocalPartials(int32_t baseDim);
    __aicore__ inline void ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim);
    __aicore__ inline void AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim);
    __aicore__ inline void RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim,
                                           int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                          int32_t dim);
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
    __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, int32_t &cacheIdx) const;
    __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices, int32_t &cacheIdx) const;
    __aicore__ inline bool ResolveSeqHasInit(int32_t seq) const;
    __aicore__ inline bool ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const;
    __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                           int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim);
    __aicore__ inline void ProcessDefault();
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode();
    __aicore__ inline void ProcessVarlenTokenTiled();
    __aicore__ inline void CopyInFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t chunkStart,
                                         int32_t chunkLen, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void ComputeFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t channelStart, int32_t baseDim,
                                          int32_t dim);
    __aicore__ inline void CopyOutFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart, int32_t seqLen,
                                          int32_t cacheIdx, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void ProcessFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t seqLen,
                                          int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                          int32_t baseDim, int32_t dim);
    __aicore__ inline const CausalConv1dTilingData *GetTilingData() const;
    __aicore__ inline bool HasActivation() const;
    __aicore__ inline bool HasBias() const;
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
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];
    TEventID stateWritebackMte3ToVEvent_;
    TEventID stateWritebackMte3ToMte2Event_;
    TEventID stateShiftMte2ToMte3Event_;
    TEventID stateShiftVToMte3Event_;
    TEventID stateShiftMte3ToMte2Event_;
    TEventID specWritebackMte2ToMte3Event_[2];
    TEventID specWritebackMte3ToMte2Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int64_t> queryStartLocGm;
    GlobalTensor<int64_t> cacheIndicesGm;
    GlobalTensor<int64_t> initialStateModeGm;
    GlobalTensor<int64_t> numAcceptedTokensGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData *tilingData_{nullptr};
    bool weightCacheValid_{false};
    int32_t cachedC0_{-1};
    int32_t cachedDimTileSize_{-1};
};

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ResetRuntimeState(const CausalConv1dTilingData *tilingData)
{
    tilingData_ = tilingData;
    weightCacheValid_ = false;
    cachedC0_ = -1;
    cachedDimTileSize_ = -1;
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
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    stateShiftMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    stateShiftVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateShiftMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
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
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
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
    const int32_t jStart = MAX_WIDTH - width; // Align weight rows to taps.
    const bool hasBias = HasBias();
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<T> weightT;
    LocalTensor<T> biasT;

    if constexpr (!std::is_same<T, float>::value) {
        weightT = weightF.ReinterpretCast<T>();
        biasT = biasF.ReinterpretCast<T>();
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
            DataCopy(weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
        }
    }

    if (hasBias) {
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(biasF, biasGm[channelStart], baseDim);
        } else {
            DataCopy(biasT[MAX_BLOCK_DIM], biasGm[channelStart], baseDim);
        }
    }

    // Batch all GM->UB copies for the resident weight/bias tile, then synchronize once before Vector consumes them.
    SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);

    if constexpr (!std::is_same<T, float>::value) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t jDst = jStart + j;
            Cast(weightF[jDst * MAX_BLOCK_DIM], weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], RoundMode::CAST_NONE,
                 baseDim);
        }
        if (hasBias) {
            Cast(biasF, biasT[MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        }
        // The resident float tiles are consumed by the next RunSeq almost immediately.
        // Without an explicit vector drain here, real-device tail tiles can reuse stale
        // float bias data from the previous channel block (observed on dim=4112 bias-on update tails).
        PipeBarrier<PIPE_V>();
    }

    if (!hasBias) {
        Duplicate(biasF, 0.0f, baseDim);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                     int32_t start, int32_t len, int32_t channelStart,
                                                     int32_t baseDim, int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t ringStart = MAX_WIDTH - width; // Align history tokens to taps.
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
            const int64_t stateOffset =
                static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + channelStart;
            DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], baseDim);
        }
        // Ensure history loads are complete before later Vector reads (Cast/MulAdd) consume ring history slots.
        // Use an explicit (MTE2 -> V) event instead of PipeBarrier to avoid cross-pipe deadlocks.
        SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    } else {
        for (int32_t i = 0; i < (width - 1); ++i) {
            Duplicate(ring[(ringStart + i) * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
        }
        // `Duplicate` writes happen on Vector pipe. Ensure the ring slots are fully zero-initialized
        // before `RunSeq` starts reading them (Cast/MulAdd) to avoid using stale UB contents.
        PipeBarrier<PIPE_V>();
    }

    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);
        // Mark (state + x0) ready for the first token. RunSeq will WaitFlag(MTE2_V) before reading ring[SlotCurr(0)].
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
    }

    // Initialize prefetch throttle token so the first MTE2 prefetch can start.
    // RunSeq uses (V->MTE2) to prevent MTE2 from outrunning V and overwriting ring slots early.
    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
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
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasBias = HasBias();
    const bool hasActivation = HasActivation();
    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        // Wait for MTE2 to finish loading x[t] (and initial states for t=0) into the ring buffer.
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);

        // Prefetch x[t+1] into the 5-slot ring buffer to overlap with Vector compute.
        // Use (V->MTE2) throttling to prevent MTE2 from outrunning V and overwriting ring slots too early.
        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t); // == SlotCurr(t+1)
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
            DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
        }

        bool accInitialized = false;
        if (hasBias) {
            // Real device dispatch proved that local->local DataCopy here can drop the resident bias tile.
            // Use an explicit vector copy so accF is reliably initialized from biasF before accumulation.
            Adds(accF, biasF, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();
            accInitialized = true;
        }

        // Align with vLLM/Triton causal conv1d convention (PyTorch conv1d correlation):
        //   weight[j] * X_{t-(K-1)+j}, where K == width.
        //
        // weight rows are aligned to taps by jStart:
        //   width=4 -> jStart=0  => use j=0..3 (tap=3..0)
        //   width=3 -> jStart=1  => use j=1..3 (tap=2..0)
        //   width=2 -> jStart=2  => use j=2..3 (tap=1..0)
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

        if (hasActivation) {
            Silu(tmpF, accF, baseDim);
        } else {
            // Large-dim update cases can consume the final accumulator tile too early on real device
            // when the no-activation path casts accF directly to the output buffer.
            PipeBarrier<PIPE_V>();
        }

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {
            // Each out slot is first used without waiting, then reused every other token.
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

        // Signal that the Vector engine has finished writing to outT.
        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
        // Ensure outT is ready before MTE3 reads it.
        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outSlotT, baseDim);
        if (t + 2 < len) {
            // Only publish availability when this slot will be reused later in the same task.
            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        // Release MTE2 throttle token for the next prefetch (t+1), if any.
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

    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> state2F = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> state1F = state2F[MAX_BLOCK_DIM];
    LocalTensor<float> state0F = state1F[MAX_BLOCK_DIM];
    LocalTensor<float> currF = state0F[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    constexpr int32_t ringStart = MAX_WIDTH - kTemplateWidth;
    constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;

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
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> state2F = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> state1F = state2F[MAX_BLOCK_DIM];
    LocalTensor<float> state0F = state1F[MAX_BLOCK_DIM];
    LocalTensor<float> currF = state0F[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    (void)state2F;

    Cast(currF, ring[slotCurr * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
    PipeBarrier<PIPE_V>();
    MulAddDst(state0F, currF, weightF[3 * MAX_BLOCK_DIM], baseDim);
    PipeBarrier<PIPE_V>();

    const bool hasActivation = HasActivation();
    if (hasActivation) {
        Silu(currF, state0F, baseDim);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> state2F = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> state1F = state2F[MAX_BLOCK_DIM];
    LocalTensor<float> state0F = state1F[MAX_BLOCK_DIM];
    LocalTensor<float> currF = state0F[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;

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
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart,
                                                            int32_t baseDim, int32_t dim)
{
    if constexpr (!kHasCompileTimeWidth) {
        return;
    }

    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> state2F = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> state1F = state2F[MAX_BLOCK_DIM];
    LocalTensor<float> state0F = state1F[MAX_BLOCK_DIM];
    LocalTensor<float> currF = state0F[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasActivation = HasActivation();
    RestoreFnLocalPartials(baseDim);

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

        ComputeFnRollingOutput(slotCurr, baseDim);

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

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

        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outSlotT, baseDim);
        if (t + 2 < len) {
            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        }
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::DrainTaskMte3()
{
    // Fence all outstanding MTE3 traffic before the next task reuses the shared UB.
    SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();
    const int32_t lastSlot = SlotCurr(lastT);
    const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;

    for (int32_t pos = 0; pos < (width - 1); ++pos) {
        const int32_t tap = (width - 2) - pos;
        const int32_t slot = RetreatRingSlot(lastSlot, tap);
        const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * dim;
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], baseDim);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackStateSpec(int32_t cacheIdx, bool hasInit,
                                                               int32_t stateTokenOffset, int32_t start, int32_t len,
                                                               int32_t channelStart, int32_t baseDim, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    // vLLM speculative decoding semantics (width=4):
    // - conv_states stores a sliding buffer of length (width-1) + (len-1) = len + 2.
    // - token offset selects which 3-token window is treated as "current history".
    // - After forward, update conv_states to:
    //     [old[offset+1], old[offset+2], x[0], x[1], ..., x[len-1]]
    //   i.e., shift by 1 then append the whole segment.
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    if (width != 4) {
        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    constexpr int32_t keep = MAX_WIDTH - 2; // == 2 for width=4
    const int32_t reqStateLen = keep + len; // == len + 2 for width=4
    if (reqStateLen > stateLen) {
        // Fallback to normal semantics if conv_states is not large enough.
        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> buf0 = ring[0 * MAX_BLOCK_DIM];
    LocalTensor<T> buf1 = ring[1 * MAX_BLOCK_DIM];

    // 1) Write shifted history tokens: [old[offset+1], old[offset+2]]
    if (hasInit) {
        const int32_t srcPos0 = stateTokenOffset + 1;
        const int32_t srcPos1 = stateTokenOffset + 2;
        const int64_t srcOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos0) * dim + channelStart;
        const int64_t srcOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos1) * dim + channelStart;
        DataCopy(buf0, convStatesGm[srcOffset0], baseDim);
        DataCopy(buf1, convStatesGm[srcOffset1], baseDim);
        SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        const int64_t dstOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
        const int64_t dstOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf1, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    } else {
        Duplicate(buf0, static_cast<T>(0), baseDim);
        SetFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        const int64_t dstOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
        const int64_t dstOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf0, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    }

    // 2) Append current segment tokens with a ping-pong MTE2/MTE3 pipeline:
    //    while MTE3 writes buf[t], MTE2 can prefetch x[t+1] into the alternate buffer.
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

        const int64_t dstOffset =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(keep + t) * dim + channelStart;
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
        const int32_t startVal = queryStartLocGm.GetValue(seq);
        const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
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
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqCacheIndex(int32_t seq, int32_t &cacheIdx) const
{
    return ResolveSeqCacheIndex(seq, tilingData_->hasCacheIndices != 0, cacheIdx);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices,
                                                                 int32_t &cacheIdx) const
{
    cacheIdx = seq;
    if (!hasCacheIndices) {
        return true;
    }

    const int64_t cacheIdx64 = cacheIndicesGm.GetValue(seq);
    if (cacheIdx64 == tilingData_->padSlotId) {
        return false;
    }
    cacheIdx = static_cast<int32_t>(cacheIdx64);
    return true;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqHasInit(int32_t seq) const
{
    return ResolveSeqHasInit(seq, tilingData_->hasInitialStateMode != 0);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const
{
    if constexpr (kIsUpdateMode) {
        return true;
    }
    return hasInitialStateMode ? (initialStateModeGm.GetValue(seq) != 0) : false;
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
    const int32_t seqLen = tilingData_->seqLen; // 3D batch / decode(mode=1) 多 token 使用
    const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
    const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
    const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
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

        const bool hasInit = ResolveSeqHasInit(seq, hasInitialStateMode);
        int32_t stateTokenOffset = 0;
        if (isSpecDecodingGlobal) {
            int32_t accepted = static_cast<int32_t>(numAcceptedTokensGm.GetValue(seq));
            stateTokenOffset = accepted - 1;
            const int32_t maxOffset = static_cast<int32_t>(tilingData_->stateLen - (width - 1));
            if (stateTokenOffset < 0) {
                stateTokenOffset = 0;
            } else if (stateTokenOffset > maxOffset) {
                stateTokenOffset = maxOffset;
            }
        }

        const bool weightCacheHit =
            weightCacheValid_ && (cachedC0_ == channelStart) && (cachedDimTileSize_ == curBaseDim);
        if (!weightCacheHit) {
            LoadWeightAndBias(channelStart, curBaseDim);
            weightCacheValid_ = true;
            cachedC0_ = channelStart;
            cachedDimTileSize_ = curBaseDim;
        }

        InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        RunSeq(start, len, channelStart, curBaseDim, dim);

        if (isSpecDecodingGlobal) {
            // Spec writeback launches another MTE3-heavy tail on the same task.
            // Drain the final y store from RunSeq first, otherwise the last token output can be
            // corrupted when numAcceptedTokens-spec writeback starts reusing the MTE3 pipeline.
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

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H
