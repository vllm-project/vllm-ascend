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

template <typename T>
class CausalConv1d
{
public:
    __aicore__ inline CausalConv1d() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                const CausalConv1dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start, int32_t len,
                                    int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void WriteBackStateSpec(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                              int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                              int32_t dim);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
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

    const CausalConv1dTilingData* tilingData_ {nullptr};

    bool weightCacheValid_ {false};
    int32_t cachedC0_ {-1};
    int32_t cachedDimTileSize_ {-1};
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                            GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                            GM_ADDR numAcceptedTokens, GM_ADDR y, const CausalConv1dTilingData* tilingData)
{
    tilingData_ = tilingData;
    weightCacheValid_ = false;
    cachedC0_ = -1;
    cachedDimTileSize_ = -1;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    if (tilingData_->hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convStates));
    if (tilingData_->inputMode == 0) {
        queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(queryStartLoc));
    }
    if (tilingData_->hasCacheIndices != 0) {
        cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cacheIndices));
    }
    if (tilingData_->hasInitialStateMode != 0) {
        initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(initialStateMode));
    }
    if (tilingData_->hasNumAcceptedTokens != 0) {
        numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(numAcceptedTokens));
    }
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float));

    AllocEvents();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
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
    specWritebackMte2ToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte2ToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte3ToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    specWritebackMte3ToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ReleaseEvents()
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
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize)
{
    const int32_t dim = tilingData_->dim;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    const bool hasBias = (tilingData_->hasBias != 0);

    for (int32_t j = 0; j < width; ++j) {
        const int32_t jDst = jStart + j;
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;

        if constexpr (std::is_same<T, float>::value) {
            DataCopy(weightF[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], dimTileSize);
        } else {
            DataCopy(weightF.ReinterpretCast<T>()[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], weightGm[weightOffset], dimTileSize);
        }
    }

    if (hasBias) {
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(biasF, biasGm[c0], dimTileSize);
        } else {
            DataCopy(biasF.ReinterpretCast<T>()[MAX_BLOCK_DIM], biasGm[c0], dimTileSize);
        }
    }

    SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);

    if constexpr (!std::is_same<T, float>::value) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t jDst = jStart + j;
            Cast(weightF[jDst * MAX_BLOCK_DIM], weightF.ReinterpretCast<T>()[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM],
                 RoundMode::CAST_NONE, dimTileSize);
        }
        if (hasBias) {
            Cast(biasF, biasF.ReinterpretCast<T>()[MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
        }
    }

    if (!hasBias) {
        Duplicate(biasF, 0.0f, dimTileSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                 int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                                 int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t ringStart = MAX_WIDTH - width;
    LocalTensor<T> ring = inBuf.Get<T>();

    if (hasInit) {
        for (int32_t i = 0; i < (width - 1); ++i) {
            const int32_t pos = stateTokenOffset + i;
            const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(pos) * dim + c0;
            DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize);
        }
        SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    } else {
        for (int32_t i = 0; i < (width - 1); ++i) {
            Duplicate(ring[(ringStart + i) * MAX_BLOCK_DIM], static_cast<T>(0), dimTileSize);
        }
        PipeBarrier<PIPE_V>();
    }

    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
    }

    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                               int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasActivation = (tilingData_->activationMode != 0);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);

        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t);
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + c0;
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
            DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], dimTileSize);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
        }

        DataCopy(accF, biasF, dimTileSize);
        PipeBarrier<PIPE_V>();

        for (int32_t j = jStart; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
//            PipeBarrier<PIPE_V>();
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);
        }

        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);
        }

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }
        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outSlotT, tmpF, dimTileSize);
            } else {
                DataCopy(outSlotT, accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outSlotT, tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outSlotT, accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }

        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;

        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outSlotT, dimTileSize);
        if (t + 2 < len) {
            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                                       int32_t dimTileSize, int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t pos = 0; pos < (width - 1); ++pos) {
        const int32_t tap = (width - 2) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackStateSpec(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                           int32_t start, int32_t len, int32_t c0,
                                                           int32_t dimTileSize, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    if (width != 4) {
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim);
        return;
    }

    constexpr int32_t keep = MAX_WIDTH - 2;
    const int32_t reqStateLen = keep + len;
    if (reqStateLen > stateLen) {
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim);
        return;
    }

    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> buf0 = ring[0 * MAX_BLOCK_DIM];
    LocalTensor<T> buf1 = ring[1 * MAX_BLOCK_DIM];

    if (hasInit) {
        const int32_t srcPos0 = stateTokenOffset + 1;
        const int32_t srcPos1 = stateTokenOffset + 2;
        const int64_t srcOffset0 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos0) * dim + c0;
        const int64_t srcOffset1 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos1) * dim + c0;
        DataCopy(buf0, convStatesGm[srcOffset0], dimTileSize);
        DataCopy(buf1, convStatesGm[srcOffset1], dimTileSize);
        PipeBarrier<PIPE_MTE2>();
        const int64_t dstOffset0 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + c0;
        const int64_t dstOffset1 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + c0;
        DataCopy(convStatesGm[dstOffset0], buf0, dimTileSize);
        DataCopy(convStatesGm[dstOffset1], buf1, dimTileSize);
        PipeBarrier<PIPE_MTE3>();
    } else {
        Duplicate(buf0, static_cast<T>(0), dimTileSize);
        PipeBarrier<PIPE_V>();
        const int64_t dstOffset0 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + c0;
        const int64_t dstOffset1 = static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + c0;
        DataCopy(convStatesGm[dstOffset0], buf0, dimTileSize);
        DataCopy(convStatesGm[dstOffset1], buf0, dimTileSize);
        PipeBarrier<PIPE_MTE3>();
    }

    const int64_t xOffset0 = static_cast<int64_t>(start) * dim + c0;
    DataCopy(buf0, xGm[xOffset0], dimTileSize);
    SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t curr = t & 1;
        const int32_t next = curr ^ 1;
        LocalTensor<T> currBuf = (curr == 0) ? buf0 : buf1;
        LocalTensor<T> nextBuf = (next == 0) ? buf0 : buf1;

        WaitFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[curr]);

        if (t + 1 < len) {
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + c0;
            if (t > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[next]);
            }
            DataCopy(nextBuf, xGm[xOffsetNext], dimTileSize);
            SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[next]);
        }

        const int64_t dstOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                  static_cast<int64_t>(keep + t) * dim + c0;
        DataCopy(convStatesGm[dstOffset], currBuf, dimTileSize);
        SetFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[curr]);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    if (len > 1) {
        WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::Process()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t inputMode = tilingData_->inputMode;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t dimTileSize = static_cast<int32_t>(tilingData_->dimTileSize);
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_->blocksPerSeq);
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const bool isSpecDecodingGlobal =
        (tilingData_->runMode == 1) && (tilingData_->hasNumAcceptedTokens != 0) && (width == 4);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        if (c0 >= dim) {
            continue;
        }
        const int32_t dimTileSizeActual = (c0 + dimTileSize <= dim) ? dimTileSize : (dim - c0);

        int32_t start = 0;
        int32_t len = 0;
        if (inputMode == 0) {
            const int32_t startVal = queryStartLocGm.GetValue(seq);
            const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
            start = startVal;
            len = endVal - startVal;
        } else if (inputMode == 2) {
            start = seq;
            len = 1;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }

        if (len <= 0) {
            continue;
        }

        int32_t cacheIdx = seq;
        if (tilingData_->hasCacheIndices != 0) {
            const int64_t cacheIdx64 = cacheIndicesGm.GetValue(seq);
            if (cacheIdx64 == tilingData_->padSlotId) {
                continue;
            }
            cacheIdx = static_cast<int32_t>(cacheIdx64);
        }

        const bool hasInit =
            (tilingData_->hasInitialStateMode != 0) ? (initialStateModeGm.GetValue(seq) != 0) : false;
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
            weightCacheValid_ && (cachedC0_ == c0) && (cachedDimTileSize_ == dimTileSizeActual);
        if (!weightCacheHit) {
            LoadWeightAndBias(c0, dimTileSizeActual);
            weightCacheValid_ = true;
            cachedC0_ = c0;
            cachedDimTileSize_ = dimTileSizeActual;
        }

        InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, c0, dimTileSizeActual, dim);
        RunSeq(start, len, c0, dimTileSizeActual, dim);

        SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);

        if (isSpecDecodingGlobal) {
            WriteBackStateSpec(cacheIdx, hasInit, stateTokenOffset, start, len, c0, dimTileSizeActual, dim);
        } else {
            WriteBackState(cacheIdx, len, c0, dimTileSizeActual, dim);
        }

        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_MTE2>();
        PipeBarrier<PIPE_MTE3>();
    }

    ReleaseEvents();
}

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H
