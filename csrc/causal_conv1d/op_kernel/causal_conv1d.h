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
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_common.h"

// #define ENABLE_CAUSAL_CONV1D_DEBUG

// #ifdef ENABLE_CAUSAL_CONV1D_DEBUG
// #define CCONV_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
// #else
// #define CCONV_PRINTF(fmt, ...)
// #endif

// #define CCONV_PRINT_IF(cond, fmt, ...)     \
//     do {                                   \
//         if (cond) {                        \
//             CCONV_PRINTF(fmt, ##__VA_ARGS__); \
//         }                                  \
//     } while (0)

// #ifdef ENABLE_CAUSAL_CONV1D_DEBUG

// #define CCONV_DUMP_TENSOR_IF(cond, tensor, size) \
//     do {                                         \
//         if (cond) {                              \
//             DumpTensor(tensor, __LINE__, size);  \
//         }                                        \
//     } while (0)
// #else
constexpr int32_t CCONV_DBG_SEQ = -1;
constexpr int32_t CCONV_DBG_C0 = -1;
constexpr int32_t CCONV_DBG_MAX_TOKENS = 0;
constexpr int32_t CCONV_DBG_VERBOSE_TOKENS = 0;
constexpr int32_t CCONV_DBG_DUMP_SIZE = 0;
constexpr bool CCONV_DBG_PRINT_SYNC = false;
constexpr bool CCONV_DBG_DUMP_WEIGHTS = false;
constexpr bool CCONV_DBG_DUMP_BIAS = false;
constexpr bool CCONV_DBG_DUMP_INIT_RING = false;
constexpr bool CCONV_DBG_DUMP_RUNSEQ = false;
constexpr bool CCONV_DBG_DUMP_PREFETCH = false;
constexpr bool CCONV_DBG_DUMP_STATE = false;

// #define CCONV_DUMP_TENSOR_IF(cond, tensor, size) \
//     do {                                         \
//     } while (0)
// #endif
using namespace AscendC;
namespace NsCausalConv1d {
using namespace NsCausalConv1dCommon;

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;

    int64_t batch;

    // attrs
    int64_t activationMode; // 0: none, 1: silu/swish
    int64_t padSlotId;      // default -1

    // optional inputs
    int64_t hasBias;        // 0/1

    // Channel-wise tiling
    int64_t dimTileSize;
    int64_t blocksPerSeq;
};
#endif // CAUSAL_CONV1D_TILING_DATA_H_

template <typename T>
class CausalConv1d
{
public:
    __aicore__ inline CausalConv1d() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y
                                 ,
                                 const  CausalConv1dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                    int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                          int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID tempVToMte2Event_;
    TEventID tempMte2ToVEvent_;
    TEventID inputMte2ToVEvent_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> cacheIndicesGm;
    GlobalTensor<bool> hasInitialStateGm;
    GlobalTensor<T> yGm;

    const  CausalConv1dTilingData* tilingData_ {nullptr};
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                            GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                            GM_ADDR y
                                             , const  CausalConv1dTilingData* tilingData)
{
    // REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    // auto tiling = (__gm__ CausalConv1dTilingData*)tilingGM;
    // GET_TILING_DATA(tilingData, tilingGM);
    tilingData_ = tilingData;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    if (tilingData_->hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convStates));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cacheIndices));
    hasInitialStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(hasInitialState));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float));

    AllocEvents();

    // CCONV_PRINT_IF(GetBlockIdx() == 0U, "[Init] dim=%d, dimTileSize=%d, blocksPerSeq=%d, batch=%d\n",
    //                tilingData_->dim, tilingData_->dimTileSize, tilingData_->blocksPerSeq, tilingData_->batch);
    // CCONV_PRINT_IF(GetBlockIdx() == 0U, "[Init] hasBias=%d, activationMode=%d, stateLen=%d, inputMode=%d\n",
    //                tilingData_->hasBias, tilingData_->activationMode, tilingData_->stateLen, tilingData_->inputMode);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
{
    tempVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    tempMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    inputMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(tempVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg)
{
    const int32_t dim = tilingData_->dim;
    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;
    (void)dbgSync;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<T> tempT = outBuf.Get<T>();

    // CCONV_PRINT_IF(dbg, "[LoadWeightAndBias] c0=%d, dimTileSize=%d\n", c0, dimTileSize);

    for (int32_t j = 0; j < MAX_WIDTH; ++j) {
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, weightGm[weightOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(weightF[j * MAX_BLOCK_DIM], tempT, RoundMode::CAST_NONE, dimTileSize);
        PipeBarrier<PIPE_ALL>();
        // if (dbg && CCONV_DBG_DUMP_WEIGHTS) {
        //     CCONV_PRINTF("[Dump][weightF] j=%d\n", j);
        //     CCONV_DUMP_TENSOR_IF(true, weightF[j * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        // }
    }

    if (tilingData_->hasBias != 0) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, biasGm[c0], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(biasF, tempT, RoundMode::CAST_NONE, dimTileSize);
        PipeBarrier<PIPE_ALL>();
        // if (dbg && CCONV_DBG_DUMP_BIAS) {
        //     CCONV_PRINTF("[Dump][biasF]\n");
        //     CCONV_DUMP_TENSOR_IF(true, biasF, CCONV_DBG_DUMP_SIZE);
        // }
    } else {
        Duplicate(biasF, 0.0f, dimTileSize);
        // CCONV_PRINT_IF(dbg, "[LoadWeightAndBias] bias=0 (no bias)\n");
    }
        PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                                 int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    LocalTensor<T> ring = inBuf.Get<T>();

    PipeBarrier<PIPE_ALL>();
    if (hasInit) {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(i) * dim + c0;
            DataCopy(ring[i * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize);
        }
    } else {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), dimTileSize);
        }

    }
    PipeBarrier<PIPE_ALL>();

    if (len > 0) {
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(ring[SlotCurr(0) * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                               int32_t dim, bool dbg)
{
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;
    (void)dbgSync;
    const bool hasActivation = (tilingData_->activationMode != 0);
    const int32_t dbgMaxTokens = CCONV_DBG_MAX_TOKENS;
    const int32_t dbgVerboseTokens = CCONV_DBG_VERBOSE_TOKENS;

    for (int32_t t = 0; t < len; ++t) {
        const bool dbgTok = dbg && (t < dbgMaxTokens);
        const bool dbgVerbose = dbg && CCONV_DBG_DUMP_RUNSEQ && (t < dbgVerboseTokens);
        const bool dbgStep = dbgVerbose && (t == 0);
        const int32_t slotCurr = SlotCurr(t);
        const int32_t slotH1 = SlotHist(t, 1);
        const int32_t slotH2 = SlotHist(t, 2);
        const int32_t slotH3 = SlotHist(t, 3);
        const int32_t slotPref = (t + 1 < len) ? SlotPrefetch(t) : -1;
        const int32_t outSlot = t & 1;

        if (t + 1 < len) {
            const int64_t xOffset = static_cast<int64_t>(start + t + 1) * dim + c0;
            PipeBarrier<PIPE_ALL>();
            DataCopy(ring[slotPref * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
            PipeBarrier<PIPE_ALL>();

        }

        DataCopy(accF, biasF, dimTileSize);


        for (int32_t j = 0; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            PipeBarrier<PIPE_ALL>();
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
            PipeBarrier<PIPE_ALL>();

            PipeBarrier<PIPE_ALL>();
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);
            PipeBarrier<PIPE_ALL>();
        }

        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);
        }

        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], tmpF, dimTileSize);
            } else {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outT[outSlot * MAX_BLOCK_DIM], tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outT[outSlot * MAX_BLOCK_DIM], accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }
        PipeBarrier<PIPE_ALL>();

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm[outOffset], outT[outSlot * MAX_BLOCK_DIM], dimTileSize);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                                       int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t pos = 0; pos < (MAX_WIDTH - 1); ++pos) {
        const int32_t tap = (MAX_WIDTH - 2) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize);
        PipeBarrier<PIPE_ALL>();
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

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || blocksPerSeq * dimTileSize != dim) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        const bool dbg = (seq == CCONV_DBG_SEQ) && (c0 == CCONV_DBG_C0);

        LoadWeightAndBias(c0, dimTileSize, dbg);

        int32_t start = 0;
        int32_t len = 0;
        if (inputMode == 0) {
            const int32_t startVal = queryStartLocGm.GetValue(seq);
            const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
            start = startVal;
            len = endVal - startVal;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }

        if (len <= 0) {
            continue;
        }

        const int32_t cacheIdx = cacheIndicesGm.GetValue(seq);
        if (cacheIdx == tilingData_->padSlotId) {
            continue;
        }

        const bool hasInit = hasInitialStateGm.GetValue(seq);

        InitRing(cacheIdx, hasInit, start, len, c0, dimTileSize, dim, dbg);
        RunSeq(start, len, c0, dimTileSize, dim, dbg);
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim, dbg);
    }

    ReleaseEvents();
}

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H