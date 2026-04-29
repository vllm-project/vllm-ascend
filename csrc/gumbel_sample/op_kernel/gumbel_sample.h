/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GUMBEL_SAMPLE_H
#define GUMBEL_SAMPLE_H

#include "kernel_operator.h"

// TilingData 由 opc 从 op_host/gumbel_sample_tiling.h 的 BEGIN_TILING_DATA_DEF 自动生成，
// 仅在 *.cpp 经 GET_TILING_DATA_WITH_STRUCT 展开后可见。此处用前向声明。
struct GumbelSampleTilingData;

using namespace AscendC;

namespace NsGumbelSample {

constexpr int32_t GUMBEL_BLOCK_SIZE  = 4096;
constexpr float   GUMBEL_EPS         = 1e-20f;
constexpr float   GUMBEL_NEG_INF     = -3.4e38f;

// Float LCG hash 常量（Numerical Recipes，与 CPU golden 字节级对齐）
// state_f = state_f * LCG_A + LCG_C  (float 乘法，IEEE 754 确定性)
// 归一化：u = Abs(state_f) * NORM_SCALE → [0, 1)
constexpr float GUMBEL_LCG_A    = 1664525.0f;
constexpr float GUMBEL_LCG_C    = 1013904223.0f;
constexpr float GUMBEL_NORM     = 2.3283064e-10f;  // 1 / 2^32

template <bool APPLY_TEMPERATURE>
class GumbelSampleOp {
public:
    __aicore__ inline GumbelSampleOp() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR temperature,
                                 GM_ADDR seeds, GM_ADDR pos,
                                 GM_ADDR sampled, GM_ADDR workspace,
                                 const GumbelSampleTilingData& t,
                                 TPipe* pipePtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneRow(uint32_t reqIdx);
    __aicore__ inline void GenerateGumbelTile(uint32_t tileOffset, uint32_t alignedLen,
                                              float gumbelSeedF);
    __aicore__ inline int32_t FindFirstMatchIdx(LocalTensor<float>& logitsLocal,
                                                 float maxVal, uint32_t alignedLen);

    // TPipe 移到核函数入口，此处改为指针（减少头尾开销）
    TPipe* pipe_ = nullptr;

    GlobalTensor<float>   logitsGm_;
    GlobalTensor<float>   tempGm_;
    GlobalTensor<int64_t> seedsGm_;
    GlobalTensor<int64_t> posGm_;
    GlobalTensor<int64_t> sampledGm_;

    // per-req 标量 buffer（Init 一次性搬入 UB，避免 GlobalTensor::GetValue）
    TBuf<TPosition::VECCALC> tempUbBuf_;
    TBuf<TPosition::VECCALC> seedsUbBuf_;
    TBuf<TPosition::VECCALC> posUbBuf_;

    // tile 临时 buffer
    TQue<QuePosition::VECIN, 2>  inQueueLogits_;   // DoubleBuffer
    TQue<QuePosition::VECOUT, 1> outQueueSampled_;
    TBuf<TPosition::VECCALC> hashBuf_;       // float hash state / g_i
    TBuf<TPosition::VECCALC> idxBuf_;        // float index [0..BLOCK_SIZE-1]
    TBuf<TPosition::VECCALC> scratchBuf_;    // argmax bcast / sentinel 中间
    TBuf<TPosition::VECCALC> reduceWorkBuf_; // ReduceMax/Min work
    TBuf<TPosition::VECCALC> maskBuf_;       // Compare 输出 bitmap (uint8)
    TBuf<TPosition::VECCALC> sentinelBuf_;   // float 全 +INF（argmax sentinel）

    uint32_t numReqs_     = 0;
    uint32_t vocabSize_   = 0;
    uint32_t blockSize_   = GUMBEL_BLOCK_SIZE;
    uint32_t numTiles_    = 0;
    uint32_t lastTileLen_ = 0;
    uint32_t myStartRow_  = 0;
    uint32_t myRows_      = 0;
};

// =================================================================
// Init
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline void GumbelSampleOp<APPLY_TEMPERATURE>::Init(
    GM_ADDR logits, GM_ADDR temperature, GM_ADDR seeds, GM_ADDR pos,
    GM_ADDR sampled, GM_ADDR /*workspace*/, const GumbelSampleTilingData& t,
    TPipe* pipePtr)
{
    pipe_ = pipePtr;
    numReqs_     = t.numReqs;
    vocabSize_   = t.vocabSize;
    blockSize_   = t.blockSize;
    numTiles_    = t.numTiles;
    lastTileLen_ = t.lastTileLen;

    uint32_t blockIdx   = GetBlockIdx();
    uint32_t formerNum  = t.formerNum;
    uint32_t nRowsLarge = t.nRowsLarge;
    uint32_t nRowsSmall = t.nRowsSmall;

    if (blockIdx < formerNum) {
        myRows_     = nRowsLarge;
        myStartRow_ = blockIdx * nRowsLarge;
    } else {
        myRows_     = nRowsSmall;
        myStartRow_ = formerNum * nRowsLarge + (blockIdx - formerNum) * nRowsSmall;
    }

    logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits));
    tempGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temperature));
    seedsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seeds));
    posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pos));
    sampledGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sampled));

    auto Align32 = [](uint32_t bytes) -> uint32_t { return ((bytes + 31u) / 32u) * 32u; };
    pipe_->InitBuffer(tempUbBuf_,  Align32(numReqs_ * static_cast<uint32_t>(sizeof(float))));
    pipe_->InitBuffer(seedsUbBuf_, Align32(numReqs_ * static_cast<uint32_t>(sizeof(int64_t))));
    pipe_->InitBuffer(posUbBuf_,   Align32(numReqs_ * static_cast<uint32_t>(sizeof(int64_t))));

    pipe_->InitBuffer(inQueueLogits_,   2, GUMBEL_BLOCK_SIZE * sizeof(float));  // DoubleBuffer
    pipe_->InitBuffer(outQueueSampled_, 1, 32);
    pipe_->InitBuffer(hashBuf_,       GUMBEL_BLOCK_SIZE * sizeof(float));
    pipe_->InitBuffer(idxBuf_,        GUMBEL_BLOCK_SIZE * sizeof(float));
    pipe_->InitBuffer(scratchBuf_,    GUMBEL_BLOCK_SIZE * sizeof(float));
    pipe_->InitBuffer(reduceWorkBuf_, GUMBEL_BLOCK_SIZE * sizeof(float));
    pipe_->InitBuffer(maskBuf_,       GUMBEL_BLOCK_SIZE / 8 + 32);
    pipe_->InitBuffer(sentinelBuf_,   GUMBEL_BLOCK_SIZE * sizeof(float));

    // 一次性初始化 idxBuf = [0.0, 1.0, 2.0, ..., BLOCK_SIZE-1]
    {
        LocalTensor<float> idxLocal = idxBuf_.Get<float>();
        CreateVecIndex(idxLocal, 0.0f, GUMBEL_BLOCK_SIZE);
    }
    // 一次性初始化 sentinelBuf = +INF（FindFirstMatchIdx 用于 ReduceMin sentinel）
    {
        LocalTensor<float> sentinelLocal = sentinelBuf_.Get<float>();
        Duplicate(sentinelLocal, static_cast<float>(3.4e38f), GUMBEL_BLOCK_SIZE);
    }
    PipeBarrier<PIPE_V>();

    // 一次性把 temperature/seeds/pos 搬入 UB（禁 GlobalTensor::GetValue）
    {
        LocalTensor<float>   tempLocal  = tempUbBuf_.Get<float>();
        LocalTensor<int64_t> seedsLocal = seedsUbBuf_.Get<int64_t>();
        LocalTensor<int64_t> posLocal   = posUbBuf_.Get<int64_t>();

        DataCopyExtParams pTemp;
        pTemp.blockCount = 1;
        pTemp.blockLen   = numReqs_ * static_cast<uint32_t>(sizeof(float));
        pTemp.srcStride  = 0;
        pTemp.dstStride  = 0;
        pTemp.rsv        = 0;
        DataCopyPadExtParams<float> ppTemp{false, 0, 0, 0.0f};
        DataCopyPad(tempLocal, tempGm_, pTemp, ppTemp);

        DataCopyExtParams pSeeds;
        pSeeds.blockCount = 1;
        pSeeds.blockLen   = numReqs_ * static_cast<uint32_t>(sizeof(int64_t));
        pSeeds.srcStride  = 0;
        pSeeds.dstStride  = 0;
        pSeeds.rsv        = 0;
        DataCopyPadExtParams<int64_t> ppSeeds{false, 0, 0, 0};
        DataCopyPad(seedsLocal, seedsGm_, pSeeds, ppSeeds);

        DataCopyExtParams pPos = pSeeds;
        DataCopyPadExtParams<int64_t> ppPos{false, 0, 0, 0};
        DataCopyPad(posLocal, posGm_, pPos, ppPos);

        event_t eMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eMte2S);
        WaitFlag<HardEvent::MTE2_S>(eMte2S);
    }
}

// =================================================================
// Process
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline void GumbelSampleOp<APPLY_TEMPERATURE>::Process()
{
    for (uint32_t i = 0; i < myRows_; ++i) {
        ProcessOneRow(myStartRow_ + i);
    }
}

// =================================================================
// GenerateGumbelTile: 全 vectorized float LCG hash → fp32 g_i 写到 hashBuf_
//   算法：state[i] = gumbelSeedF + (tileOffset+i) → 两轮 LCG → u=Abs*NORM → g=-ln(-ln(u+ε)+ε)
//   与 CPU golden 完全一致（float IEEE 754 确定性）
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline void GumbelSampleOp<APPLY_TEMPERATURE>::GenerateGumbelTile(
    uint32_t tileOffset, uint32_t alignedLen, float gumbelSeedF)
{
    LocalTensor<float> hashLocal = hashBuf_.Get<float>();
    LocalTensor<float> idxLocal  = idxBuf_.Get<float>();

    float tileOffsetF = static_cast<float>(static_cast<int32_t>(tileOffset));
    Adds(hashLocal, idxLocal, tileOffsetF + gumbelSeedF, alignedLen);
    PipeBarrier<PIPE_V>();

    Muls(hashLocal, hashLocal, GUMBEL_LCG_A, alignedLen);
    PipeBarrier<PIPE_V>();
    Adds(hashLocal, hashLocal, GUMBEL_LCG_C, alignedLen);
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, GUMBEL_LCG_A, alignedLen);
    PipeBarrier<PIPE_V>();
    Adds(hashLocal, hashLocal, GUMBEL_LCG_C, alignedLen);
    PipeBarrier<PIPE_V>();

    Abs(hashLocal, hashLocal, alignedLen);
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, GUMBEL_NORM, alignedLen);
    PipeBarrier<PIPE_V>();

    Adds(hashLocal, hashLocal, GUMBEL_EPS, alignedLen);
    PipeBarrier<PIPE_V>();
    Ln(hashLocal, hashLocal, alignedLen);
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, -1.0f, alignedLen);
    PipeBarrier<PIPE_V>();
    Adds(hashLocal, hashLocal, GUMBEL_EPS, alignedLen);
    PipeBarrier<PIPE_V>();
    Ln(hashLocal, hashLocal, alignedLen);
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, -1.0f, alignedLen);
    PipeBarrier<PIPE_V>();
}

// =================================================================
// FindFirstMatchIdx: 在 logitsLocal 中找第一个 == maxVal 的索引（vectorized）
//   过程：Compare(== maxVal) → mask → Select(idx, sentinel) → ReduceMin
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline int32_t GumbelSampleOp<APPLY_TEMPERATURE>::FindFirstMatchIdx(
    LocalTensor<float>& logitsLocal, float maxVal, uint32_t alignedLen)
{
    LocalTensor<float> bcastMax = scratchBuf_.Get<float>();
    Duplicate(bcastMax, maxVal, alignedLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
    Compare(mask, logitsLocal, bcastMax, CMPMODE::EQ, alignedLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> idxLocal      = idxBuf_.Get<float>();
    LocalTensor<float> sentinelLocal  = sentinelBuf_.Get<float>();
    LocalTensor<float> maskedIdx      = scratchBuf_.Get<float>();
    Select(maskedIdx, mask, idxLocal, sentinelLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> reduceWork = reduceWorkBuf_.Get<float>();
    ReduceMin<float>(reduceWork, maskedIdx, reduceWork[8], alignedLen, false);

    event_t eVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eVS);
    WaitFlag<HardEvent::V_S>(eVS);
    float idxF = reduceWork.GetValue(0);
    return static_cast<int32_t>(idxF);
}

// =================================================================
// ProcessOneRow: 单请求 r 完整流程
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline void GumbelSampleOp<APPLY_TEMPERATURE>::ProcessOneRow(uint32_t reqIdx)
{
    LocalTensor<float>   tempLocal  = tempUbBuf_.Get<float>();
    LocalTensor<int64_t> seedsLocal = seedsUbBuf_.Get<int64_t>();
    LocalTensor<int64_t> posLocal   = posUbBuf_.Get<int64_t>();
    float   temp   = tempLocal.GetValue(reqIdx);
    int64_t seed64 = seedsLocal.GetValue(reqIdx);
    int64_t posI64 = posLocal.GetValue(reqIdx);
    int32_t pI32   = static_cast<int32_t>(posI64 & 0xFFFFFFFF);

    // splitmix64(seed64, pI32) → gumbelSeedF（与 CPU golden 一致）
    uint64_t h = static_cast<uint64_t>(seed64) ^
                 (static_cast<uint64_t>(static_cast<uint32_t>(pI32)) << 32);
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h = h ^ (h >> 31);
    float gumbelSeedF = static_cast<float>(static_cast<int64_t>(h));

    bool  isGreedy  = (temp == 0.0f);
    float recipTemp = isGreedy ? 0.0f : (1.0f / temp);

    float   runningMax = GUMBEL_NEG_INF;
    int32_t runningIdx = 0;

    for (uint32_t tileIdx = 0; tileIdx < numTiles_; ++tileIdx) {
        uint32_t tileOffset = tileIdx * blockSize_;
        uint32_t curLen     = (tileIdx == numTiles_ - 1) ? lastTileLen_ : blockSize_;
        uint32_t alignedLen = blockSize_;

        LocalTensor<float> logitsLocal = inQueueLogits_.AllocTensor<float>();
        Duplicate(logitsLocal, GUMBEL_NEG_INF, alignedLen);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(curLen * sizeof(float));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(logitsLocal,
                    logitsGm_[static_cast<uint64_t>(reqIdx) * vocabSize_ + tileOffset],
                    copyParams, padParams);
        inQueueLogits_.EnQue(logitsLocal);
        logitsLocal = inQueueLogits_.DeQue<float>();

        if (!isGreedy) {
            if constexpr (APPLY_TEMPERATURE) {
                Muls(logitsLocal, logitsLocal, recipTemp, alignedLen);
                PipeBarrier<PIPE_V>();
            }
            GenerateGumbelTile(tileOffset, alignedLen, gumbelSeedF);
            LocalTensor<float> gLocal = hashBuf_.Get<float>();
            Add(logitsLocal, logitsLocal, gLocal, alignedLen);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> reduceWork = reduceWorkBuf_.Get<float>();
        ReduceMax<float>(reduceWork, logitsLocal, reduceWork[8], alignedLen, false);

        event_t eVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eVS);
        WaitFlag<HardEvent::V_S>(eVS);
        float localMax = reduceWork.GetValue(0);

        if (tileIdx == 0 || localMax > runningMax) {
            int32_t localFirstIdx = FindFirstMatchIdx(logitsLocal, localMax, alignedLen);
            if (localFirstIdx >= static_cast<int32_t>(curLen)) {
                localFirstIdx = 0;
            }
            runningMax = localMax;
            runningIdx = static_cast<int32_t>(tileOffset) + localFirstIdx;
        }

        inQueueLogits_.FreeTensor(logitsLocal);
    }

    // 写回 sampled[reqIdx]
    LocalTensor<int64_t> outLocal = outQueueSampled_.AllocTensor<int64_t>();
    outLocal.SetValue(0, static_cast<int64_t>(runningIdx));

    event_t eSMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eSMte3);
    WaitFlag<HardEvent::S_MTE3>(eSMte3);

    outQueueSampled_.EnQue(outLocal);
    outLocal = outQueueSampled_.DeQue<int64_t>();

    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen   = static_cast<uint32_t>(sizeof(int64_t));
    outParams.srcStride  = 0;
    outParams.dstStride  = 0;
    outParams.rsv        = 0;
    DataCopyPad(sampledGm_[reqIdx], outLocal, outParams);

    outQueueSampled_.FreeTensor(outLocal);
}

}  // namespace NsGumbelSample

#endif  // GUMBEL_SAMPLE_H
