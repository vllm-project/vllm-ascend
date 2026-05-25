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

// Philox4x32-10 常量（与 triton tl.rand/tl.randint 完全一致）
constexpr uint32_t PHILOX_KEY_A   = 0x9E3779B9u;
constexpr uint32_t PHILOX_KEY_B   = 0xBB67AE85u;
constexpr uint32_t PHILOX_ROUND_A = 0xD2511F53u;
constexpr uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;
constexpr int32_t  PHILOX_ROUNDS  = 10;
// uint_to_uniform_float scale（triton: 4.6566127342e-10 = (2^23-1)/2^23 * 2^(32-1) 的倒数）
constexpr float PHILOX_FLOAT_SCALE = 4.6566127342e-10f;

template <bool APPLY_TEMPERATURE>
class GumbelSampleOp {
public:
    __aicore__ inline GumbelSampleOp() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR temperature,
                                 GM_ADDR seeds, GM_ADDR pos,
                                 GM_ADDR idxMapping,
                                 GM_ADDR sampled, GM_ADDR workspace,
                                 const GumbelSampleTilingData& t,
                                 TPipe* pipePtr);  // [opt-1] TPipe 移到核函数入口，传指针
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneRow(uint32_t reqIdx);
    __aicore__ inline void GenerateGumbelTile(uint32_t tileOffset, uint32_t curLen,
                                              uint32_t alignedLen,
                                              uint32_t gumbelSeedLo, uint32_t gumbelSeedHi);
    __aicore__ inline int32_t FindFirstMatchIdx(LocalTensor<float>& logitsLocal,
                                                 float maxVal, uint32_t alignedLen);
    // Philox4x32-10 单元素实现（与 triton tl.rand/tl.randint 字节级对齐）
    __aicore__ inline uint32_t Philox4x32(uint32_t c0, uint32_t k0, uint32_t k1) const;

    // [opt-1] TPipe 移到核函数入口，此处改为指针（减少头尾开销）
    TPipe* pipe_ = nullptr;

    GlobalTensor<float>   logitsGm_;
    GlobalTensor<float>   tempGm_;
    GlobalTensor<int64_t> seedsGm_;
    GlobalTensor<int64_t> posGm_;
    GlobalTensor<int32_t> idxMappingGm_;
    GlobalTensor<int64_t> sampledGm_;

    // per-req 标量 buffer（Init 一次性搬入 UB，避免 GlobalTensor::GetValue，约束 #7）
    TBuf<TPosition::VECCALC> tempUbBuf_;
    TBuf<TPosition::VECCALC> seedsUbBuf_;
    TBuf<TPosition::VECCALC> posUbBuf_;
    TBuf<TPosition::VECCALC> idxMappingUbBuf_;

    // tile 临时 buffer
    TQue<QuePosition::VECIN, 2>  inQueueLogits_;   // [opt-2] DoubleBuffer（单→双缓冲）
    TQue<QuePosition::VECOUT, 1> outQueueSampled_;
    TBuf<TPosition::VECCALC> hashBuf_;       // float hash state / g_i
    TBuf<TPosition::VECCALC> idxBuf_;        // float index [0..BLOCK_SIZE-1]
    TBuf<TPosition::VECCALC> scratchBuf_;    // argmax bcast / sentinel 中间
    TBuf<TPosition::VECCALC> reduceWorkBuf_; // ReduceMax/Min work
    TBuf<TPosition::VECCALC> maskBuf_;       // Compare 输出 bitmap (uint8)
    TBuf<TPosition::VECCALC> sentinelBuf_;   // float 全 +INF（argmax sentinel）

    uint32_t numTokens_   = 0;  // total token slots (= logits.dim(0))
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
    GM_ADDR idxMapping,
    GM_ADDR sampled, GM_ADDR /*workspace*/, const GumbelSampleTilingData& t,
    TPipe* pipePtr)  // [opt-1] 接受外部 TPipe 指针
{
    pipe_ = pipePtr;  // [opt-1] 绑定外部 TPipe
    numTokens_   = t.numTokens;
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
    idxMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));
    sampledGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sampled));

    auto Align32 = [](uint32_t bytes) -> uint32_t { return ((bytes + 31u) / 32u) * 32u; };
    pipe_->InitBuffer(tempUbBuf_,       Align32(t.numReqStates * static_cast<uint32_t>(sizeof(float))));
    pipe_->InitBuffer(seedsUbBuf_,      Align32(t.numReqStates * static_cast<uint32_t>(sizeof(int64_t))));
    pipe_->InitBuffer(idxMappingUbBuf_, Align32(t.numTokens * static_cast<uint32_t>(sizeof(int32_t))));
    pipe_->InitBuffer(posUbBuf_,        Align32(t.numTokens * static_cast<uint32_t>(sizeof(int64_t))));

    pipe_->InitBuffer(inQueueLogits_,   2, GUMBEL_BLOCK_SIZE * sizeof(float));  // [opt-2] DoubleBuffer
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

    // 一次性把 temperature/seeds/idx_mapping/pos 搬入 UB（约束 #7：禁 GlobalTensor::GetValue）
    // isPad=true + rightPadding：当 blockLen 不是 32 字节倍数时，用安全值填充超出部分，
    // 避免 DataCopyPad 读取 GM 越界数据（越界数据可能是相邻张量的字节，导致计算错误）。
    {
        auto Align32Bytes = [](uint32_t bytes) -> uint32_t {
            return ((bytes + 31u) / 32u) * 32u;
        };

        LocalTensor<float>   tempLocal       = tempUbBuf_.Get<float>();
        LocalTensor<int64_t> seedsLocal      = seedsUbBuf_.Get<int64_t>();
        LocalTensor<int32_t> idxMappingLocal = idxMappingUbBuf_.Get<int32_t>();
        LocalTensor<int64_t> posLocal        = posUbBuf_.Get<int64_t>();

        {
            uint32_t rawBytes = t.numReqStates * static_cast<uint32_t>(sizeof(float));
            uint32_t padElems = (Align32Bytes(rawBytes) - rawBytes) / static_cast<uint32_t>(sizeof(float));
            DataCopyExtParams pTemp;
            pTemp.blockCount = 1; pTemp.blockLen = rawBytes;
            pTemp.srcStride = 0; pTemp.dstStride = 0; pTemp.rsv = 0;
            DataCopyPadExtParams<float> ppTemp{padElems > 0, 0, static_cast<uint8_t>(padElems), 0.0f};
            DataCopyPad(tempLocal, tempGm_, pTemp, ppTemp);
        }
        {
            uint32_t rawBytes = t.numReqStates * static_cast<uint32_t>(sizeof(int64_t));
            uint32_t padElems = (Align32Bytes(rawBytes) - rawBytes) / static_cast<uint32_t>(sizeof(int64_t));
            DataCopyExtParams pSeeds;
            pSeeds.blockCount = 1; pSeeds.blockLen = rawBytes;
            pSeeds.srcStride = 0; pSeeds.dstStride = 0; pSeeds.rsv = 0;
            DataCopyPadExtParams<int64_t> ppSeeds{padElems > 0, 0, static_cast<uint8_t>(padElems), 0};
            DataCopyPad(seedsLocal, seedsGm_, pSeeds, ppSeeds);
        }
        {
            uint32_t rawBytes = t.numTokens * static_cast<uint32_t>(sizeof(int32_t));
            uint32_t padElems = (Align32Bytes(rawBytes) - rawBytes) / static_cast<uint32_t>(sizeof(int32_t));
            DataCopyExtParams pIdxMapping;
            pIdxMapping.blockCount = 1; pIdxMapping.blockLen = rawBytes;
            pIdxMapping.srcStride = 0; pIdxMapping.dstStride = 0; pIdxMapping.rsv = 0;
            DataCopyPadExtParams<int32_t> ppIdxMapping{padElems > 0, 0, static_cast<uint8_t>(padElems), 0};
            DataCopyPad(idxMappingLocal, idxMappingGm_, pIdxMapping, ppIdxMapping);
        }
        {
            uint32_t rawBytes = t.numTokens * static_cast<uint32_t>(sizeof(int64_t));
            uint32_t padElems = (Align32Bytes(rawBytes) - rawBytes) / static_cast<uint32_t>(sizeof(int64_t));
            DataCopyExtParams pPos;
            pPos.blockCount = 1; pPos.blockLen = rawBytes;
            pPos.srcStride = 0; pPos.dstStride = 0; pPos.rsv = 0;
            DataCopyPadExtParams<int64_t> ppPos{padElems > 0, 0, static_cast<uint8_t>(padElems), 0};
            DataCopyPad(posLocal, posGm_, pPos, ppPos);
        }

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
// Philox4x32-10 单元素实现
//   对应 triton-ascend: philox(seed, c0=offset, c1=0, c2=0, c3=0)[0]
//   NPU 后端将 tt.mulhiui 编译为有符号 smulhi，需与此对齐。
//   k0 = seed & 0xFFFFFFFF, k1 = (seed >> 32) & 0xFFFFFFFF
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline uint32_t GumbelSampleOp<APPLY_TEMPERATURE>::Philox4x32(
    uint32_t c0, uint32_t k0, uint32_t k1) const
{
    uint32_t c1 = 0u, c2 = 0u, c3 = 0u;
    for (int32_t r = 0; r < PHILOX_ROUNDS; ++r) {
        // smulhi: upper 32 bits of signed 64-bit product
        // (triton-ascend compiles tt.mulhiui as signed mulhi on NPU)
        int64_t prod_b = static_cast<int64_t>(static_cast<int32_t>(PHILOX_ROUND_B))
                       * static_cast<int64_t>(static_cast<int32_t>(c2));
        int64_t prod_a = static_cast<int64_t>(static_cast<int32_t>(PHILOX_ROUND_A))
                       * static_cast<int64_t>(static_cast<int32_t>(c0));
        uint32_t hi_b = static_cast<uint32_t>(static_cast<uint64_t>(prod_b) >> 32);
        uint32_t hi_a = static_cast<uint32_t>(static_cast<uint64_t>(prod_a) >> 32);
        uint32_t new_c0 = hi_b ^ c1 ^ k0;
        uint32_t new_c2 = hi_a ^ c3 ^ k1;
        uint32_t new_c1 = PHILOX_ROUND_B * c2;
        uint32_t new_c3 = PHILOX_ROUND_A * c0;
        c0 = new_c0; c1 = new_c1; c2 = new_c2; c3 = new_c3;
        k0 += PHILOX_KEY_A;
        k1 += PHILOX_KEY_B;
    }
    return c0;
}

// =================================================================
// GenerateGumbelTile: Philox4x32 → uniform → Gumbel 噪声（向量化 Ln）
//   1. 标量循环：Philox4x32 → uint_to_uniform_float → 写入 hashBuf_
//   2. 向量化：Adds/Ln/Muls/Adds/Ln/Muls 完成 Gumbel 变换
//   使用 AscendC 硬件 Ln 保证与 triton 精度对齐
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline void GumbelSampleOp<APPLY_TEMPERATURE>::GenerateGumbelTile(
    uint32_t tileOffset, uint32_t curLen, uint32_t alignedLen,
    uint32_t gumbelSeedLo, uint32_t gumbelSeedHi)
{
    LocalTensor<float> hashLocal = hashBuf_.Get<float>();

    // (a) 标量 Philox → uniform float，padding 填 0.5（Gumbel(0.5)=0，不影响 argmax）
    for (uint32_t i = 0; i < alignedLen; ++i) {
        float u;
        if (i < curLen) {
            uint32_t vocabIdx = tileOffset + i;
            uint32_t raw = Philox4x32(vocabIdx, gumbelSeedLo, gumbelSeedHi);
            int32_t sx = static_cast<int32_t>(raw);
            if (sx < 0) sx = -sx - 1;
            u = static_cast<float>(sx) * PHILOX_FLOAT_SCALE;
        } else {
            u = 0.5f;  // padding: Gumbel(0.5) = 0, 不影响有效元素的 argmax
        }
        hashLocal.SetValue(i, u);
    }
    PipeBarrier<PIPE_V>();

    // (b) 向量化 Gumbel 变换：g = -ln(-ln(u + eps) + eps)
    //     使用 AscendC 硬件 Ln，精度与 triton 对齐
    Adds(hashLocal, hashLocal, GUMBEL_EPS, alignedLen);   // u + eps
    PipeBarrier<PIPE_V>();
    Ln(hashLocal, hashLocal, alignedLen);                  // ln(u + eps)
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, -1.0f, alignedLen);         // -ln(u + eps)
    PipeBarrier<PIPE_V>();
    Adds(hashLocal, hashLocal, GUMBEL_EPS, alignedLen);    // -ln(u+eps) + eps
    PipeBarrier<PIPE_V>();
    Ln(hashLocal, hashLocal, alignedLen);                  // ln(-ln(u+eps)+eps)
    PipeBarrier<PIPE_V>();
    Muls(hashLocal, hashLocal, -1.0f, alignedLen);         // -ln(-ln(u+eps)+eps)
    PipeBarrier<PIPE_V>();

    // padding 元素的 Gumbel 值为 -ln(-ln(0.5+eps)+eps) ≈ 0，不影响有效元素
    // 但为安全起见，将 padding 位置覆盖为 GUMBEL_NEG_INF
    for (uint32_t i = curLen; i < alignedLen; ++i) {
        hashLocal.SetValue(i, GUMBEL_NEG_INF);
    }
    PipeBarrier<PIPE_V>();
}

// =================================================================
// FindFirstMatchIdx: 在 logitsLocal 中找第一个 == maxVal 的索引（vectorized）
//   过程：Compare(== maxVal) → mask → Select(idx, sentinel) → ReduceMin
//   sentinel = +INF（float），idx = [0..BLOCK_SIZE-1]（float）
//   返回 int32_t 索引
// =================================================================
template <bool APPLY_TEMPERATURE>
__aicore__ inline int32_t GumbelSampleOp<APPLY_TEMPERATURE>::FindFirstMatchIdx(
    LocalTensor<float>& logitsLocal, float maxVal, uint32_t alignedLen)
{
    // (1) 广播 maxVal
    LocalTensor<float> bcastMax = scratchBuf_.Get<float>();
    Duplicate(bcastMax, maxVal, alignedLen);
    PipeBarrier<PIPE_V>();

    // (2) Compare(logitsLocal == bcastMax) → bitmap
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
    Compare(mask, logitsLocal, bcastMax, CMPMODE::EQ, alignedLen);
    PipeBarrier<PIPE_V>();

    // (3) Select: maskedIdx = mask ? idx : sentinel(+INF)
    LocalTensor<float> idxLocal      = idxBuf_.Get<float>();
    LocalTensor<float> sentinelLocal  = sentinelBuf_.Get<float>();
    LocalTensor<float> maskedIdx      = scratchBuf_.Get<float>();
    Select(maskedIdx, mask, idxLocal, sentinelLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedLen);
    PipeBarrier<PIPE_V>();

    // (4) ReduceMin(maskedIdx) → first matching idx（float）
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
    // ----- (1) 读 per-req 标量（来自 Init 阶段已 DataCopyPad 到 UB 的 buffer）-----
    LocalTensor<float>   tempLocal       = tempUbBuf_.Get<float>();
    LocalTensor<int64_t> seedsLocal      = seedsUbBuf_.Get<int64_t>();
    LocalTensor<int32_t> idxMappingLocal = idxMappingUbBuf_.Get<int32_t>();
    LocalTensor<int64_t> posLocal        = posUbBuf_.Get<int64_t>();

    // idx_mapping[reqIdx] → reqStateIdx，仅用于索引 temperature 和 seeds
    uint32_t reqStateIdx = static_cast<uint32_t>(idxMappingLocal.GetValue(reqIdx));

    float   temp   = tempLocal.GetValue(reqStateIdx);   // temperature[idx_mapping[reqIdx]]
    int64_t seed64 = seedsLocal.GetValue(reqStateIdx);  // seeds[idx_mapping[reqIdx]]
    // pos 和 logits 按 batch_idx（reqIdx）直接索引
    int64_t posI64 = posLocal.GetValue(reqIdx);
    int32_t pI32   = static_cast<int32_t>(posI64 & 0xFFFFFFFF);

    // gumbel_seed = tl.randint(seed, pos) = Philox4x32(c0=pos_i32, k0=seed_lo32, k1=seed_hi32)
    uint32_t seedLo = static_cast<uint32_t>(static_cast<uint64_t>(seed64) & 0xFFFFFFFFu);
    uint32_t seedHi = static_cast<uint32_t>((static_cast<uint64_t>(seed64) >> 32) & 0xFFFFFFFFu);
    uint32_t gumbelSeed = Philox4x32(static_cast<uint32_t>(pI32), seedLo, seedHi);
    uint32_t gumbelSeedLo = gumbelSeed;  // tl.rand uses gumbel_seed as uint32 key
    uint32_t gumbelSeedHi = 0u;          // seed_hi = (gumbel_seed >> 32) = 0 (uint32 fits in lo)

    bool  isGreedy  = (temp == 0.0f);
    float recipTemp = isGreedy ? 0.0f : (1.0f / temp);

    // ----- (2) 跨 tile 累积 argmax -----
    float   runningMax = GUMBEL_NEG_INF;
    int32_t runningIdx = 0;

    for (uint32_t tileIdx = 0; tileIdx < numTiles_; ++tileIdx) {
        uint32_t tileOffset = tileIdx * blockSize_;
        uint32_t curLen     = (tileIdx == numTiles_ - 1) ? lastTileLen_ : blockSize_;
        uint32_t alignedLen = blockSize_;

        // ---- CopyIn ----
        LocalTensor<float> logitsLocal = inQueueLogits_.AllocTensor<float>();
        Duplicate(logitsLocal, GUMBEL_NEG_INF, alignedLen);
        PipeBarrier<PIPE_V>();

        // blockLen 必须是 32 字节的倍数（DMA 对齐要求）。
        // 当 curLen * sizeof(float) 不是 32 的倍数时，用 isPad=true 填充超出部分，
        // 避免读取 GM 越界数据（越界数据可能大于有效 logits，导致 argmax 返回越界索引，
        // 进而被 FindFirstMatchIdx 截断为 0）。
        uint32_t rawBytes    = curLen * static_cast<uint32_t>(sizeof(float));
        uint32_t alignBytes  = ((rawBytes + 31u) / 32u) * 32u;
        uint32_t rightPadElems = (alignBytes - rawBytes) / static_cast<uint32_t>(sizeof(float));
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = rawBytes;
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;
        DataCopyPadExtParams<float> padParams{
            rightPadElems > 0,
            0,
            static_cast<uint8_t>(rightPadElems),
            GUMBEL_NEG_INF
        };
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
            GenerateGumbelTile(tileOffset, curLen, alignedLen, gumbelSeedLo, gumbelSeedHi);
            LocalTensor<float> gLocal = hashBuf_.Get<float>();
            Add(logitsLocal, logitsLocal, gLocal, alignedLen);
            PipeBarrier<PIPE_V>();
        }

        // ---- ReduceMax ----
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

    // ----- (3) 写回 sampled[reqIdx] -----
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
