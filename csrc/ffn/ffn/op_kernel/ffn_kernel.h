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
 * \file ffn_kernel.h
 * \brief
 */

#ifndef ASCENDC_FFN_KERNEL_H
#define ASCENDC_FFN_KERNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FFN {
using namespace AscendC;

template <class AT_, class BT_, class CT_, class BiasT_, const MatmulConfig &MM_CFG = CFG_MDL>
struct MMType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

template <typename T>
using ActiveFuncPtr = void (*)(const LocalTensor<T>, const LocalTensor<T>, uint32_t);

enum class ActiveType {
    FASTGELU = 0,
    RELU,
    SILU,
    GELU,
    INVALID_TYPE
};

constexpr int numActiveTypes = static_cast<int>(ActiveType::INVALID_TYPE); // num of useful activeTypes

template <typename T>
class ActiveType2Func {
public:
    ActiveType activeType;
    ActiveFuncPtr<T> funcPointer;
};

template <typename T>
using GluActiveFuncPtr = void (*)(const LocalTensor<T>, const LocalTensor<T>, const LocalTensor<T>, uint32_t);

enum class GluActiveType {
    GEGLU = 0,
    SWIGLU,
    REGLU,
    INVALID_TYPE
};

constexpr int numGluActiveTypes = static_cast<int>(GluActiveType::INVALID_TYPE); // num of useful gluActiveTypes

template <typename T>
class GluActiveType2Func {
public:
    GluActiveType gluActiveType;
    GluActiveFuncPtr<T> gluFuncPointer;
};

constexpr float BETA_ = 1.0;                                // beta param of swiglu
constexpr uint32_t MAX_EXPERT_PARALLELISM = 10;             // allow `MAX_EXPERT_PARALLELISM` experts to compute together
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;                 // 32: a block has 32 bytes data
constexpr uint32_t CUBE_BASE_ALIGN_FACTOR = 16;             // 16: baseM align requirement for ai cube
constexpr uint32_t CUBE_QUANT_BASE_ALIGN_FACTOR = 32;       // 32: quant matmul baseM align requirement for ai cube
constexpr uint32_t INT8_BITS = 8;                           // 8: a int8 data has 8 bits
constexpr uint32_t FP16_INT8_BEST_DATACOPY_BASE_SIZE = 512; // 512: can copy 512 elements of fp16 int8 type every time
constexpr uint32_t BF16_INT8_BEST_DATACOPY_BASE_SIZE = 256; // 256: can copy 256 elements of bf16 int8 type every time
constexpr uint32_t INT8_SYNC_N1_SIZE = 256;                 // 256: when n1 is small than 256, should enable SyncbeforeMM1
// a unit block can contain `EXPERT_NUM_ALIGN` int64_t elements
constexpr uint32_t EXPERT_NUM_ALIGN = UB_BLOCK_UNIT_SIZE / sizeof(int64_t);
constexpr uint32_t ANTIQUANT_MSD_STEP = 2;
constexpr uint32_t NUM_ALIGN_TO_THIRTYTWO = 31;                  // used to align to 32
constexpr uint32_t NUM_ALIGN_TO_SIXTEEN = 15;                    // used to align to 16
constexpr uint32_t NUM_ALIGN_TO_ONE_HUNDRED_TWEENTY_EIGHT = 127; // used to align to 128
constexpr uint32_t FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE = 8;         // a float type num need to duplicate 8 times to align 32
constexpr uint32_t DATASIZE_EACH_REPEAT_TIME = 256;              // each repeat time can calc 256Byte data
constexpr uint32_t MSD_EACH_UB_BLOCK_SIZR = 6 * 1024;            // each repeat time can calc 256Byte data
constexpr uint32_t DATABLOCK_NUM_IN_GATHER = 8;                  // In Gather API, each repeat collects 8 data blocks

template <class T>
__aicore__ inline constexpr uint32_t GetNumInUbBlock()
{
    return UB_BLOCK_UNIT_SIZE / sizeof(T);
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
    // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
    // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
    // num = a + 3 also satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero
}

/** @brief store variables for core split configuration
 */
struct MNConfig {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t coreNum;
    uint32_t blockDimM;
    uint32_t blockDimN;
    uint32_t mIdx;
    uint32_t nIdx;
    uint32_t singleM;
    uint32_t singleN;

    __aicore__ inline void SetConstriant(const uint32_t &m_, const uint32_t &n_, const uint32_t &baseM_,
                                         const uint32_t &baseN_, const uint32_t &coreNum_)
    {
        m = m_;
        n = n_;
        baseM = baseM_;
        baseN = baseN_;
        coreNum = coreNum_;
    }
};

/** @brief store history information for expert parallelism
 */
struct ExpertParallInfo {
    uint32_t GlobalOffset[MAX_EXPERT_PARALLELISM] = {0};
    int32_t expertIdxBuf[MAX_EXPERT_PARALLELISM] = {0};
    uint32_t LocalOffset[MAX_EXPERT_PARALLELISM] = {0};
    uint32_t size = 0;
    uint32_t start = 0;
    uint32_t expertParallelism = 1; // number of experts for mm1 parallelly computing
    uint32_t maxExpertParallelism;
    uint32_t maxSize;

    /** @brief constructor
     */
    __aicore__ inline ExpertParallInfo(const uint32_t &maxCoreNum, uint32_t nLoops)
    {
        if (unlikely(nLoops == 0)) {
            nLoops = 1;
        }
        maxExpertParallelism = Max<uint32_t>(maxCoreNum / nLoops, 1);
        if (maxExpertParallelism > MAX_EXPERT_PARALLELISM) {
            maxExpertParallelism = MAX_EXPERT_PARALLELISM;
        }
        maxSize = maxExpertParallelism;
    }

    /** @brief Add an expert into the buffer
     * @return isFull
     */
    __aicore__ inline bool AddExpert(uint32_t expertIdx, uint32_t tokens, uint32_t tokensOffset)
    {
        if (unlikely(size >= maxSize)) {
            return true;
        }
        GlobalOffset[size] = tokensOffset;
        expertIdxBuf[size] = expertIdx;
        if (size + 1 < maxSize) {
            LocalOffset[size + 1] = LocalOffset[size] + tokens;
        }
        size += 1;
        return size == maxSize;
    }

    /** @brief Called once the matmul processed experts in this buffer.
     */
    __aicore__ inline void Clear(const uint32_t start = 0)
    {
        // assert 0 <= start <= size-1
        size = start;
    }
};

struct UpdateKernelTilingInfo {
    uint32_t n1;
    uint32_t n2;
    uint32_t expertNum;
    uint32_t maxMM1ExpertParallelism;
};

__aicore__ inline void TokensIndicesToValues(LocalTensor<int64_t> &ubTokens, uint32_t expertNum)
{
    int64_t preOffset = 0;
    for (uint32_t i = 0; i < expertNum; i++) {
        int64_t offset = ubTokens.GetValue(i);
        ubTokens.SetValue(i, offset - preOffset);
        preOffset = offset;
    }
}

__aicore__ inline LocalTensor<int64_t> GetUbTokens(__gm__ uint8_t *expertTokens, GlobalTensor<int64_t> &expertTokensGm,
                                                   const FFNTilingData *__restrict tilingData, TPipe *pipe)
{
    uint32_t expertNum = tilingData->ffnBaseParams.expertNum;
    TBuf<QuePosition::VECCALC> eTokens64Buf;
    pipe->InitBuffer(eTokens64Buf, AlignUp<UB_BLOCK_UNIT_SIZE>(expertNum * sizeof(int64_t))); // 32-byte alignment
    LocalTensor<int64_t> ubTokens = eTokens64Buf.Get<int64_t>();
    if (likely(expertTokens != nullptr)) {
        // copy tokens array from GM
        expertTokensGm.SetGlobalBuffer((__gm__ int64_t *)expertTokens);
        DataCopy(ubTokens, expertTokensGm, AlignUp<EXPERT_NUM_ALIGN>(expertNum)); // 32-byte alignment
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (tilingData->ffnBaseParams.tokensIndexFlag) {
            TokensIndicesToValues(ubTokens, expertNum);
        }
    } else {
        ubTokens.SetValue(0, static_cast<int64_t>(tilingData->ffnBaseParams.maxTokens));
    }
    return ubTokens;
}

__aicore__ inline void ComputeExpertParallelNum(uint32_t expertI, uint32_t baseM, uint32_t tokens,
                                                uint32_t tokensOffset, ExpertParallInfo &expertParallInfo)
{
    expertParallInfo.expertParallelism = 1;
    if (tokens <= baseM) { // small expert
        bool isFull = expertParallInfo.AddExpert(expertI, tokens, tokensOffset);
        if (isFull) {
            // the buffer is full, it's time to compute experts parallelly
            expertParallInfo.expertParallelism = expertParallInfo.maxExpertParallelism;
            expertParallInfo.start = 0;
        } else {
            expertParallInfo.expertParallelism = 0; // store this expert information, not to compute the expert
        }
    } else { // large expert
        expertParallInfo.AddExpert(expertI, tokens, tokensOffset);
        expertParallInfo.expertParallelism = 1; // compute this expert solely
        expertParallInfo.start = expertParallInfo.size - 1;
    }
}

/** device side tiling function. It computes singleCoreM and singleCoreN for each AI core.
 * @param mnConfig: M and N config info for matmul.
 */
__aicore__ inline void KernelTiling(MNConfig &mnConfig)
{
    uint32_t maxNLoops = Ceil(mnConfig.n, mnConfig.baseN);
    uint32_t maxMLoops = Ceil(mnConfig.m, mnConfig.baseM);
    uint32_t curNLoops = Min(maxNLoops, mnConfig.coreNum);
    if (unlikely(curNLoops == 0)) {
        curNLoops = 1;
    }
    uint32_t curMLoops = Min(maxMLoops, mnConfig.coreNum / curNLoops);
    uint32_t curSingleN = AlignUp(Ceil(mnConfig.n, curNLoops), mnConfig.baseN);
    uint32_t curSingleM = AlignUp<CUBE_BASE_ALIGN_FACTOR>(Ceil(mnConfig.m, curMLoops));
    curSingleM = Min(Max(curSingleM, mnConfig.baseM), mnConfig.m);
    mnConfig.singleM = curSingleM;
    mnConfig.singleN = curSingleN;
    mnConfig.blockDimM = Ceil(mnConfig.m, curSingleM);
    mnConfig.blockDimN = Ceil(mnConfig.n, curSingleN);
    if (curNLoops * curMLoops <= (mnConfig.coreNum >> 1)) {
        return;
    }
    uint32_t minSingleCore = mnConfig.singleM * mnConfig.singleN; // calc loop on the single core
    while (curNLoops > 1) {
        // skip curNLoops in range (maxNLoops/2) + 1 to (maxNLoops - 1)
        curNLoops = Min(curNLoops - 1, Ceil(mnConfig.n, curSingleN + mnConfig.baseN));
        curSingleN = AlignUp(Ceil(mnConfig.n, curNLoops), mnConfig.baseN);
        curNLoops = Ceil(mnConfig.n, curSingleN);
        if (unlikely(curNLoops == 0)) {
            break;
        }
        curMLoops = Min(mnConfig.coreNum / curNLoops, maxMLoops);
        if (curNLoops * curMLoops <= (mnConfig.coreNum >> 1)) {
            break;
        }
        curSingleM = AlignUp<CUBE_BASE_ALIGN_FACTOR>(Ceil(mnConfig.m, curMLoops));
        curSingleM = Min(Max(curSingleM, mnConfig.baseM), mnConfig.m);
        curMLoops = Ceil(mnConfig.m, curSingleM);
        uint32_t curSingleCore = curSingleN * curSingleM;
        // select the smaller calc loop on the single core, preferred split N
        if (curSingleCore < minSingleCore ||
            (curSingleCore == minSingleCore && curNLoops * curMLoops < mnConfig.blockDimN * mnConfig.blockDimM) ||
            (curSingleCore == minSingleCore && curNLoops * curMLoops == mnConfig.blockDimN * mnConfig.blockDimM &&
             curSingleM + curSingleN < mnConfig.singleM + mnConfig.singleN)) {
            mnConfig.blockDimM = curMLoops;
            mnConfig.blockDimN = curNLoops;
            mnConfig.singleM = curSingleM;
            mnConfig.singleN = curSingleN;
            minSingleCore = curSingleCore;
        }
    }
}

__aicore__ inline void UpdateKernelTilingBeforeMM1(MNConfig &mnConfig, uint32_t &maxMM1UsedCubeCore, uint32_t &tokens,
                                                   const UpdateKernelTilingInfo &info,
                                                   const FFNTilingData *__restrict tilingData)
{
    uint32_t expertNum = info.expertNum;
    uint32_t maxMM1ExpertParallelism = info.maxMM1ExpertParallelism;
    uint32_t n1 = info.n1;
    uint32_t n2 = info.n2;

    uint32_t mm1ExpertParallelUsedCore;
    uint32_t mm1SingleExpertUsedCore;

    tokens = tilingData->mm1TilingData.baseM;
    if (unlikely(maxMM1ExpertParallelism == 0)) {
        maxMM1ExpertParallelism = 1;
    }
    mnConfig.SetConstriant(tokens, n1, tilingData->mm1TilingData.baseM,
                           tilingData->mm1TilingData.baseN * tilingData->mm1TilingData.stepN,
                           tilingData->ffnBaseParams.coreNum / maxMM1ExpertParallelism);
    KernelTiling(mnConfig);
    mm1ExpertParallelUsedCore = mnConfig.blockDimM * mnConfig.blockDimN * maxMM1ExpertParallelism;

    if (expertNum > 1 && tilingData->ffnBaseParams.maxTokens <= (uint32_t)tilingData->mm1TilingData.baseM) {
        maxMM1UsedCubeCore = mm1ExpertParallelUsedCore;
    } else {
        tokens = tilingData->ffnBaseParams.maxTokens;
        mnConfig.SetConstriant(tokens, n1, tilingData->mm1TilingData.baseM,
                               tilingData->mm1TilingData.baseN * tilingData->mm1TilingData.stepN,
                               tilingData->ffnBaseParams.coreNum);
        KernelTiling(mnConfig);
        mm1SingleExpertUsedCore = mnConfig.blockDimM * mnConfig.blockDimN;
        maxMM1UsedCubeCore = Max(mm1SingleExpertUsedCore, mm1ExpertParallelUsedCore);
    }

    tokens = tilingData->mm2TilingData.baseM;
    mnConfig.SetConstriant(tokens, n2, tilingData->mm2TilingData.baseM,
                           tilingData->mm2TilingData.baseN * tilingData->mm2TilingData.stepN,
                           tilingData->ffnBaseParams.coreNum);
    KernelTiling(mnConfig);
}

template <typename T>
__aicore__ inline void ApplyActivation(ActiveType activationType, LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       LocalTensor<uint8_t> &tmpLocal, uint32_t dataSize)
{
    if (activationType == ActiveType::FASTGELU) {
        FasterGelu(dst, src, tmpLocal, dataSize);
    } else if (activationType == ActiveType::RELU) {
        Relu(dst, src, dataSize);
        PipeBarrier<PIPE_V>();
    } else if (activationType == ActiveType::SILU) {
        Silu(dst, src, dataSize);
    } else if (activationType == ActiveType::GELU) {
        Gelu(dst, src, tmpLocal, dataSize);
    }
}
} // namespace FFN

#endif // ASCENDC_FFN_H
