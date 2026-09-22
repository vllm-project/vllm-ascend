/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_li_manage_mtp_c8_kernel.h
 * \brief
 */

#ifndef FUSED_LI_MANAGE_MTP_C8_KERNEL_H
#define FUSED_LI_MANAGE_MTP_C8_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_li_manage_mtp_c8_common.h"
#include "fused_li_manage_mtp_c8_workspace.h"
#include "fused_li_manage_mtp_c8_service_vector.h"
#include "fused_li_manage_mtp_c8_service_cube.h"

namespace LIMtpC8Kernel {
using namespace LIMtpC8Common;
using namespace LIMtpC8ServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;      // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;       // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;     // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;    // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
};

template <typename LIT>
class LIMtpC8Preload {
public:
    __aicore__ inline LIMtpC8Preload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryDequantScale, __gm__ uint8_t *keyDequantScale,
                                __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKey,
                                __gm__ uint8_t *offloadSeqLengthsKey, __gm__ uint8_t *requestState,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                __gm__ uint8_t *unionPair0, __gm__ uint8_t *unionPair1,
                                __gm__ uint8_t *scoreScratch, __gm__ uint8_t *thresholdScratch,
                                __gm__ uint8_t *routeMissCounts,
                                __gm__ uint8_t *workspace, const FusedLiManageMtpC8TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using W_T = typename LIT::weightType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;

    using MM1_OUT_T = float;

    LIMatmulMtpC8<LIT> matmulService;
    LIVectorMtpC8<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t M_BASE_SIZE = 512;
    // s2 chunk 粒度: 单一来源 MtpC8Workspace::S2_BASE_SIZE(2048, 对齐
    // lightning_indexer_quant); host tiling 公式同源, 详见 workspace.h 注释。
    static constexpr uint32_t S2_BASE_SIZE = static_cast<uint32_t>(MtpC8Workspace::S2_BASE_SIZE);
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    // C8: weights (index head-merge) stay bf16 while query/key are int8.
    GlobalTensor<W_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> valueOutGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> slotOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> offloadSeqLengthsGm;
    GlobalTensor<int32_t> requestStateGm;
    // C8 dequant scales (fp16) and the vec0->cube brcb'd weight workspace.
    GlobalTensor<half> queryScaleGm;
    GlobalTensor<half> keyScaleGm;
    GlobalTensor<half> weightWsGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t totalQueries = 0U;

    LIMtpC8Common::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LIMtpC8Common::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const FusedLiManageMtpC8TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                            __gm__ uint8_t *actualSeqLengthsKey,
                                            __gm__ uint8_t *offloadSeqLengthsKey,
                                            __gm__ uint8_t *requestState);
    __aicore__ inline bool IsCausal(uint32_t bIdx);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LIMtpC8Common::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::InitTilingData(const FusedLiManageMtpC8TilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    totalQueries = tilingData->tSize;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->n1Size;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = 14U;
    constInfo.attenMaskFlag = false;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = 2048U;
    constInfo.preTokens = INT64_MAX;
    constInfo.nextTokens = INT64_MAX;
    constInfo.returnValue = false;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    constInfo.s2BaseSize = S2_BASE_SIZE;

    constInfo.s1BaseSize = 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                        __gm__ uint8_t *actualSeqLengthsKey,
                                                        __gm__ uint8_t *offloadSeqLengthsKey,
                                                        __gm__ uint8_t *requestState)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsKey == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsKey, constInfo.actualLenDims);
        offloadSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)offloadSeqLengthsKey, constInfo.actualLenDims);
        requestStateGm.SetGlobalBuffer((__gm__ int32_t *)requestState, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline bool LIMtpC8Preload<LIT>::IsCausal(uint32_t bIdx)
{
    return requestStateGm.GetValue(bIdx) == -3;
}

template <typename LIT>
__aicore__ inline uint32_t LIMtpC8Preload<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                           GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                           uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2Size = IsCausal(bIdx)
                    ? actualSeqLengthsGm.GetValue(bIdx)
                    : offloadSeqLengthsGm.GetValue(bIdx);
    const uint32_t queryEnd = actualSeqLengthsGmQ.GetValue(bIdx);
    const uint32_t queryStart = bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(bIdx - 1U);
    const int32_t state = requestStateGm.GetValue(bIdx);
    if (queryEnd <= queryStart || queryEnd > totalQueries ||
        queryEnd - queryStart > 14U ||
        (state != -3 && state != -2 && state != -1) ||
        actS2Size > constInfo.kSeqSize || actS2Size < actS1Size) {
        actS1Size = 0U;
        actS2Size = 0U;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LIMtpC8Preload<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                   uint32_t actS2Size)
{
    if (actS2Size == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = s1Offset + validS2LenBase + constInfo.s1BaseSize;
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2Size));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t LIMtpC8Preload<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!IsCausal(bIdx)) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline LIMtpC8Preload<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LIMtpC8Common::SplitCoreInfo &info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (IsCausal(bIdx)) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            for (uint32_t s2Idx = 0; s2Idx < s2BaseNum;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2BaseNum - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize -1) {
                            info.bN2End = constInfo.batchSize -1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                            __gm__ uint8_t *queryDequantScale, __gm__ uint8_t *keyDequantScale,
                                            __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                            __gm__ uint8_t *actualSeqLengthsQ,
                                            __gm__ uint8_t *actualSeqLengthsKey,
                                            __gm__ uint8_t *offloadSeqLengthsKey,
                                            __gm__ uint8_t *requestState,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                            __gm__ uint8_t *unionPair0, __gm__ uint8_t *unionPair1,
                                            __gm__ uint8_t *scoreScratch, __gm__ uint8_t *thresholdScratch,
                                            __gm__ uint8_t *routeMissCounts,
                                            __gm__ uint8_t *workspace, const FusedLiManageMtpC8TilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsKey,
                     offloadSeqLengthsKey, requestState);

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S, C8下每核[s1BaseSize,s2BaseSize]行)|weightWsGm(vec0折叠的w*q_scale)|
    // vec1ResGm(存LD中间结果)|vec1ParamGm(存LD参数)
    uint64_t offset = 0;

    // mm1开DoubleBuffer: 两段mma后已是[s1,s2] fp32最终分, 每核 s1BaseSize 行
    uint64_t singleCoreMm1ResSize = WS_DOBULE * constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // C8 weight workspace: 每核 mBaseSize * BLOCK_CUBE 个 half (vec0 Brcb 写, cube WeightDmaCopy 读)
    uint64_t singleCoreWeightWsSize = constInfo.mBaseSize * BLOCK_CUBE * sizeof(half);
    weightWsGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * singleCoreWeightWsSize));
    offset += GetBlockNum() * singleCoreWeightWsSize;

    // ld流程需要ws大小: [aicnum, 2, CeilDiv(constInfo.mBaseSize, constInfo.gSize), topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOBULE * WS_DOBULE * BASE_TOPK * sizeof(float);

    // (aic, 8, 2, 16)
    // (aic, s1_cube, 头尾，16ele)
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOBULE * LD_PARAM_NUM * sizeof(int64_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        slotOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseValues);
        // query_dequant_scale: [T, N1] fp16; index_key_dequant_scale: [blocks, 128, 1] fp16
        queryScaleGm.SetGlobalBuffer((__gm__ half *)queryDequantScale, totalQueries * constInfo.qHeadNum);
        keyScaleGm.SetGlobalBuffer((__gm__ half *)keyDequantScale,
                                   constInfo.batchSize * constInfo.maxBlockNumPerBatch * constInfo.kCacheBlockSize);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, indiceOutGm, valueOutGm,
                                           reqPoolEntriesGm, cacheSlotsGm, slotOutGm,
                                           actualSeqLengthsGmQ, requestStateGm,
                                           static_cast<uint32_t>(constInfo.batchSize),
                                           unionPair0, unionPair1, scoreScratch, thresholdScratch, routeMissCounts);
        vectorService.InitC8GlobalTensor(queryScaleGm, keyScaleGm, blockTableGm, weightWsGm, weightsGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWsGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (IsCausal(tempLoopInfo.bIdx)) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIMtpC8Common::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.causal = IsCausal(runInfo.bIdx);
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        LIMtpC8Common::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, LIMtpC8Common::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
                              runInfo.n2Idx * constInfo.sparseCount;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum
    * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }
    ProcessMain();
    ProcessDecode();
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t queryRows = totalQueries;
        if constexpr (LAYOUT_T != LI_LAYOUT::TND) {
            queryRows = constInfo.batchSize * constInfo.qSeqSize;
        }
        uint64_t totalOutputSize =
            queryRows * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            LIMtpC8Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
            if (constInfo.returnValue) {
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint16_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint16_t> valueOut = valueOutGmTmp[baseSize];

                uint16_t negInf = 0;
                if constexpr(std::is_same<K_T, float16_t>::value) {
                    negInf = 0xFC00;
                } else {
                    negInf = 0xFF80;
                }
                AscendC::InitGlobalMemory(valueOut, dealSize, negInf);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        // 预置两份 ws 释放信号, 使第一个 gS1 块的 vec0 折叠无需等待 (c8 同款)
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    LIMtpC8Common::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, LIMtpC8Common::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        if (runInfo.isFirstS2InnerLoop) {
            // 本 gS1 块的 w'=w*qScale 折叠就绪 (vec0 每 gS1 块只跑一次)
            CrossCoreWaitFlag(constInfo.syncV0C1);
        }
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
        if (runInfo.isLastS2InnerLoop) {
            // ws 已被本块 WeightDmaCopy 消费完, 释放给下一次折叠
            CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        }
    } else {
        if (runInfo.isFirstS2InnerLoop) {
            // 仅偶 AIV 实际折叠 (ProcessVec0 内部奇 AIV 早退), 奇 AIV 只参与握手
            CrossCoreWaitFlag(constInfo.syncC1V0);
            vectorService.ProcessVec0(runInfo);
            CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
        }
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        // 照抄 c8: PIPE_MTE3 而非 PIPE_MTE2 —— cube 会覆写 parity 缓冲,
        // mm1Res 的 MTE2 读必须 drained; MTE3 drained 经 V 管传递保证
        CrossCoreSetFlag<LIMtpC8Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LIMtpC8Preload<LIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLD) {
            vectorService.ProcessLD();
        }
    }
}
} // namespace LIMtpC8Kernel
#endif // FUSED_LI_MANAGE_MTP_C8_KERNEL_H
