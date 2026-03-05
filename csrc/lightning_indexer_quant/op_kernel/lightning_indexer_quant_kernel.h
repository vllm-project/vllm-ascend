/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_quant_kernel.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_QUANT_KERNEL_H
#define LIGHTNING_INDEXER_QUANT_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_quant_common.h"
#include "lightning_indexer_quant_service_vector.h"
#include "lightning_indexer_quant_service_cube.h"

namespace LIQKernel {
using namespace LIQCommon;
using namespace LIQServiceVec;
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
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
};

template <typename LIQT>
class LIQPreload {
public:
    __aicore__ inline LIQPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const LIQTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename LIQT::queryType;
    using K_T = typename LIQT::keyType;
    using OUT_T = typename LIQT::outputType;
    static constexpr bool PAGE_ATTENTION = LIQT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = LIQT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIQT::keyLayout;

    using MM1_OUT_T = float;

    LIQMatmul<LIQT> matmulService;
    LIQVector<LIQT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S2_BASE_SIZE = 2048;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t LI_QUANT_PRELOAD_TASK_CACHE_SIZE = 2;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    LIQCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LIQCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const LIQTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LIQCommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            LIQCommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIQCommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::InitTilingData(const LIQTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.outputLayout = Q_LAYOUT_T;  // 输出和输入形状一致
    if (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.mBaseSize = M_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = (constInfo.mBaseSize + constInfo.gSize - 1) / constInfo.gSize;
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

template <typename LIQT>
__aicore__ inline uint32_t LIQPreload<LIQT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
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

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2Size =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
}

template <typename LIQT>
__aicore__ inline uint32_t LIQPreload<LIQT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
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

template <typename LIQT>
__aicore__ inline uint32_t LIQPreload<LIQT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
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

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIQT>
__aicore__ void inline LIQPreload<LIQT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                   LIQCommon::SplitCoreInfo &info)
{
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
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
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
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
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

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +  // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                  // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount +  // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                        // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                              __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                              __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                              __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices,
                                              __gm__ uint8_t *workspace, const LIQTilingData *__restrict tiling,
                                              TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)|vec1ResGm(存LD中间结果)|vec1ParamGm(存LD参数)
    // |Core0_mm1ResDB0-Core0_mm1ResDB1-Core1_mm1ResDB0....Core23_mm1ResDB0-Core23_mm1ResDB1|Core0_vec1Res...
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    uint64_t singleCoreMm1ResSize = WS_DOBULE * constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程需要ws大小: [aicnum, 2, CeilDiv(constInfo.mBaseSize, constInfo.gSize), topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    GlobalTensor<float> vec1ResGm;  // 存放TopK计算中间结果
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOBULE * WS_DOBULE * BASE_TOPK * sizeof(float);

    // (aic, 8, 2, 16)
    // (aic, s1_cube, 头尾，16ele)
    GlobalTensor<int64_t> vec1ParamGm;  // 存放LD参数信息
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOBULE * LD_PARAM_NUM * sizeof(int64_t);

    GlobalTensor<half> weightWorkspaceGm;  // v1阶段处理w*scale后的结果
    uint64_t weightMemSize = BLOCK_CUBE * constInfo.mBaseSize * WS_DOBULE * sizeof(half);
    weightWorkspaceGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * weightMemSize));
    offset += GetBlockNum() * weightMemSize;

    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ half *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ half *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ half *)keyScale);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecWorkspaceTensor(weightWorkspaceGm, mm1ResGm, vec1ResGm, vec1ParamGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
    }
    InitBuffers();
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
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
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIQCommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;

    if (!runInfo.isValid) {
        return;  // 需要验证， v1 时候需要runInfo
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        LIQCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, LIQCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();

    ProcessDecode();
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            LIQCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize > totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    LIQCommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];

    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);

            if ASCEND_IS_AIV {
                if (bN2LoopIdx == splitCoreInfo.bN2End && gloop > 0) {
                    CrossCoreWaitFlag(constInfo.syncC1V1);
                    vectorService.ProcessVec1(runInfo[1 - gloop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
                    CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        constInfo.syncV1C1);  // 反向同步 1
                }
            }
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
            uint32_t extraLoop = isEnd ? LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1 : 0;
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= (tempLoopInfo.s2LoopEnd + extraLoop);
                 s2LoopIdx++) {
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

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          LIQCommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE;
    LIQCommon::RunInfo &curRunInfo = runInfo[curTaskId];
    LIQCommon::RunInfo &lastRunInfo = runInfo[1 - curTaskId];

    CalcRunInfo(loop, s2LoopIdx, curRunInfo);

    if (curRunInfo.isValid) {
        if ASCEND_IS_AIC {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            CrossCoreWaitFlag(constInfo.syncV1C1);  // 反向同步 1
            matmulService.ComputeMm1(curRunInfo);
            CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            if (curRunInfo.isLastS2InnerLoop) {
                CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);  // 反向同步 0
            }
        } else {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncC1V0);  // 反向同步 0
                vectorService.ProcessVec0(curRunInfo);
                CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }

    if (lastRunInfo.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            vectorService.ProcessVec1(lastRunInfo);
            CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);  // 反向同步 1
        }
        lastRunInfo.isValid = false;
    }
}

template <typename LIQT>
__aicore__ inline void LIQPreload<LIQT>::ProcessDecode()
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
}  // namespace LIQKernel
#endif  // LIGHTNING_INDEXER_QUANT_KERNEL_H