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
 * \file quant_lightning_indexer_kernel.h
 * \brief
 */

#ifndef quant_lightning_indexer_KERNEL_H
#define quant_lightning_indexer_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "quant_lightning_indexer_service_vector.h"
#include "quant_lightning_indexer_service_cube.h"
#include "../quant_lightning_indexer_metadata.h"

namespace QLIKernel {
using namespace QLICommon;
using namespace QLIServiceVec;
using namespace matmul;
using namespace optiling::detail;
using namespace optiling;
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
    uint32_t actS2SizeOrig = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
};

template <typename QLIT>
class QLIPreload {
public:
    __aicore__ inline QLIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const QLITilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using OUT_T = typename QLIT::outputType;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;

    using MM1_OUT_T = float;

    QLIMatmul<QLIT> matmulService;
    QLIVector<QLIT> vectorService;

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
    static constexpr uint32_t ELE_NUM_PER_BLOCK = 16;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint32_t coreZeroEnable = 1U;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    QLICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig,
                                                       uint32_t &validS2Len);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen, uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitTilingData(const QLITilingData *__restrict tilingData)
{
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.cmpRatio = tilingData->cmpRatio;
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

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
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

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
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

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize, constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;   // 真实使用的压缩后S2长度
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                     uint32_t actS2SizeOrig, uint32_t &validS2Len)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        validS2Len = 0;
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);    // 压缩前的validS2LenBase
    validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) / static_cast<int32_t>(constInfo.cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t validS2Len = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig, validS2Len);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename QLIT>
__aicore__ void inline QLIPreload<QLIT>::SplitCore()
{
    constInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_CORE_ENABLE_INDEX, false));
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_M_END_INDEX, false));
    constInfo.s2End  = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, LI_S2_END_INDEX, false));

    // 如果0核都没有启动，说明所有核都没启动
    coreZeroEnable = metadataGm.GetValue(GetAttrAbsIndex(0, LI_CORE_ENABLE_INDEX, false));
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
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

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                              __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                              __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                              __gm__ uint8_t *blockTable, __gm__ uint8_t *metadata,
                                              __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                              const QLITilingData *__restrict tiling, TPipe *tPipe)
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

    if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
        // 计算分核
        SplitCore();
    }

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

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
    uint32_t s2BlockNum;
    uint32_t validS2Len = 0;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig,
                                             tempLoopInfo.validS2Len);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        tempLoopInfo.validS2Len = tempLoopInfo.actS2Size;
    }
    tempLoopInfo.s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : s2BlockNum;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.validS2Len % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail = (tempLoopInfo.s2BasicSizeTail == 0) ?
                                   constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx + 1 == constInfo.bN2End && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx < tempLoopInfo.s2LoopEnd;

    if (!runInfo.isValid) {
        return;  // 需要验证， v1 时候需要runInfo
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QLICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QLICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == constInfo.s2Start;
    runInfo.isLastS2InnerLoop = (s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd);
    runInfo.isAllLoopEnd = (runInfo.bN2Idx + 1 == constInfo.bN2End) && (runInfo.gS1Idx + 1 == constInfo.gS1End) &&
                           (runInfo.s2Idx + 1 == constInfo.s2End); // TODO:LD场景专用，后续通过metadata传入，此变量可删除

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

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Process()
{
    // 没有计算任务，直接清理输出
    if (coreZeroEnable == 0) {
        ProcessInvalid();
        return;
    }

    ProcessMain();

    ProcessDecode();
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QLICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessMain()
{
    // 无任务核直接返回
    if (constInfo.coreEnable == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    QLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];

        // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)){
        constInfo.bN2End += 1;
    }

    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);

            if ASCEND_IS_AIV {
                if (bN2LoopIdx + 1 == constInfo.bN2End && gloop > 0) {
                    CrossCoreWaitFlag(constInfo.syncC1V1);
                    vectorService.ProcessVec1(runInfo[1 - gloop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
                    CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        constInfo.syncV1C1);  // 反向同步 1
                }
            }
            continue;
        }
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx); // TODO: isEnd重复
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
            uint32_t extraLoop = isEnd ? LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1 : 0;  // 只preload一轮

            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopEnd + extraLoop); s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            constInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        constInfo.gS1Start = 0;
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

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          QLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE;
    QLICommon::RunInfo &curRunInfo = runInfo[curTaskId];
    QLICommon::RunInfo &lastRunInfo = runInfo[1 - curTaskId];

    CalcRunInfo(loop, s2LoopIdx, curRunInfo);

    if (curRunInfo.isValid) {
        if ASCEND_IS_AIC {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            CrossCoreWaitFlag(constInfo.syncV1C1);  // 反向同步 1
            matmulService.ComputeMm1(curRunInfo);
            CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            if (curRunInfo.isLastS2InnerLoop) {
                CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);  // 反向同步 0
            }
        } else {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncC1V0);  // 反向同步 0
                vectorService.ProcessVec0(curRunInfo);
                CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }

    if (lastRunInfo.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            vectorService.ProcessVec1(lastRunInfo);
            CrossCoreSetFlag<QLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);  // 反向同步 1
        }
        lastRunInfo.isValid = false;
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessDecode()
{
    // TODO: 当前LD暂时关闭
    return;

    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (true) {  // isLD
            vectorService.ProcessLD();
        }
    }
}
}  // namespace QLIKernel
#endif  // quant_lightning_indexer_KERNEL_H