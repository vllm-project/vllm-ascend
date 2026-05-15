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
 * \file sparse_attn_sharedkv_swa_kernel.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_SWA_KERNEL_H
#define SPARSE_ATTN_SHAREDKV_SWA_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../sparse_attn_sharedkv_common.h"
#include "sparse_attn_sharedkv_swa_block_cube.h"
#include "sparse_attn_sharedkv_swa_block_vector.h"
#include "../sparse_attn_sharedkv_metadata.h"

namespace SASKernel {
using namespace matmul;
using namespace optiling;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct SwaTempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1

    int32_t actS1Size = 0; // TND场景下当前Batch循环处理的S1轴的大小
    int32_t actOriS2Size = 0;
    int32_t actCmpS2Size = 0;

    bool curActSeqLenIsZero = false;

    uint32_t tndCoreStartKVSplitPos = 0;
    bool tndIsS2SplitCore = false;

    uint32_t gS1Idx = 0U;
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小
    uint32_t cmpLoopTimes = 0;
    uint32_t oriLoopTimes = 0;

    int32_t oriMaskRight = 0;
    int32_t oriMaskLeft = 0;
    int32_t cmpMaskRight = 0;

    uint64_t actualSeqQPrefixSum = 0;
    uint64_t actualSeqKVPrefixSum = 0;
    uint64_t actualSeqCmpKVPrefixSum = 0;
};

template <typename SAST>
class SparseAttnSharedkvSwa {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using SINKS_T = float;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline SparseAttnSharedkvSwa(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensKV, __gm__ uint8_t *cuSeqlensCmpKV,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, 
                                const SparseAttnSharedkvTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SAS_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const SparseAttnSharedkvTilingData *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;
    GlobalTensor<uint32_t> metadataGm;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorCmpBCoreOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    ConstInfo constInfo{};
    SwaTempLoopInfo tempLoopInfo{};

    SWACubeBlock<SAST> cubeBlock;
    SWAVectorBlock<SAST> vectorBlock;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;
    GlobalTensor<SINKS_T> sinksGm;

    GlobalTensor<OUT_T> attentionOutGm;

    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;

    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKv);
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKV,
                                            __gm__ uint8_t *actualSeqLengthsCmpKV);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE]);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetSparseActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void CalcParams(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start, uint32_t s2LoopIdx,
                                      RunInfo &info);
    __aicore__ inline int32_t GetActualSeqLenQ(uint32_t bIdx);
    __aicore__ inline int32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitTilingData()
{
    // singleCoreParams
    // singleCoreTensorSize
    constInfo.mmResUbSize = tilingData->baseParams.mmResUbSize;
    constInfo.bmm2ResUbSize = tilingData->baseParams.bmm2ResUbSize;
    // baseParams
    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvSeqSize = tilingData->baseParams.kvSeqSize;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.oriMaxBlockNumPerBatch = tilingData->baseParams.oriMaxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->baseParams.paBlockSize;

    constInfo.paOriBlockSize = tilingData->baseParams.oriBlockSize;
    constInfo.paCmpBlockSize = tilingData->baseParams.cmpBlockSize;
    constInfo.outputLayout = static_cast<SAS_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.oriMaskMode = tilingData->baseParams.oriMaskMode;
    constInfo.oriWinLeft = tilingData->baseParams.oriWinLeft;
    constInfo.oriWinRight = tilingData->baseParams.oriWinRight;

    constInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    constInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;

    // innerSplitParams
    constInfo.mBaseSize = constInfo.gSize;   // tilingData->baseParams.mBaseSize
    constInfo.s2BaseSize = tilingData->baseParams.s2BaseSize;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.templateMode = TEMPLATE_MODE;

    // cmp
    if (constInfo.templateMode == CFA_TEMPLATE) {
        constInfo.cmpRatio = tilingData->cmpParams.cmpRatio;
        constInfo.cmpMaskMode = tilingData->cmpParams.cmpMaskMode;
        constInfo.cmpMaxBlockNumPerBatch = tilingData->cmpParams.cmpMaxBlockNumPerBatch;
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorBlock.InitBuffers(pipe);
    } else {
        cubeBlock.InitBuffers(pipe);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                     __gm__ uint8_t *actualSeqLengthsKv)
{
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKv, constInfo.actualLenDimsKV);
    }
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
}

template <typename SAST>
__aicore__ inline void
SparseAttnSharedkvSwa<SAST>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKV,
                                               __gm__ uint8_t *actualSeqLengthsCmpKV)
{
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKV, constInfo.actualLenDimsKV);
        if (constInfo.templateMode == CFA_TEMPLATE) {
            actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsCmpKV, constInfo.actualLenDimsKV);
        } 
    }
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                  uint32_t n2Idx)
{
    if (constInfo.outputLayout == SAS_LAYOUT::TND) {
        if (tempLoopInfo.actS1Size == 0) {
            return;
        }
        uint32_t tBase = actualSeqLengthsQGm.GetValue(bIdx);
        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim + // T轴、s1轴偏移
                                  n2Idx * constInfo.gSize * headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    } else if (constInfo.outputLayout == SAS_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                  s1Idx * kvHeadNum * constInfo.gSize * headDim + // B轴、S1轴偏移
                                  n2Idx * constInfo.gSize * headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum); // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename SAST>
__aicore__ inline int32_t SparseAttnSharedkvSwa<SAST>::GetActualSeqLenQ(uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        int32_t actualSeqQPrefixSum = actualSeqLengthsQGm.GetValue(bIdx);
        int32_t actualSeqQNextSum = actualSeqLengthsQGm.GetValue(bIdx + 1);
        tempLoopInfo.actualSeqQPrefixSum = static_cast<uint64_t>(actualSeqQPrefixSum);
        return actualSeqQNextSum - actualSeqQPrefixSum;
    } else {
        tempLoopInfo.actualSeqQPrefixSum = static_cast<uint64_t>(bIdx * constInfo.qSeqSize);
        if (constInfo.actualLenDimsQ == 0) {
            return static_cast<int32_t>(constInfo.qSeqSize);
        } else {
            return actualSeqLengthsQGm.GetValue(bIdx);
        }
    }
}

template <typename SAST>
__aicore__ inline int32_t SparseAttnSharedkvSwa<SAST>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        int32_t actualSeqKVPrefixSum = actualSeqLengthsKVGm.GetValue(bIdx);
        int32_t actualSeqKVNextSum = actualSeqLengthsKVGm.GetValue(bIdx + 1);
        if (constInfo.templateMode == CFA_TEMPLATE) {
            tempLoopInfo.actualSeqCmpKVPrefixSum = actualSeqLengthsCmpKVGm.GetValue(bIdx);
        } 
        tempLoopInfo.actualSeqKVPrefixSum = actualSeqKVPrefixSum;
        return actualSeqKVNextSum - actualSeqKVPrefixSum;
    } else {
        tempLoopInfo.actualSeqKVPrefixSum = static_cast<uint64_t>(bIdx * constInfo.kvSeqSize);
        if (constInfo.actualLenDimsKV == 0) {
            return static_cast<int32_t>(constInfo.kvSeqSize);
        } else {
            return actualSeqLengthsKVGm.GetValue(bIdx);
        }
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::GetSparseActualSeqLen()
{
    // 行无效通过ori部分判断, ori部分如果有行无效那么ori和cmp都有
    if (static_cast<int32_t>(tempLoopInfo.s1EndIdx) < -(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size)) {
        tempLoopInfo.actOriS2Size = 0;
        tempLoopInfo.actCmpS2Size = 0;
        return;
    }

    // 对于cmp部分还有top k, tempLoopInfo.actS2Size只针对cmp
    if (constInfo.templateMode == CFA_TEMPLATE) {
        int32_t thresHold = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;
        tempLoopInfo.actCmpS2Size = thresHold;
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.actCmpS2Size == 0 && tempLoopInfo.actOriS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::Init(
                                __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensKV, __gm__ uint8_t *cuSeqlensCmpKV,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const SparseAttnSharedkvTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    if (KV_LAYOUT_T == SAS_LAYOUT::TND && LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(cuSeqlensQ, cuSeqlensKV, cuSeqlensCmpKV);
    } else if (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(seqUsedQ, cuSeqlensKV, cuSeqlensCmpKV);
    } else if (KV_LAYOUT_T == SAS_LAYOUT::PA_ND && LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(cuSeqlensQ, seqUsedKV);
    } else if (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
        InitActualSeqLen(seqUsedQ, seqUsedKV);
    }
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    InitCalcParamsEach();

    pipe = tPipe;
    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    oriKvGm.SetGlobalBuffer((__gm__ KV_T *)oriKV);
    if (constInfo.templateMode == CFA_TEMPLATE) {
        cmpKvGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ SINKS_T *)sinks);
    }

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if ASCEND_IS_AIV {
        if (LAYOUT_T != SAS_LAYOUT::TND) {
            if (constInfo.needInit) {
                InitOutputSingleCore();
            }
        }
    }

    if constexpr (PAGE_ATTENTION) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        if (constInfo.templateMode == CFA_TEMPLATE) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
    }

    // workspace 内存排布
    // |Q--|mm1ResGm|vec1ResGm|mm2ResGm|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ Q_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    vec2ResGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T);

    if ASCEND_IS_AIV {
        vectorBlock.InitParams(constInfo, tilingData);
        vectorBlock.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm, actualSeqLengthsKVGm, sinksGm);
        vectorBlock.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        cubeBlock.InitParams(constInfo);
        cubeBlock.InitMm1GlobalTensor(queryGm, oriKvGm, cmpKvGm, mm1ResGm);
        cubeBlock.InitMm2GlobalTensor(vec1ResGm, mm2ResGm, attentionOutGm);
        cubeBlock.InitPageAttentionInfo(oriKvGm, oriBlockTableGm, cmpBlockTableGm);
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitCalcParamsEach()
{
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_END_INDEX, false));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_END_INDEX, false));
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::CalcParams(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start,
                                                               uint32_t s2LoopIdx, RunInfo &info)
{
    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;
    info.loop = loop;
    info.cmpLoop = cmpLoop;
    info.bIdx = tempLoopInfo.bIdx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;
    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;
    info.actS1Size = tempLoopInfo.actS1Size;

    // M方向的尾块
    info.actMBaseSize = tempLoopInfo.mBasicSizeTail;

    if ASCEND_IS_AIV {
        info.mSize = info.actMBaseSize;
        info.mSizeV = (info.mSize <= 16) ? info.mSize : ((CeilDiv(info.mSize, 16) + 1) / 2 * 16);
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isFirstSInnerLoop = s2LoopIdx == s2Start;
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = (s2LoopIdx == (tempLoopInfo.s2LoopTimes - 1));
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;

    uint64_t tndBIdxOffsetForQ = tempLoopInfo.actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
    uint64_t tndBIdxOffsetForKV = tempLoopInfo.actualSeqKVPrefixSum * constInfo.kvHeadNum * constInfo.headDim;
    uint64_t tndBIdxOffsetForCmpKV = tempLoopInfo.actualSeqCmpKVPrefixSum * constInfo.kvHeadNum * constInfo.headDim;

    if (info.isFirstSInnerLoop) {
        tensorACoreOffset = tndBIdxOffsetForQ + info.gS1Idx * constInfo.headDim;
        tensorBCoreOffset = tndBIdxOffsetForKV + info.n2Idx * constInfo.headDim; 
        tensorCmpBCoreOffset = tndBIdxOffsetForCmpKV + info.n2Idx * constInfo.headDim;
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorBOffset = tensorBCoreOffset; 
    info.tensorCmpBOffset = tensorCmpBCoreOffset;
    info.attenOutOffset = tensorACoreOffset;

    if (s2LoopIdx < tempLoopInfo.oriLoopTimes) {
        // S2首次循环只能在ori_kv
        info.isOri = true;
        info.relativeS2Idx = 0;
        uint64_t s2Offset = info.s2Idx * constInfo.s2BaseSize;
        if (s2LoopIdx + 1 == tempLoopInfo.oriLoopTimes) {
            info.actualSingleProcessSInnerSize = (tempLoopInfo.oriMaskRight - tempLoopInfo.oriMaskLeft + 1) - s2Offset;
        } else {
            info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
        }
        info.s2StartPoint = tempLoopInfo.oriMaskLeft;
        info.cmpS2IdLimit = 0;
    } else {
        if (constInfo.templateMode == CFA_TEMPLATE) {
            info.isOri = false;
            info.relativeS2Idx = info.s2Idx - tempLoopInfo.oriLoopTimes;
            uint64_t s2Offset = (info.s2Idx - tempLoopInfo.oriLoopTimes) * constInfo.s2BaseSize;
            if (s2LoopIdx + 1 == tempLoopInfo.s2LoopTimes) {
                info.actualSingleProcessSInnerSize = tempLoopInfo.actCmpS2Size - s2Offset;
            } else {
                info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
            }
            info.s2StartPoint = 0;
            info.cmpS2IdLimit = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;
        }
    }

    info.actualSingleProcessSInnerSizeAlign =
        SASAlign(info.actualSingleProcessSInnerSize, SASVectorBlock<SAST>::BYTE_BLOCK);
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = CeilDiv(info.actMBaseSize, constInfo.nBufferMBaseSize);
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        cubeBlock.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        cubeBlock.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::Process()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_CORE_ENABLE_INDEX, false));
    if (hasLoad == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorBlock.AllocEventID();
        vectorBlock.InitSoftmaxDefaultBuffer();
    } else {
        cubeBlock.AllocEventID();
    }
    ProcessBalance();
    if ASCEND_IS_AIV {
        vectorBlock.FreeEventID();
    } else {
        cubeBlock.FreeEventID();
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::ProcessBalance()
{
    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    uint32_t cmpLoop = 0;
    uint32_t gS1LoopEnd = 0;
    bool globalLoopStart = true;
    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)) {
        constInfo.bN2End += 1;
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        tempLoopInfo.actS1Size = GetActualSeqLenQ(tempLoopInfo.bIdx); // 获取actualSeqLength
        bool isS1ZeroAndLastBatch = (tempLoopInfo.actS1Size == 0) && ((constInfo.outputLayout == SAS_LAYOUT::BSND) ||
                                                                      (bN2LoopIdx + 1 == constInfo.bN2End));
        uint32_t gS1SplitNum = CeilDiv(tempLoopInfo.actS1Size * constInfo.gSize, constInfo.mBaseSize);

        // 当处于最后一个BN2时, 且gS1End为0时, 说明当前BN2里的所有数据都在当前核处理
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End - 1 && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
        // 当处于最后一个BN2且当前S1为0时，需要进入循环计算preload导致的未完成的部分
        gS1LoopEnd = isS1ZeroAndLastBatch ? gS1LoopEnd + 1 : gS1LoopEnd;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.actOriS2Size = GetActualSeqLenKV(tempLoopInfo.bIdx);
            // 对于各轴上的真实的idx, 采用左闭右闭的方案
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            tempLoopInfo.s1StartIdx = tempLoopInfo.gS1Idx / constInfo.gSize;
            tempLoopInfo.s1EndIdx =
                Min((tempLoopInfo.s1StartIdx + constInfo.mBaseSize / constInfo.gSize - 1), tempLoopInfo.actS1Size - 1);
            // 此处均为闭区间
            tempLoopInfo.oriMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                        static_cast<int32_t>(tempLoopInfo.s1EndIdx) + constInfo.oriWinRight;
            tempLoopInfo.oriMaskLeft = Max(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                               static_cast<int32_t>(tempLoopInfo.s1EndIdx) - constInfo.oriWinLeft,
                                           0);
            if (constInfo.templateMode == CFA_TEMPLATE) {
                tempLoopInfo.cmpMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size;
            }
            GetSparseActualSeqLen();
            UpdateInnerLoopCond();
            uint32_t oriSplitNum = 0;
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == gS1LoopEnd);
            if (tempLoopInfo.curActSeqLenIsZero) {
                if ASCEND_IS_AIV {
                    InitAllZeroOutput(tempLoopInfo.bIdx, tempLoopInfo.s1StartIdx, tempLoopInfo.n2Idx);
                }
                if (!isEnd) {
                    continue;
                }
            } else {
                oriSplitNum = CeilDiv(tempLoopInfo.oriMaskRight - tempLoopInfo.oriMaskLeft + 1, constInfo.s2BaseSize);
            }
            uint32_t s2SplitNum = oriSplitNum;
            if (constInfo.templateMode == CFA_TEMPLATE) {
                uint32_t cmpSplitNum = CeilDiv(tempLoopInfo.actCmpS2Size, constInfo.s2BaseSize);
                s2SplitNum = oriSplitNum + cmpSplitNum;
                tempLoopInfo.cmpLoopTimes = cmpSplitNum;
            }
            tempLoopInfo.s2LoopTimes = s2SplitNum;
            tempLoopInfo.oriLoopTimes = oriSplitNum;
            uint32_t s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : tempLoopInfo.s2LoopTimes;
            tempLoopInfo.s2LoopTimes = s2LoopEnd;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore = ((constInfo.s2Start == 0) && (s2LoopEnd == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? PRELOAD_NUM : 0;

            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (s2LoopEnd + extraLoop); s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, cmpLoop, constInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
}

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::PreloadPipeline(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start,
                                                                    uint64_t s2LoopIdx,
                                                                    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % SAS_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, cmpLoop, s2Start, s2LoopIdx, extraInfo0);
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}
} // namespace SASKernel
#endif // SPARSE_ATTN_SHAREDKV_SWA_KERNEL_H