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
 * \file sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"
#include "sparse_flash_attention_service_cube_mla.h"
#include "sparse_flash_attention_service_vector_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U;
    uint32_t s2LoopTimes = 0U;
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    bool curActSeqLenIsZero = false;
    int32_t nextTokensPerBatch = 0;

    uint64_t actS1Size = 1ULL;
    uint32_t tndCoreStartKVSplitPos;
    bool tndIsS2SplitCore;

    uint32_t gS1Idx = 0U;
    uint64_t mBasicSizeTail = 0U;
};

template <typename SFAT> class SparseFlashAttentionMla {
public:
    using T = float;
    using Q_T = typename SFAT::queryType;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline SparseFlashAttentionMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const SparseFlashAttentionTilingDataMla *__restrict tiling,
				                __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SFA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = SFAVectorService<SFAT>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const SparseFlashAttentionTilingDataMla *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};

    SFAMatmulService<SFAT> matmulService;
    SFAVectorService<SFAT> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // ================================Init functions===================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcSinnerTopKBegin(RunInfo &info, uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &info);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    __aicore__ inline void GetPreNextTokensLeftUp();
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::InitTilingData()
{
    usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
    constInfo.splitKVNum = tilingData->splitKVParams.s2;
    constInfo.mmResUbSize = tilingData->singleCoreTensorSize.mmResUbSize;
    constInfo.bmm2ResUbSize = tilingData->singleCoreTensorSize.bmm2ResUbSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize * msdIterNum;

    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvSeqSize = tilingData->baseParams.seqSize;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->baseParams.blockSize;
    constInfo.outputLayout = static_cast<SFA_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    constInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimRope = headDimRope;
    constInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    constInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    constInfo.sparseMode = tilingData->baseParams.sparseMode;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename SFAT>
__aicore__ inline void
SparseFlashAttentionMla<SFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                __gm__ uint8_t *actualSeqLengths)
{
    constInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    constInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, constInfo.actualLenDimsKV);
    }
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if (constInfo.outputLayout == SFA_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim +
                                    n2Idx * constInfo.gSize * headDim;
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    } else if (constInfo.outputLayout == SFA_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                    s1Idx * kvHeadNum * constInfo.gSize * headDim +
                                    n2Idx * constInfo.gSize * headDim;
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum);  // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                            uint32_t n2Idx)
{
    if (tempLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-tempLoopInfo.nextTokensPerBatch)) {
        tempLoopInfo.curActualSeqLen = 0;
        return;
    }
    int64_t threshold = tempLoopInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + s1Idx + 1;
    }

    tempLoopInfo.curActualSeqLen = (constInfo.sparseBlockCount * constInfo.sparseBlockSize > threshold) ?
                                           threshold :
                                           constInfo.sparseBlockCount * constInfo.sparseBlockSize;
}

template <typename SFAT>
__aicore__ inline uint32_t SparseFlashAttentionMla<SFAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsKV == 0) {
            return constInfo.kvSeqSize;
        } else if (constInfo.actualLenDimsKV == 1) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return actualSeqLengthsKVGm.GetValue(bIdx);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetPreNextTokensLeftUp()
{
    if (constInfo.sparseMode == 3) {
        tempLoopInfo.nextTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.curActualSeqLenOri) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.curActualSeqLen == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                                  uint32_t s1Idx, bool isEnd)
{ 
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken = Min(s1Offset + tempLoopInfo.nextTokensPerBatch + s1BaseSize,tempLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(constInfo.sparseBlockSize * constInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::Init(__gm__ uint8_t *query,
                       __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                       __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                       __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                       const SparseFlashAttentionTilingDataMla *__restrict tiling,
                       __gm__ uint8_t *gmTiling, TPipe *tPipe)
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
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    InitCalcParamsEach();
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);
    qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if ASCEND_IS_AIV {
        if (constInfo.needInit && LAYOUT_T != SFA_LAYOUT::TND) {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);

    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // s2  d+rope bufNum
        kvMergeGm_.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + aiCoreIdx * 512 * 576 * 4 * sizeof(KV_T)));
        offset += GetBlockNum() * 512 * 576 * 4 * sizeof(KV_T);

        kvValidSizeGm_.SetGlobalBuffer(
            (__gm__ int32_t *)(workspace + offset + (aiCoreIdx * 2) * 128 * 4 * sizeof(int32_t)));
    }

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitMm2ResInt32GmGlobalTensor(mm2ResInt32Gm);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            vectorService.InitVec0GlobalTensor(kvValidSizeGm_, kvMergeGm_, kRopeGm, keyGm, blockTableGm);
        }
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm,
                                           actualSeqLengthsKVGm, lseMaxFdGm, lseSumFdGm, topKGm);
        vectorService.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        matmulService.InitPageAttentionInfo(kvMergeGm_, blockTableGm, topKGm,
                                            constInfo.kvCacheBlockSize, constInfo.maxBlockNumPerBatch);
    }
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::InitCalcParamsEach()
{
    uint32_t totalBaseNum = 0;
	uint32_t s1GBaseSize = constInfo.gSize;
	uint32_t actBatchS2 = 1;
	uint32_t coreNum = GetBlockNum();
    uint32_t currCoreIdx = aiCoreIdx;
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
		uint32_t actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        if (actBatchS1 < constInfo.qSeqSize) {
            constInfo.needInit = true;
        }
        totalBaseNum += actBatchS1*actBatchS2 ;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    }else {
        usedCoreNum = totalBaseNum;
    }
    if(aiCoreIdx>=usedCoreNum){
        return;
    }
	uint32_t accumBaseNum = 0;
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1=0;
    bool setStart=false;
	targetBaseNum = (currCoreIdx + 1) * avgBaseNum;
    uint32_t targetStartBaseNum = targetBaseNum-avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kvHeadNum; bN2Idx++) { 
        uint32_t bIdx = bN2Idx / constInfo.kvHeadNum;
		actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if(!setStart && accumBaseNum >= targetStartBaseNum){
                constInfo.bN2Start = bN2Idx;
                constInfo.gS1Start = s1GIdx;
                setStart=true;
            }
            if (accumBaseNum >= targetBaseNum) {
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                constInfo.s2End = 0;
                constInfo.coreStartKVSplitPos = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
                }
                return;
			}
		}
		if ((actBatchS1 > 0) && (actBatchS2 > 0)) {
            lastValidBIdx = bIdx;
            lastValidactBatchS1 = actBatchS1;
        }
    }
    if (!setStart){
        constInfo.bN2Start = lastValidBIdx;
        constInfo.gS1Start = lastValidactBatchS1-1;
    }
    if (accumBaseNum < targetBaseNum) {
		constInfo.bN2End = lastValidBIdx;
        constInfo.gS1End = lastValidactBatchS1-1;
        constInfo.s2End = 0;
        constInfo.coreStartKVSplitPos = 0;
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename SFAT>
__aicore__ inline void
SparseFlashAttentionMla<SFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (SFAVectorService<SFAT>::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}


template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                                 uint32_t s2LoopIdx, RunInfo &info)
{
    info.loop = loop;
    info.bIdx = tempLoopInfo.bIdx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;

    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;

    info.actS1Size = tempLoopInfo.actS1Size;
    
    
    info.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        info.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;

    if ASCEND_IS_AIV {
        info.mSize = info.actMBaseSize;
        info.mSizeV = (info.mSize <= 16) ? info.mSize : (((info.mSize + 15) / 16 + 1) / 2 * 16);
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isChangeBatch = false;

    info.isFirstSInnerLoop = s2LoopIdx == s2Start;
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = s2LoopIdx == tempLoopInfo.s2LoopTimes - 1;
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;
    uint64_t actualSeqQPrefixSum;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
    }
    info.tndBIdxOffsetForQ = actualSeqQPrefixSum * constInfo.qHeadNum * headDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.kvSeqSize;
    }
    info.tndBIdxOffsetForKV = actualSeqKVPrefixSum * constInfo.kvHeadNum * headDim;

    if (info.isFirstSInnerLoop) {
        uint64_t tndBIdxRopeOffsetForQ = actualSeqQPrefixSum * constInfo.qHeadNum * headDimRope;
        tensorACoreOffset = info.tndBIdxOffsetForQ + info.gS1Idx * headDim;
        tensorARopeCoreOffset = tndBIdxRopeOffsetForQ + info.gS1Idx * headDimRope;
        
        uint64_t tndBIdxRopeOffsetForK = actualSeqKVPrefixSum * constInfo.kvHeadNum * headDimRope;
        tensorBCoreOffset = info.tndBIdxOffsetForKV + info.n2Idx * headDim;
        tensorBRopeCoreOffset = tndBIdxRopeOffsetForK + info.n2Idx * headDimRope;
        if (constInfo.sparseMode == 3) {
            threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + info.gS1Idx / constInfo.gSize + 1;
        } else {
            threshold = tempLoopInfo.curActualSeqLenOri;
        }
        if constexpr(LAYOUT_T == SFA_LAYOUT::BSND) {     // B,S1,N2 K
            topKBaseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == SFA_LAYOUT::TND) {        // T N2 K
            topKBaseOffset = info.tndBIdxOffsetForQ / constInfo.gSize / constInfo.headDim * constInfo.kvHeadNum *
                             constInfo.sparseBlockCount + info.n2Idx * constInfo.sparseBlockCount +
                             info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount;
        } else {                                         // B N2 S1 K
            topKBaseOffset = info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
        }
    }
    info.topKBaseOffset = topKBaseOffset;
    info.threshold = threshold;
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    info.tensorBOffset = tensorBCoreOffset;
    info.tensorBRopeOffset = tensorBRopeCoreOffset;
    info.attenOutOffset = tensorACoreOffset;

    uint64_t sInnerOffsetDataSize = info.s2Idx * constInfo.s2BaseSize;
    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;

    info.curActualSeqLenOri = tempLoopInfo.curActualSeqLenOri;
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (tempLoopInfo.curActualSeqLen > sInnerOffsetDataSize) {
            info.actualSingleProcessSInnerSize = tempLoopInfo.curActualSeqLen - sInnerOffsetDataSize;
            info.actualSingleProcessSInnerSize = info.actualSingleProcessSInnerSize > constInfo.s2BaseSize ?
                                                constInfo.s2BaseSize : info.actualSingleProcessSInnerSize;
            info.actualSingleProcessSInnerSize =
                SFAAlign((int64_t)info.actualSingleProcessSInnerSize, (int64_t)constInfo.sparseBlockSize);
        } else {
            info.actualSingleProcessSInnerSize = 0;
        }
        info.actualSingleProcessSInnerSizeAlign =
            SFAAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)SFAVectorService<SFAT>::BYTE_BLOCK);
    }
    
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        matmulService.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            vectorService.InitSoftmaxDefaultBuffer();
        } else {
            matmulService.AllocEventID();
        }
        ProcessBalance();

        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                                uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename SFAT> __aicore__ inline void SparseFlashAttentionMla<SFAT>::ProcessBalance()
{
    RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx <= constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        GetActualSeqLen(tempLoopInfo.bIdx);
        GetPreNextTokensLeftUp();
        if (tempLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End) ? constInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            GetSparseActualSeqLen(tempLoopInfo.bIdx, gS1LoopIdx, tempLoopInfo.n2Idx);
            UpdateInnerLoopCond();

            if (tempLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, gS1LoopIdx, tempLoopInfo.n2Idx);
            }
            int s2SplitNum =
                (tempLoopInfo.curActualSeqLen + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            bool isEnd = (bN2LoopIdx == constInfo.bN2End) && (gS1LoopIdx == constInfo.gS1End);
            tempLoopInfo.s2LoopTimes = s2SplitNum;
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == 0) && (tempLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;

            uint32_t curTopKIdx = 0;
            uint64_t curOffsetInSparseBlock = 0;
            for (int s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopTimes + extraLoop); s2LoopIdx++) {
                PreloadPipeline(gloop, constInfo.s2Start, s2LoopIdx, extraInfo, curTopKIdx, curOffsetInSparseBlock);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(constInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
        }
    }
}

template <typename SFAT>
__aicore__ inline void
SparseFlashAttentionMla<SFAT>::PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                                               RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock)
{
    RunInfo &extraInfo0 = extraInfo[loop % SFA_PRELOAD_TASK_CACHE_SIZE];
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % SFA_PRELOAD_TASK_CACHE_SIZE];
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % SFA_PRELOAD_TASK_CACHE_SIZE];

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);
    CalcSinnerTopKBegin(extraInfo0, curTopKIdx, curOffsetInSparseBlock);

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            ComputeMm1(extraInfo0);
        } else {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(3);
                vectorService.MergeKv(extraInfo0);
                CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            vectorService.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            vectorService.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename SFAT>
__aicore__ inline uint64_t
SparseFlashAttentionMla<SFAT>::GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths,
                                                                          uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsQ == 0) {
            return constInfo.qSeqSize;
        } else if (constInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                      uint32_t s1GEndPrev,
                                                                                      uint32_t s2EndPrev)
{
    uint32_t bEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
    uint32_t s1GPrevBaseNum = (actualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;
    
    constInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) {
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}

template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::CalcSinnerTopKBegin(RunInfo &info, uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock)

{
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        return;
    }
    
    uint64_t thresholdSparseCount = (info.threshold + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    uint64_t validCount = (constInfo.sparseBlockCount > thresholdSparseCount) ? thresholdSparseCount : constInfo.sparseBlockCount;

    int32_t sparseIndices = topKGm.GetValue(info.topKBaseOffset + curTopKIdx);
    if (sparseIndices == -1 || curTopKIdx == validCount) {
        info.actualSingleProcessSInnerSize = 0;
        info.actualSingleProcessSInnerSizeAlign = 0;
        tempLoopInfo.s2BasicSizeTail = 0;
        if (curTopKIdx == 0) {
            DealActSeqLenIsZero(info.bIdx, info.gS1Idx / constInfo.gSize, tempLoopInfo.n2Idx);
        }
        return;
    }

    uint32_t sparseLen = 0;
    uint64_t blockBegin = sparseIndices * constInfo.sparseBlockSize;
    uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > info.threshold) ? info.threshold : blockBegin + constInfo.sparseBlockSize;
    int32_t blockLen = blockEnd - blockBegin;
    sparseLen += (blockLen > static_cast<int32_t>(curOffsetInSparseBlock)) ? blockLen - curOffsetInSparseBlock : 0;

    bool firstVaildFlag = false;
    if (curTopKIdx > 0) {
        info.curTopKIdx = curTopKIdx;
        info.curOffsetInSparseBlock = curOffsetInSparseBlock;
    } else if (curTopKIdx == 0 && sparseLen > 0) {
        info.curTopKIdx = curTopKIdx;
        info.curOffsetInSparseBlock = 0;
        firstVaildFlag = true;
    }
    
    for (uint64_t topkIdx = curTopKIdx + 1; topkIdx < validCount; topkIdx++) {
        int32_t sparseIndices = topKGm.GetValue(info.topKBaseOffset + topkIdx);
        if (sparseIndices == -1) {
            curTopKIdx = topkIdx;
            curOffsetInSparseBlock = 0;
            break;
        }
        uint64_t blockBegin = sparseIndices * constInfo.sparseBlockSize;
        if (blockBegin >= info.threshold) {
            continue;
        }
        if (firstVaildFlag == false && curTopKIdx == 0) {
            info.curTopKIdx = topkIdx;
            info.curOffsetInSparseBlock = 0;
            firstVaildFlag = true;
        }
        uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > info.threshold) ? info.threshold : blockBegin + constInfo.sparseBlockSize;
        uint64_t blockLen = blockEnd - blockBegin;
        sparseLen += blockLen;
        if (sparseLen >= constInfo.s2BaseSize) {
            curTopKIdx = topkIdx;
            curOffsetInSparseBlock = blockLen - (sparseLen - constInfo.s2BaseSize);
            sparseLen = constInfo.s2BaseSize;
            break;
        }

        if (topkIdx == validCount - 1) {
            curTopKIdx = validCount;
            curOffsetInSparseBlock = 0;
        }
    }

    info.actualSingleProcessSInnerSize = sparseLen;
    info.actualSingleProcessSInnerSizeAlign = SFAAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)SFAVectorService<SFAT>::BYTE_BLOCK);
    tempLoopInfo.s2BasicSizeTail = (sparseLen == constInfo.s2BaseSize) ? 0 : sparseLen;
    if (curTopKIdx == 0 && sparseLen == 0) {
        DealActSeqLenIsZero(info.bIdx, info.gS1Idx / constInfo.gSize, tempLoopInfo.n2Idx);
    }
}
#endif // SPARSE_FLASH_ATTENTION_KERNEL_MLA_H