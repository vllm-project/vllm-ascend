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
 * \file fused_sparse_attention_overlap_kernel_mla.h
 * \brief
 */

#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_MLA_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_sparse_attention_overlap_common.h"
#include "fused_sparse_attention_overlap_service_cube_mla.h"
#include "fused_sparse_attention_overlap_service_vector_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计�?
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大�?
    uint32_t s2LoopTimes = 0U;     // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    bool curActSeqLenIsZero = false;
    int32_t nextTokensPerBatch = 0;

    uint64_t actS1Size = 1ULL;     // TND场景下当前Batch循环处理的S1轴的大小
    uint32_t tndCoreStartKVSplitPos;
    bool tndIsS2SplitCore;

    uint32_t gS1Idx = 0U;
    uint64_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大�?
};

template <typename FusedSparseAttentionOverlapTraits> class FusedSparseAttentionOverlapMla {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename FusedSparseAttentionOverlapTraits::queryType;
    using KV_T = typename FusedSparseAttentionOverlapTraits::kvType;
    using OUT_T = typename FusedSparseAttentionOverlapTraits::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline FusedSparseAttentionOverlapMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FusedSparseAttentionOverlapTilingDataMla *__restrict tiling,
				                __gm__ uint8_t *gmTiling, TPipe *tPipe);
    __aicore__ inline void InitSelectionUpdateGlobalTensor(__gm__ uint8_t *selectionKRope,
                                __gm__ uint8_t *selectionKvCache,
                                __gm__ uint8_t *selectionKvBlockTable,
                                __gm__ uint8_t *selectionKvBlockStatus,
                                __gm__ uint8_t *selectionMembershipMap,
                                __gm__ uint8_t *selectionKvActualSeq,
                                bool enableSelectionUpdate);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = FusedSparseAttentionOverlapTraits::pageAttention;
    static constexpr int TEMPLATE_MODE = FusedSparseAttentionOverlapTraits::templateMode;
    static constexpr bool FLASH_DECODE = FusedSparseAttentionOverlapTraits::flashDecode;
    static constexpr FusedSparseAttentionOverlapLayout LAYOUT_T = FusedSparseAttentionOverlapTraits::layout;
    static constexpr FusedSparseAttentionOverlapLayout KV_LAYOUT_T = FusedSparseAttentionOverlapTraits::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const FusedSparseAttentionOverlapTilingDataMla *__restrict tilingData = nullptr;

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
    // helper：空闲 AICore 的 V0/V1 帮搬 MergeKv（仅 V_TEMPLATE 启用）
    bool isHelperAiCore_ = false;
    uint32_t numHelpers_ = 0U;          // 参与帮忙的空闲核数（0 = 功能整体关闭）
    uint32_t helperTargetAiCoreIdx_ = 0U;
    uint32_t helperPartIdx_ = 0U;
    uint32_t helperPartCount_ = 1U;
    // ⚠ 波次泛化与尾段 splitKV 已于 2026-09-03 实测结案后整体移除（hit 高端
    //    负载下净负 5~6%，台账第十四~十七节有完整证据链）。复活路径 =
    //    git revert 本次删除提交。
    // ⚠ 这里曾经放过两个成员（fdAttenOutOffset / fdActMBaseSize），存归约要用的
    //    输出位置和行数。12 字节，FD=0 时也照样占着类的空间 —— 实测 bs=16
    //    全命中端退化 4.4%（+4.7 μs），而那两个值 FD=0 时一次都没被写过。
    //    现在改成从 tensorACoreOffset 和 tempLoopInfo 现算，一个成员都不留。
    //    同类教训见优化 05：两个 int64_t[32] 挂成成员，退化 32.7%。

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};

    FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits> matmulService;
    FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits> vectorService;

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
                                           RunInfo extraInfo[FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock);
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
    __aicore__ inline void ReduceSplitKVOut();
};

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitTilingData()
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
    constInfo.outputLayout = static_cast<FusedSparseAttentionOverlapLayout>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    constInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimRope = headDimRope;
    constInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    constInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    constInfo.selectionBlockTableStride = tilingData->baseParams.selectionBlockTableStride;
    constInfo.selectionStatusStride = tilingData->baseParams.selectionStatusStride;
    constInfo.selectionMembershipStride = tilingData->baseParams.selectionMembershipStride;
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

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if (constInfo.outputLayout == FusedSparseAttentionOverlapLayout::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim +   // T轴、s1轴偏�?
                                    n2Idx * constInfo.gSize * headDim;                        // N2轴偏�?
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    } else if (constInfo.outputLayout == FusedSparseAttentionOverlapLayout::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                    s1Idx * kvHeadNum * constInfo.gSize * headDim +           // B轴、S1轴偏�?
                                    n2Idx * constInfo.gSize * headDim;                        // N2轴偏�?
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitOutputSingleCore()
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                            uint32_t n2Idx)
{
    if (tempLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-tempLoopInfo.nextTokensPerBatch)) { //存在行无�?
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline uint32_t FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetPreNextTokensLeftUp()
{
    if (constInfo.sparseMode == 3) {
        tempLoopInfo.nextTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.curActualSeqLenOri) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::UpdateInnerLoopCond()
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                                  uint32_t s1Idx, bool isEnd)
{ 
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken = Min(s1Offset + tempLoopInfo.nextTokensPerBatch + s1BaseSize,tempLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(constInfo.sparseBlockSize * constInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitSelectionUpdateGlobalTensor(
    __gm__ uint8_t *selectionKRope, __gm__ uint8_t *selectionKvCache,
    __gm__ uint8_t *selectionKvBlockTable, __gm__ uint8_t *selectionKvBlockStatus,
    __gm__ uint8_t *selectionMembershipMap, __gm__ uint8_t *selectionKvActualSeq,
    bool enableSelectionUpdate)
{
    if (!enableSelectionUpdate) {
        return;
    }

    GlobalTensor<KV_T> selectionKRopeGm;
    GlobalTensor<KV_T> selectionKvCacheGm;
    GlobalTensor<int32_t> selectionKvBlockTableGm;
    GlobalTensor<int32_t> selectionKvBlockStatusGm;
    GlobalTensor<int16_t> selectionMembershipMapGm;
    GlobalTensor<int32_t> selectionKvActualSeqGm;
    selectionKRopeGm.SetGlobalBuffer((__gm__ KV_T *)selectionKRope);
    selectionKvCacheGm.SetGlobalBuffer((__gm__ KV_T *)selectionKvCache);
    selectionKvBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvBlockTable);
    selectionKvBlockStatusGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvBlockStatus);
    selectionMembershipMapGm.SetGlobalBuffer((__gm__ int16_t *)selectionMembershipMap);
    selectionKvActualSeqGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvActualSeq);

    int64_t selectionKvBlockSize = constInfo.kvCacheBlockSize;
    if (selectionKvBlockSize <= 0) {
        return;
    }
    int64_t selectionMaxBlockNum = constInfo.selectionBlockTableStride;
    if (selectionMaxBlockNum <= 0) {
        selectionMaxBlockNum =
            (static_cast<int64_t>(constInfo.sparseBlockCount) + selectionKvBlockSize - 1) /
            selectionKvBlockSize;
    }
    vectorService.InitSelectionUpdateGlobalTensor(selectionKRopeGm, selectionKvCacheGm,
        selectionKvBlockTableGm, selectionKvBlockStatusGm, selectionMembershipMapGm,
        selectionKvActualSeqGm, selectionKvBlockSize, selectionMaxBlockNum, 1,
        constInfo.selectionStatusStride, constInfo.selectionMembershipStride, true);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::Init(__gm__ uint8_t *query,
                       __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                       __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                       __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                       const FusedSparseAttentionOverlapTilingDataMla *__restrict tiling,
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

    // 初始化计算参�?
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
        if (constInfo.needInit && LAYOUT_T != FusedSparseAttentionOverlapLayout::TND) {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);

    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
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
        // ⚠ 这一行原来没有。它上面每一个 SetGlobalBuffer 之后都推进了 offset，
        // 只有 kvValidSizeGm_ 没推 —— FLASH_DECODE=1 时下面的 accumOutGm 就
        // 直接盖在它身上，两块缓冲完全重叠。FD=1 从没编译过，所以没人发现。
        // ⚠ FD=0 时 offset 在这之后没人再用，这一行对现有路径无影响。
        offset += GetBlockNum() * 2 * 128 * 4 * sizeof(int32_t);
        // helper 握手区独立选址在 validSize 区之后（取址见 InitHelperSync 处），
        // 这里把 offset 推过它：FD=1 若哪天恢复，accumOut 才不会盖到握手区上。
        offset += GetBlockNum() * 2 * 64 * 8 * sizeof(int32_t);
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
            if (numHelpers_ > 0) {
                // 握手中转区独立选址：放到全部 merge 缓冲 + validSize 区之后
                //（host 已按 2×coreNum×64×8×4B 扩容 workspace）——原先借"第一个
                // 空闲核的 merge 槽"，独立区免除与 busy merge 写的一切别名顾虑。
                // ring 按 pair 隔离：done/credit 各 numHelpers_ 个 ring（每 ring
                // 64 槽×32B），消除共享 ring 的跨 pair 误信用隐患。
                constexpr uint32_t syncRing =
                    FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::HELPER_SYNC_RING;
                constexpr uint32_t syncSlot =
                    FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::HELPER_SYNC_SLOT_INT32;
                const uint64_t perCoreMergeBytes = 512ULL * 576 * 4 * sizeof(KV_T);
                __gm__ uint8_t *kvMergeArena = workspace +
                    GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T) +
                    GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T) +
                    GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
                uint32_t helperBaseCore = usedCoreNum;
                __gm__ int32_t *syncBase = (__gm__ int32_t *)(kvMergeArena +
                        (uint64_t)GetBlockNum() * perCoreMergeBytes +
                        (uint64_t)GetBlockNum() * 2 * 128 * 4 * sizeof(int32_t));
                GlobalTensor<int32_t> mergeDoneGm;
                GlobalTensor<int32_t> mergeCreditGm;
                mergeDoneGm.SetGlobalBuffer(syncBase);
                mergeCreditGm.SetGlobalBuffer(syncBase +
                    (uint64_t)numHelpers_ * syncRing * syncSlot);
                vectorService.InitHelperSync(mergeDoneGm, mergeCreditGm, numHelpers_);
                if (isHelperAiCore_) {
                    // helper 直接写 target 核的 merge 缓冲
                    GlobalTensor<KV_T> targetKvMergeGm;
                    targetKvMergeGm.SetGlobalBuffer((__gm__ KV_T *)(kvMergeArena +
                        (uint64_t)helperTargetAiCoreIdx_ * perCoreMergeBytes));
                    vectorService.InitHelperV0(targetKvMergeGm, helperTargetAiCoreIdx_);
                } else if (helperPartCount_ > 1) {
                    // target：helper 核号 = helperBaseCore + 本核号（一对一），
                    // 回收它的部分有效计数要读它的 kvValidSizeGm_（紧跟 kvMerge 区之后）。
                    GlobalTensor<int32_t> helperValidSizeGm;
                    helperValidSizeGm.SetGlobalBuffer((__gm__ int32_t *)(kvMergeArena +
                        GetBlockNum() * perCoreMergeBytes +
                        (uint64_t)(helperBaseCore + aiCoreIdx) * 2 * 128 * 4 * sizeof(int32_t)));
                    vectorService.SetHelperParts(helperPartIdx_, helperPartCount_, helperValidSizeGm,
                                                 aiCoreIdx);
                }
            }
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
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
    // 全核屏障（功能代码，非 timer）：原属 timer 初始化块，timerAddr 恒非空时每次 launch 都执行。
    // helper 握手区（mergeDone/mergeCredit GM 槽）跨 launch 的干净状态依赖它——SignalMergeDone
    // 的原子加发出后不等落地核即继续，缺少本屏障时上一 launch 慢核的在飞写与本 launch
    // ZeroHelperSyncArea 的清零失去串行化，done 槽残留计数会让 WaitMergeDone 提前满足，
    // target 在 helper 搬完前读 kvMerge，输出错乱（剥离 timer 后精度 bs=2/4/8 挂的根因）。
    // 收窄（2026-09-01）：只在 helper 参与的 launch 才屏障。握手区的写
    //（SignalMergeDone/credit）与清（ZeroHelperSyncArea）都只发生在
    // numHelpers_>0 的 launch；无 helper 的 launch 不碰握手区，屏障是纯开销。
    // 危险对（helper launch 的在飞写 × 后续 helper launch 的清零）由后一个
    // helper launch 的本屏障兜住。numHelpers_ 由同一组 GM 输入算出、全核
    // 同值，屏障不会分叉。
    if (numHelpers_ > 0U) {
        AscendC::SyncAll<false>();
    }
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::InitCalcParamsEach()
{
    //计算总的基本�?
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
    uint32_t kvSplitNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    }else {
        usedCoreNum = totalBaseNum;
        // splitKV：核没用满时，把每个任务的 KV 再切成 kvSplitNum 段交给空转的核，
        // 各算一份局部结果，Process() 末尾再归约成一份。
        if constexpr (FLASH_DECODE) {
            uint32_t roomPerTask = (totalBaseNum > 0) ? (coreNum / totalBaseNum) : 1;
            kvSplitNum = constInfo.splitKVNum;   // host 给的上限
            if (kvSplitNum > roomPerTask) {
                kvSplitNum = roomPerTask;
            }
            if (kvSplitNum < 1) {
                kvSplitNum = 1;
            }
            usedCoreNum = totalBaseNum * kvSplitNum;
        }
    }
    // ⚠ 要在下面那个 return 之前写回：不干活的核也得知道切了几段，归约要用
    constInfo.splitKVNum = kvSplitNum;
    // helper：只在一对一覆盖全部活跃核时才启用 —— kernel 时间由最慢核决定，
    // 半覆盖时没 helper 的核是瓶颈，配上 helper 的核白付握手费（实测 bs=4/16
    // 劣化 2~8%）。bs=1/2 满足全覆盖（4+20 / 8+16），bs=4/16 回落原路径。
    if (usedCoreNum < coreNum && coreNum - usedCoreNum >= usedCoreNum) {
        numHelpers_ = usedCoreNum;
    }
    if(aiCoreIdx>=usedCoreNum){
        // helper：前 numHelpers_ 个空闲核各帮一个活跃核
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            uint32_t helperIdx = aiCoreIdx - usedCoreNum;
            if (helperIdx >= numHelpers_) {
                return; // 多余的空闲核，照旧空转
            }
            isHelperAiCore_ = true;
            helperTargetAiCoreIdx_ = helperIdx; // 与 target 一一对应
            currCoreIdx = helperTargetAiCoreIdx_ / kvSplitNum; // 与 target 同一个任务
            helperPartIdx_ = 1;
            helperPartCount_ = 2;
        } else {
            return;
        }
    } else {
        // 本核负责第几个任务
        currCoreIdx = aiCoreIdx / kvSplitNum;
        // 前 numHelpers_ 个活跃核各配了一个 helper
        if (aiCoreIdx < numHelpers_) {
            helperPartIdx_ = 0;
            helperPartCount_ = 2;
        }
    }
    // 局部结果存 workspace 的第几格。总格号 = 任务号 * 段数 + 段号，而
    // 任务号 = aiCoreIdx / k、段号 = aiCoreIdx % k，两下相乘再相加恰好还原成
    // aiCoreIdx —— 于是每个核写自己那一格，天然不撞车，CalcAccumOffset 保持 0。
    // 归约时任务 t 的 k 份就躺在核号 [t*k, t*k+k) 这段连续格子里。
    constInfo.coreStartKVSplitPos = aiCoreIdx;
	//计算当前核的基本�?
	uint32_t accumBaseNum = 0;                       // 当前累积的基本块�?
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1=0;
    bool setStart=false;
	targetBaseNum = (currCoreIdx + 1) * avgBaseNum;  // 计算当前的目标权�?
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
                // 更新当前核的End分核信息
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                constInfo.s2End = 0;
                // ⚠ 切了段就不能清零 —— 这个值是本核的 workspace 格号，归约靠它定位
                constInfo.coreStartKVSplitPos = (kvSplitNum > 1) ? aiCoreIdx : 0;
                // ⚠ 判据是任务号不是核号：同一个任务的 k 个核必须拿到相同的起点
                if (currCoreIdx != 0) {
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
		// 更新最后一个核的End分核信息
		constInfo.bN2End = lastValidBIdx;
        constInfo.gS1End = lastValidactBatchS1-1;
        constInfo.s2End = 0;
        constInfo.coreStartKVSplitPos = (kvSplitNum > 1) ? aiCoreIdx : 0;
        if (currCoreIdx != 0) {
            GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}


template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::CalcParams(uint32_t loop, uint64_t s2Start,
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
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
    }
    info.tndBIdxOffsetForQ = actualSeqQPrefixSum * constInfo.qHeadNum * headDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
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
        if constexpr(LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND) {     // B,S1,N2 K
            topKBaseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {        // T N2 K
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
    //计算实际基本块size
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (tempLoopInfo.curActualSeqLen > sInnerOffsetDataSize) {
            info.actualSingleProcessSInnerSize = tempLoopInfo.curActualSeqLen - sInnerOffsetDataSize;
            info.actualSingleProcessSInnerSize = info.actualSingleProcessSInnerSize > constInfo.s2BaseSize ?
                                                constInfo.s2BaseSize : info.actualSingleProcessSInnerSize;
            info.actualSingleProcessSInnerSize =
                FusedSparseAttentionOverlapAlign((int64_t)info.actualSingleProcessSInnerSize, (int64_t)constInfo.sparseBlockSize);
        } else {
            info.actualSingleProcessSInnerSize = 0;
        }
        info.actualSingleProcessSInnerSizeAlign =
            FusedSparseAttentionOverlapAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BYTE_BLOCK);
    }
    
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        matmulService.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::Process()
{
    if ASCEND_IS_AIV {
        vectorService.RunAllCoreSelectionUpdate();
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            if (numHelpers_ > 0 && aiCoreIdx == 0 && GetSubBlockIdx() == 0) {
                // 清零 helper 握手区：走 MTE3，SyncAll 保证全核可见（标量写不可靠）
                vectorService.ZeroHelperSyncArea();
            }
        }
    }
    SyncAll<false>();

    // 事件开阖 + softmax 模板每 launch 一次：ProcessBalance 结束时缓冲旗标回到
    // +1 平衡态（否则 FreeEventID 的 WaitFlag 会挂死），softmax 模板内只读，
    // 多任务单调用即如此。
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
    } else if (isHelperAiCore_) {
        // helper：只做 MergeKv 搬运，不进完整流水（helper 的 AIC 直接等核内流水外）
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            vectorService.InitSoftmaxDefaultBuffer();
            ProcessBalance();
            vectorService.FreeEventID();
        }
    }

    // splitKV：各段都算完了，把同一个请求的几份并成一份写出去。
    // ⚠ SyncAll 必须每个核都走到，所以放在上面那个闸门之外 —— 位置照抄本函数
    //    开头 RunAllCoreSelectionUpdate 之后那一处，那里本来就是这么放的。
    //    splitKVNum 全核一致，这个 if 要么全进要么全不进，不会有人卡在同步点上。
    if constexpr (FLASH_DECODE) {
        if (constInfo.splitKVNum > 1) {
            SyncAll<false>();
            if ASCEND_IS_AIV {
                ReduceSplitKVOut();
            }
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::ReduceSplitKVOut()
{
    uint32_t part = constInfo.splitKVNum;
    // 每一组段里的头一个核负责归约 —— 它手上正好有这个请求的输出位置和行数，
    // 而它那一组的 part 份局部结果就躺在核号 [aiCoreIdx, aiCoreIdx + part) 这几格里。
    if (aiCoreIdx >= usedCoreNum || (aiCoreIdx % part) != 0) {
        return;
    }
    // 行数按 CalcParams 里的同一套公式现算；输出位置直接用 tensorACoreOffset
    // ——它在本核最后一个任务的首个 S2 循环里已经填好，ProcessBalance 结束后还留着。
    uint32_t mCount = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        mCount = tempLoopInfo.mBasicSizeTail;
    }
    vectorService.ReduceSplitKV(aiCoreIdx, part, mCount, tensorACoreOffset);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                                uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::ProcessBalance()
{
    RunInfo extraInfo[FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE];
    // CCE 不保证栈上数组默认成员初始化（isValid=false）生效，显式初始化：
    // loop 0/1 会读 loop+1/+2 槽的 isValid，脏值会导致用脏 RunInfo 做 Vec1/Vec2
    for (uint32_t i = 0; i < FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE; i++) {
        extraInfo[i].isValid = false;
    }
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx <= constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        GetActualSeqLen(tempLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        if (tempLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End) ? constInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            GetSparseActualSeqLen(tempLoopInfo.bIdx, gS1LoopIdx, tempLoopInfo.n2Idx); // TopK值sparse完后的ActualSeqLengthKV
            UpdateInnerLoopCond();

            // helper 不写零输出（target 会写），只保证迭代序列一致
            if (tempLoopInfo.curActSeqLenIsZero && !isHelperAiCore_) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, gS1LoopIdx, tempLoopInfo.n2Idx);
            }
            int s2SplitNum =
                (tempLoopInfo.curActualSeqLen + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize; // S2切分份数
            bool isEnd = (bN2LoopIdx == constInfo.bN2End) && (gS1LoopIdx == constInfo.gS1End);
            tempLoopInfo.s2LoopTimes = s2SplitNum;
            // splitKV：把 [0, s2SplitNum) 这些块按段分给本任务的 k 个核。
            // ⚠ 下面那个 tndIsS2SplitCore 的表达式本来就是对的 —— 只要起点非 0
            //    或者终点不是全部，它自己就变 true。上游注释说"分核修改后需要打开"，
            //    其实要改的只是分核，判据一个字不用动。
            if constexpr (FLASH_DECODE) {
                if (constInfo.splitKVNum > 1) {
                    uint32_t part = constInfo.splitKVNum;
                    uint32_t perPart = ((uint32_t)s2SplitNum + part - 1) / part;
                    // ⚠ 这里要的是段号（0..k-1），不是 coreStartKVSplitPos 那个格号。
                    //    格号 = 任务号 * k + 段号，两者差着一个任务号，别混用。
                    uint32_t begin = (aiCoreIdx % part) * perPart;
                    uint32_t end = begin + perPart;
                    if (begin >= (uint32_t)s2SplitNum) {
                        // 段比块还多，本核一块都分不到。让它当"这一块没数据"跑一轮：
                        // lse 填中性值、accumOut 填零，否则归约端读到的是脏数据。
                        begin = (uint32_t)s2SplitNum;
                        end = begin + 1;
                    } else if (end > (uint32_t)s2SplitNum) {
                        end = (uint32_t)s2SplitNum;
                    }
                    constInfo.s2Start = begin;
                    tempLoopInfo.s2LoopTimes = end;
                }
            }
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut�?
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == 0) && (tempLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos =
                (globalLoopStart || constInfo.splitKVNum > 1) ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;

            uint32_t curTopKIdx = 0;
            uint64_t curOffsetInSparseBlock = 0;
            for (int s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopTimes + extraLoop); s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, constInfo.s2Start, s2LoopIdx, extraInfo, curTopKIdx, curOffsetInSparseBlock);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        // helper 的 AIC 不设这些旗，一个都不等
        if (!isHelperAiCore_) {
            CrossCoreWaitFlag(constInfo.syncC2V1);
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(3);
                CrossCoreWaitFlag(3);
                CrossCoreWaitFlag(3);
                CrossCoreWaitFlag(3);
            }
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                                               RunInfo extraInfo[FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock)
{
    RunInfo &extraInfo0 = extraInfo[loop % FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE];         // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任�?
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FUSED_SPARSE_ATTENTION_OVERLAP_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任�?

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
                if (isHelperAiCore_) {
                    // helper V0/V1：等 target 发布槽位信用 → 各搬 1/4 → 计数 +1
                    vectorService.WaitHelperCredit(extraInfo0.loop);
                    vectorService.MergeKv(extraInfo0);
                    vectorService.SignalMergeDone(extraInfo0.loop);
                    // helper 分担回写：处理自己搬的那部分 token 的 update 槽位
                    // （纯数据搬运，与 target 的 update 并行、不挡 done；状态标记
                    // 由 target 统一做；对应 helper-only 仓 d06cb9f0）
                    if (vectorService.IsSelectionUpdateEnabled()) {
                        vectorService.CopyOutSelectionUpdateFromKvMerge(extraInfo0);
                    }
                } else {
                    CrossCoreWaitFlag(3);
                    if (helperPartCount_ > 1) {
                        // 拿到 flag3 信用 = AIC 已读完这个槽，发布给 helper
                        vectorService.PublishHelperCredit(extraInfo0.loop);
                    }
                    vectorService.MergeKv(extraInfo0);
                    if (helperPartCount_ > 1) {
                        // 等 helper 两半都落地，并把 helper 的有效计数并回自己的 kvValidSizeGm_。
                        // ⚠ FixValidSize 必须在 setV0C1 之前：消费者在下一轮迭代的 Vec1L，
                        // 覆盖顺序的 PIPE_MTE3 旗标只有 syncV0C1 这一个，挪后就没有序了。
                        vectorService.WaitMergeDone(extraInfo0.loop);
                        vectorService.FixValidSize(extraInfo0);
                    }
                    CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
                    if (vectorService.IsSelectionUpdateEnabled()) {
                        vectorService.CopyOutSelectionUpdateFromKvMerge(extraInfo0);
                    }
                }
            }
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            if (!isHelperAiCore_) {
                vectorService.ProcessVec1L(extraInfo2);
            }
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            if (!isHelperAiCore_) {
                vectorService.ProcessVec2L(extraInfo1);
            }
        }
        extraInfo1.isValid = false;
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline uint64_t
FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths,
                                                                          uint32_t bIdx)
{
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
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

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                      uint32_t s1GEndPrev,
                                                                                      uint32_t s2EndPrev)
{
    uint32_t bEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
    uint32_t s1GPrevBaseNum = (actualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;
    
    constInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMla<FusedSparseAttentionOverlapTraits>::CalcSinnerTopKBegin(RunInfo &info, uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock)

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
    info.actualSingleProcessSInnerSizeAlign = FusedSparseAttentionOverlapAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BYTE_BLOCK);
    tempLoopInfo.s2BasicSizeTail = (sparseLen == constInfo.s2BaseSize) ? 0 : sparseLen;
    if (curTopKIdx == 0 && sparseLen == 0) {
        DealActSeqLenIsZero(info.bIdx, info.gS1Idx / constInfo.gSize, tempLoopInfo.n2Idx);
    }
}



#endif // FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_MLA_H
