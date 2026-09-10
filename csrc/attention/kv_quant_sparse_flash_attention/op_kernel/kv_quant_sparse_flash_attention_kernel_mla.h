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
 * \file kv_quant_sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_flash_attention_common.h"
#include "kv_quant_sparse_flash_attention_service_cube_mla.h"
#include "kv_quant_sparse_flash_attention_service_vector_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    uint64_t actS1Size = 1ULL; // TND场景下当前Batch循环处理的S1轴的大小，非TND场景下不要用这个字段
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小

    int32_t nextTokensPerBatch = 0;
    uint32_t tndCoreStartKVSplitPos;
    uint32_t gS1Idx = 0U;
    bool tndIsS2SplitCore;
    bool curActSeqLenIsZero = false;
};

template <typename QSFAT> class KvQuantSparseFlashAttentionMla {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename QSFAT::queryType;
    using KV_T = typename QSFAT::kvType;
    using OUT_T = typename QSFAT::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline KvQuantSparseFlashAttentionMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
                                __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
				                __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t QSFA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = QSFAVectorService<QSFAT>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};

    QSFAMatmulService<QSFAT> matmulService;
    QSFAVectorService<QSFAT> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE]);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &info);
    __aicore__ inline void CalcMSizeInfo(RunInfo &info);
    __aicore__ inline void CalcFirstTensorOffsets(RunInfo &info, uint64_t qsfaActualSeqQPrefixSum,
                                                  uint64_t actualSeqKVPrefixSum);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void GetPreNextTokensLeftUp();
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
};

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::InitTilingData()
{
    usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
    constInfo.splitKVNum = tilingData->splitKVParams.s2;
    constInfo.mmResUbSize = tilingData->singleCoreTensorSize.mmResUbSize;
    constInfo.bmm2ResUbSize = tilingData->singleCoreTensorSize.bmm2ResUbSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize * msdIterNum;

    constInfo.qHeadNum = constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.kvSeqSize = tilingData->baseParams.seqSize;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->baseParams.blockSize;
    constInfo.outputLayout = static_cast<QSFA_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    constInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimRope = headDimRope;
    constInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    constInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    constInfo.sparseMode = tilingData->baseParams.sparseMode;
    constInfo.returnSoftmaxLse = tilingData->baseParams.returnSoftmaxLse;
    constInfo.quantScaleRepoMode = QUANT_SCALE_REPO_MODE::COMBINE;
    constInfo.attentionMode = ATTENTION_MODE::MLA_ABSORB;
    constInfo.combineHeadDim = (constInfo.quantScaleRepoMode ==
        QUANT_SCALE_REPO_MODE::COMBINE) ? headDim + headDimRope : headDim;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    // constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QSFAT>
__aicore__ inline void
KvQuantSparseFlashAttentionMla<QSFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengths)
{
    constInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    constInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, constInfo.actualLenDimsKV);
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                  uint32_t n2Idx)
{
    if (constInfo.outputLayout == QSFA_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim +   // T轴、s1轴偏移
                                    n2Idx * constInfo.gSize * headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
        if (constInfo.returnSoftmaxLse) { //N2T1G
            uint64_t softmaxSumOffset = n2Idx * actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1) * constInfo.gSize +
                                        (tBase + s1Idx) * constInfo.gSize;
            uint64_t softmaxMaxOffset = softmaxSumOffset;
            matmul::InitOutput<T>(softmaxSumGm[softmaxSumOffset], constInfo.gSize, 0);
            matmul::InitOutput<T>(softmaxMaxGm[softmaxMaxOffset], constInfo.gSize, 0);
        }
    } else if (constInfo.outputLayout == QSFA_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                    s1Idx * kvHeadNum * constInfo.gSize * headDim + // B轴、S1轴偏移
                                    n2Idx * constInfo.gSize * headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * headDim, 0);
        if (constInfo.returnSoftmaxLse) { //B, N2, S1, G
            uint64_t softmaxSumOffset = bIdx * kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                                        n2Idx * constInfo.qSeqSize * constInfo.gSize +
                                        s1Idx * constInfo.gSize;
            uint64_t softmaxMaxOffset = softmaxSumOffset;
            matmul::InitOutput<T>(softmaxSumGm[softmaxSumOffset], constInfo.gSize, 0);
            matmul::InitOutput<T>(softmaxMaxGm[softmaxMaxOffset], constInfo.gSize, 0);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::InitOutputSingleCore()
{
    uint32_t qsfaCoreNum = GetBlockNum();
    if (qsfaCoreNum != 0) {
        uint64_t qsfaTotalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize *
            constInfo.headDim;
        // 2 means c:v = 1:2
        uint64_t qsfaSingleCoreSize = (qsfaTotalOutputSize + (2 * qsfaCoreNum) - 1) / (2 * qsfaCoreNum);
        uint64_t qsfaTailSize = qsfaTotalOutputSize - tmpBlockIdx * qsfaSingleCoreSize;
        uint64_t qsfaSingleInitOutputSize = qsfaTailSize < qsfaSingleCoreSize ? qsfaTailSize : qsfaSingleCoreSize;
        if (tmpBlockIdx * qsfaSingleCoreSize < qsfaTotalOutputSize && qsfaSingleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * qsfaSingleCoreSize], qsfaSingleInitOutputSize, 0);
        }
        if (constInfo.returnSoftmaxLse) {
            uint64_t totalReturnSoftmaxSize = constInfo.batchSize * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize;
            uint64_t singleCoreReturnSoftmaxSize = (totalReturnSoftmaxSize + (2 * qsfaCoreNum) - 1) / (2 * qsfaCoreNum);
            uint64_t tailReturnSoftmaxSize = totalReturnSoftmaxSize - tmpBlockIdx * singleCoreReturnSoftmaxSize;
            uint64_t singleInitReturnSoftmaxSize = tailReturnSoftmaxSize < singleCoreReturnSoftmaxSize ?
                                                   tailReturnSoftmaxSize : singleCoreReturnSoftmaxSize;
            if (tmpBlockIdx * singleCoreReturnSoftmaxSize < totalReturnSoftmaxSize && singleInitReturnSoftmaxSize > 0) {
                matmul::InitOutput<T>(softmaxSumGm[tmpBlockIdx * singleCoreReturnSoftmaxSize], singleInitReturnSoftmaxSize, 0);
                matmul::InitOutput<T>(softmaxMaxGm[tmpBlockIdx * singleCoreReturnSoftmaxSize], singleInitReturnSoftmaxSize, 0);
            }
        }
        SyncAll();
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                                      uint32_t n2Idx)
{
    if (tempLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-tempLoopInfo.nextTokensPerBatch)) { // 存在行无效
        tempLoopInfo.curActualSeqLen = 0;
        return;
    }
    int64_t threshold = tempLoopInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + s1Idx + 1;
    }
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        tempLoopInfo.curActualSeqLen = (constInfo.sparseBlockCount * constInfo.sparseBlockSize > threshold) ?
                                           threshold :
                                           constInfo.sparseBlockCount * constInfo.sparseBlockSize;
    } else {
        uint64_t topKBaseOffset = 0;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) { // B,S1,N2 K
            topKBaseOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.sparseBlockCount +
                             s1Idx * kvHeadNum * constInfo.sparseBlockCount + n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == QSFA_LAYOUT::TND) { // T N2 K
            uint64_t actualSeqQPrefixSum = (bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
            topKBaseOffset = actualSeqQPrefixSum * kvHeadNum * constInfo.sparseBlockCount +
                             s1Idx * kvHeadNum * constInfo.sparseBlockCount +
                             n2Idx * constInfo.sparseBlockCount;
        } else { // B N2 S1 K
            topKBaseOffset = bIdx * kvHeadNum * constInfo.qSeqSize * constInfo.sparseBlockCount +
                             n2Idx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                             s1Idx * constInfo.sparseBlockCount;
        }

        uint64_t sparseLen = 0;

        for (uint64_t topkIdx = 0; topkIdx < constInfo.sparseBlockCount; topkIdx++) {
            int32_t sparseIndices = topKGm.GetValue(topKBaseOffset + topkIdx);
            uint64_t blockBegin = sparseIndices * constInfo.sparseBlockSize;
            if (blockBegin >= threshold) {
                continue;
            }
            uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > tempLoopInfo.curActualSeqLenOri) ?
                                    tempLoopInfo.curActualSeqLenOri :
                                    blockBegin + constInfo.sparseBlockSize;
            uint64_t blockLen = (blockEnd <= threshold) ? blockEnd - blockBegin : threshold - blockBegin;
            sparseLen += blockLen;
        }
        tempLoopInfo.curActualSeqLen = sparseLen;
    }
}

template <typename QSFAT>
__aicore__ inline uint32_t KvQuantSparseFlashAttentionMla<QSFAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
        if (bIdx > 0) {
            int32_t curActualSeqLen = actualSeqLengthsKVGm.GetValue(bIdx);
            int32_t prevActualSeqLen = actualSeqLengthsKVGm.GetValue(bIdx - 1);
            return (curActualSeqLen >= prevActualSeqLen) ?
                static_cast<uint32_t>(curActualSeqLen - prevActualSeqLen) : 0U;
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

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx,
                                                                                    uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::GetPreNextTokensLeftUp()
{
    if (constInfo.sparseMode == 3) {
        tempLoopInfo.nextTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.curActualSeqLenOri) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.curActualSeqLen == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.curActualSeqLen % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                            uint32_t s1Idx, bool isEnd)
{
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken = Min(s1Offset + tempLoopInfo.nextTokensPerBatch + s1BaseSize, tempLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(constInfo.sparseBlockSize * constInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::Init(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
    __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIC {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    } else {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 初始化计算参数
    InitCalcParamsEach();
    keyPtr = key;
    valuePtr = value;
    pipe = tPipe;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);

    if ASCEND_IS_AIV {
        if (constInfo.needInit && LAYOUT_T != QSFA_LAYOUT::TND) {
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
    uint64_t qsfaOffset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + qsfaOffset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ K_ROPE_T *)(workspace + qsfaOffset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize *
                            sizeof(K_ROPE_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(K_ROPE_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + qsfaOffset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer((__gm__ T *)(workspace + qsfaOffset +
                              aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // s2  d+rope bufNum
        kvMergeGm_.SetGlobalBuffer((__gm__ K_ROPE_T *)(workspace + qsfaOffset + aiCoreIdx * 512 * 576 * 4 *
                                   sizeof(K_ROPE_T)));
        qsfaOffset += GetBlockNum() * 512 * 576 * 4 * sizeof(K_ROPE_T);

        kvValidSizeGm_.SetGlobalBuffer(
            (__gm__ int32_t *)(workspace + qsfaOffset + (aiCoreIdx * 2) * 128 * 4 * sizeof(int32_t)));
    }

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset));
        qsfaOffset = qsfaOffset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset) +
            tilingData->splitKVParams.logSumExpSize / 2);
        qsfaOffset = qsfaOffset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitMm2ResInt32GmGlobalTensor(mm2ResInt32Gm);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            vectorService.InitVec0GlobalTensor(kvValidSizeGm_, kvMergeGm_, kRopeGm, keyGm, blockTableGm);
        }
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm,
                                           actualSeqLengthsKVGm, lseMaxFdGm, lseSumFdGm, topKGm,
                                           softmaxMaxGm, softmaxSumGm);
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
}

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::InitCalcParamsEach()
{
    // 计算总的基本块
    uint32_t totalBaseNum = 0;
    uint32_t s1GBaseSize = constInfo.gSize;
    uint32_t actBatchS2 = 1;
    uint32_t coreNum = GetBlockNum();
    uint32_t actBatchS1 = 1;
    uint32_t currCoreIdx = aiCoreIdx;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        uint32_t actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        if (actBatchS1 < constInfo.qSeqSize) {
            constInfo.needInit = true;
        }
        totalBaseNum += actBatchS1 * actBatchS2 ;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
	// 计算当前核的基本块
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (currCoreIdx + 1) * avgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum-avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kvHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kvHeadNum;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if (!setStart && accumBaseNum >= targetStartBaseNum) {
                constInfo.bN2Start = bN2Idx;
                constInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                constInfo.coreStartKVSplitPos = 0;
                constInfo.s2End = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
                }
                return;
            }
        }
	    if ((actBatchS1 > 0) && (actBatchS2 > 0)) {
            lastValidactBatchS1 = actBatchS1;
            lastValidBIdx = bIdx;
        }
    }
    if (!setStart) {
        constInfo.bN2Start = lastValidBIdx;
        constInfo.gS1Start = lastValidactBatchS1 - 1;
    }
    if (accumBaseNum < targetBaseNum) {
		// 更新最后一个核的End分核信息
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

template <typename QSFAT>
__aicore__ inline void
KvQuantSparseFlashAttentionMla<QSFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                         uint32_t startRow, uint32_t dealRowCount,
                                                         uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (QSFAVectorService<QSFAT>::BYTE_BLOCK /
        sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}


template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                           uint32_t s2LoopIdx, RunInfo &info)
{
    info.loop = loop;
    info.bIdx = tempLoopInfo.bIdx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;

    info.isBmm2Output = false;
    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;

    info.actS1Size = tempLoopInfo.actS1Size;
    info.actS2Size = tempLoopInfo.curActualSeqLen;

    info.actMBaseSize = constInfo.mBaseSize;
    uint32_t qsfaRemainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (qsfaRemainedGS1Size <= constInfo.mBaseSize && qsfaRemainedGS1Size > 0) {
        info.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;
    CalcMSizeInfo(info);

    info.isChangeBatch = false;

    info.isFirstSInnerLoop = (s2LoopIdx == s2Start);
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = (s2LoopIdx == tempLoopInfo.s2LoopTimes - 1);
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;
    uint64_t qsfaActualSeqQPrefixSum;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        qsfaActualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
    } else {
        qsfaActualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
    }
    info.tndBIdxOffsetForQ = qsfaActualSeqQPrefixSum * constInfo.qHeadNum * constInfo.combineHeadDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqKVPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.kvSeqSize;
    }
    info.tndBIdxOffsetForKV = actualSeqKVPrefixSum * constInfo.kvHeadNum * constInfo.combineHeadDim;

    CalcFirstTensorOffsets(info, qsfaActualSeqQPrefixSum, actualSeqKVPrefixSum);

    uint64_t sInnerOffsetDataSize = info.s2Idx * constInfo.s2BaseSize;
    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;

    info.curActualSeqLenOri = tempLoopInfo.curActualSeqLenOri;
    if (tempLoopInfo.curActualSeqLen > sInnerOffsetDataSize) {
        info.actualSingleProcessSInnerSize = tempLoopInfo.curActualSeqLen - sInnerOffsetDataSize;
        info.actualSingleProcessSInnerSize = info.actualSingleProcessSInnerSize > constInfo.s2BaseSize ?
                                             constInfo.s2BaseSize : info.actualSingleProcessSInnerSize;
    } else {
        info.actualSingleProcessSInnerSize = 0;
    }
    info.actualSingleProcessSInnerSizeAlign =
        QSFAAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)QSFAVectorService<QSFAT>::BYTE_BLOCK);
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::CalcMSizeInfo(RunInfo &info)
{
    if ASCEND_IS_AIV {
        info.mSize = info.actMBaseSize;
        info.mSizeV = (info.mSize <= 16) ? \
            info.mSize : (((info.mSize + 15) / 16 + 1) / 2 * 16);
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::CalcFirstTensorOffsets(
    RunInfo &info, uint64_t qsfaActualSeqQPrefixSum, uint64_t actualSeqKVPrefixSum)
{
    if (info.isFirstSInnerLoop) {
        tensorACoreOffset = info.tndBIdxOffsetForQ + info.gS1Idx * constInfo.combineHeadDim;
        tensorBCoreOffset = info.tndBIdxOffsetForKV + info.n2Idx * constInfo.combineHeadDim;
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            attenOutOffset = (qsfaActualSeqQPrefixSum * constInfo.qHeadNum + info.gS1Idx) * headDim;
        } else {
            uint64_t tndBIdxRopeOffsetForQ = qsfaActualSeqQPrefixSum * constInfo.qHeadNum * headDimRope;
            tensorARopeCoreOffset = tndBIdxRopeOffsetForQ + info.gS1Idx * headDimRope;
            uint64_t tndBIdxRopeOffsetForK = actualSeqKVPrefixSum * constInfo.kvHeadNum * headDimRope;
            tensorBRopeCoreOffset = tndBIdxRopeOffsetForK + info.n2Idx * headDimRope;
            attenOutOffset = tensorACoreOffset;
        }
        if (constInfo.sparseMode == 3) {
            threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + info.gS1Idx / constInfo.gSize + 1;
        } else {
            threshold = tempLoopInfo.curActualSeqLenOri;
        }
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
            topKBaseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == QSFA_LAYOUT::TND) {
            topKBaseOffset = info.tndBIdxOffsetForQ / constInfo.gSize / constInfo.combineHeadDim *
                            constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            info.n2Idx * constInfo.sparseBlockCount +
                            info.gS1Idx / constInfo.gSize * constInfo.kvHeadNum * constInfo.sparseBlockCount;
        } else {
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
    info.attenOutOffset = attenOutOffset;
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        matmulService.ComputeMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        // CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIC {
            matmulService.AllocEventID();
        } else {
            vectorService.AllocEventID();
            vectorService.InitSoftmaxDefaultBuffer();
        }
        ProcessBalance();

        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        } else {
            vectorService.FreeEventID();
        }
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                          uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename QSFAT> __aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::ProcessBalance()
{
    RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        // CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    for (uint32_t qsfaBN2LoopIdx = constInfo.bN2Start; qsfaBN2LoopIdx <= constInfo.bN2End; qsfaBN2LoopIdx++) {
        GetBN2Idx(qsfaBN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        GetActualSeqLen(tempLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        if (tempLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        gS1LoopEnd = (qsfaBN2LoopIdx == constInfo.bN2End) ? constInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t qsfaGS1LoopIdx = constInfo.gS1Start; qsfaGS1LoopIdx <= gS1LoopEnd; qsfaGS1LoopIdx++) {
            tempLoopInfo.gS1Idx = qsfaGS1LoopIdx * constInfo.mBaseSize;
            // TopK值sparse完后的ActualSeqLengthKV
            GetSparseActualSeqLen(tempLoopInfo.bIdx, qsfaGS1LoopIdx, tempLoopInfo.n2Idx);
            UpdateInnerLoopCond();

            if (tempLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, qsfaGS1LoopIdx, tempLoopInfo.n2Idx);
            }
            int s2SplitNum =
                (tempLoopInfo.curActualSeqLen + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize; // S2切分份数
            bool qsfaIsEnd = (qsfaBN2LoopIdx == constInfo.bN2End) && (qsfaGS1LoopIdx == constInfo.gS1End);
            tempLoopInfo.s2LoopTimes = s2SplitNum;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == 0) && (tempLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t qsfaExtraLoop = qsfaIsEnd ? 2 : 0;
            for (int s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopTimes + qsfaExtraLoop);
                s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, constInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void
KvQuantSparseFlashAttentionMla<QSFAT>::PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                                         RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % QSFA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % QSFA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % QSFA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);

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
                CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
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
                CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
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

template <typename QSFAT>
__aicore__ inline uint64_t
KvQuantSparseFlashAttentionMla<QSFAT>::GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths,
                                                                    uint32_t bIdx)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        if (bIdx > 0) {
            int32_t curActualSeqLen = actualSeqLengths.GetValue(bIdx);
            int32_t prevActualSeqLen = actualSeqLengths.GetValue(bIdx - 1);
            return (curActualSeqLen >= prevActualSeqLen) ?
                static_cast<uint64_t>(curActualSeqLen - prevActualSeqLen) : 0ULL;
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else if (constInfo.actualLenDimsQ == 0) {
            return constInfo.qSeqSize;
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void KvQuantSparseFlashAttentionMla<QSFAT>::GetAxisStartIdx(uint32_t bN2EndPrev,
    uint32_t s1GEndPrev, uint32_t s2EndPrev)
{
    uint32_t qsfaBEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t qsfaActualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBEndPrev);
    uint32_t qsfaS1GPrevBaseNum =
        (qsfaActualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;

    constInfo.s2Start = 0;
    if (s1GEndPrev >= qsfaS1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
