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
 * \file quant_lightning_indexer_v2_kernel_arch22.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
#define QUANT_LIGHTNING_INDEXER_V2_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_service_vector_arch22.h"
#include "quant_lightning_indexer_v2_service_cube_arch22.h"
#include "../quant_lightning_indexer_v2_metadata.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
using namespace matmul;
using namespace optiling::detail;
using namespace optiling;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息
// 同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    bool isNeedLD = false;                // 该基本块是否需要LD
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
    // ---- candidate driven (mode=2, V1) ----
    uint32_t actS1RowBase = 0U;   // 本 batch 第 0 行在 candidate_block_length/candidate_topk_index 中的行号
    uint32_t maxCandLen = 0U;     // 本任务 (bN2,gS1) 内各 token 候选块数的最大值 L_max
    uint32_t maxCandTiles = 0U;   // uniform 循环上界 = ceil(L_max × 8 / 256)，V1 tile = 32 块
    bool candDriven = false;      // 本任务是否走候选驱动流水（mode=2 且传入长度且非 FD）
};

template <typename QLIV2T>
class QLIV2Preload {
public:
    __aicore__ inline QLIV2Preload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *candidateTopkIndex, __gm__ uint8_t *candidateBlockLength,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                __gm__ uint8_t *candidateTopkIndexOut, __gm__ uint8_t *candidateBlockLengthOut,
                                __gm__ uint8_t *workspace,
                                const QLIV2TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    using OUT_T = typename QLIV2T::outputType;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;

    using MM1_OUT_T = float;

    QLIV2Matmul<QLIV2T> matmulService;
    QLIV2Vector<QLIV2T> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    // 参照 v1: s1BaseSize 固定, mBaseSize = s1BaseSize * gSize (支持 gSize=32/64)
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S2_BASE_SIZE = 2048;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t LI_QUANT_PRELOAD_TASK_CACHE_SIZE = 2;
    // V1 候选驱动：一个候选 tile = 32 个候选块 × 8 token = 256 token（设计 §3.2）
    static constexpr uint32_t QLIV2_CAND_TILE_BLOCKS = 32;
    static constexpr uint32_t QLIV2_CAND_TILE_TOKENS = QLIV2_CAND_TILE_BLOCKS * 8;

    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t ELE_NUM_PER_BLOCK = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t candidateOutCoreOffset = 0ULL;
    uint64_t outputIdxOffsetCoreOffset = 0ULL;  // A15
    uint32_t coreZeroEnable = 1U;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    // mode=1（source）每行有效候选块数（DSL candidate_length 语义：min(ceil(V/8), candidateTopkBlocks)）
    GlobalTensor<int32_t> candidateBlockLengthOutGm;
    GlobalTensor<int32_t> candidateBlockLengthGm; // 每行有效候选块数 (可选输入)
    // D3a: 候选 K/scale staging（每个 AIC 私有：4 槽 × 256 token × 128B K + 4 槽 × 256 × 2B scale）
    GlobalTensor<int8_t> candStagingKeyGm;
    GlobalTensor<half> candStagingScaleGm;
    // D3b: 候选分数区（per token × maxTiles × 256 float；B1+B2 冒烟阶段不参与输出）
    GlobalTensor<float> candScoreGm;
    // v4: 候选 kscale 辅助区（每 tile 1KB 填充布局，任务奇偶双缓冲）
    GlobalTensor<half> candAuxScaleGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;  // A15: 每行输出索引偏移

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    bool hasCandidateBlockLength_ = false; // 主算子是否传入 candidate_block_length（D2/候选驱动前置）
    // v4: 候选任务序号（AIC/AIV 两侧同步递增，仅用于 W 双槽交替 loop%2）
    uint32_t candTaskSeq_ = 0U;

    QLIV2Common::ConstInfo constInfo{};
    QLIV2Common::LdSplitCoreInfo ldInfo{};
    TempLoopInfo tempLoopInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                            __gm__ uint8_t *cmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig,
                                                       uint32_t &validS2Len);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ProcessInvalid();
    __aicore__ inline void ProcessDecode();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, uint32_t cmpResiduaKLenDims,
                                                  bool isAccumSeq, GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                  GlobalTensor<uint32_t> &cmpResidualKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitTilingData(const QLIV2TilingData *__restrict tilingData)
{
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.keyStride0 = tilingData->keyStride0;                     // A11: key 0轴非连续
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.outputLayout = Q_LAYOUT_T; // 输出和输入形状一致
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.s1BaseSize = S1_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;

    // candidate (two-level topk)
    constInfo.candidateMode = tilingData->candidateMode;
    constInfo.candidateTopkBlocks = tilingData->candidateTopkBlocks;
    constInfo.candidateBlockSize = tilingData->candidateBlockSize;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                              __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                                              __gm__ uint8_t *cmpResidualK)
{
    // Q side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize + 1;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    } else {
        if (sequsedQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)sequsedQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    }
    // K side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize + 1;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    } else {
        if (sequsedK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)sequsedK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    }
    // cmpResidualK获取
    if (cmpResidualK != nullptr) {
        constInfo.cmpResiduaKLenDims = constInfo.batchSize;
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t *)cmpResidualK, constInfo.batchSize);
    } else {
        constInfo.cmpResiduaKLenDims = 0;
    }
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                                 GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                                 uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx);
    } else {
        // non-TND with seqused: length[i] = seqused[i]
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                    uint32_t cmpResiduaKLenDims, bool isAccumSeq,
                                                                    GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                                    GlobalTensor<uint32_t> &cmpResidualKGm,
                                                                    uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    uint32_t cmpResidualK; // 当前bidx对应的cmpResidualK
    if (cmpResiduaKLenDims == 0) {
        cmpResidualK = 0;
    } else {
        cmpResidualK = cmpResidualKGm.GetValue(bIdx);
    }
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio + cmpResidualK;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return (actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    } else {
        return (actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                 uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.cmpResiduaKLenDims,
                                       constInfo.isAccumSeqS2, actualSeqLengthsGm, cmpResidualKGm, constInfo.kSeqSize,
                                       constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;         // 真实使用的压缩后S2长度
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                         uint32_t actS2SizeOrig, uint32_t &validS2Len)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        validS2Len = 0;
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase =
        static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size); // 压缩前的validS2LenBase
    validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
                 static_cast<int32_t>(constInfo.cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetTotalBaseBlockNum()
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

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数,
// 剩余的部分前面的核每个核多处理一块
template <typename QLIV2T>
__aicore__ void inline QLIV2Preload<QLIV2T>::SplitCore()
{
    constInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_CORE_ENABLE_INDEX, false));
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_END_INDEX, false));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_END_INDEX, false));

    // 如果0核都没有启动，说明所有核都没启动
    coreZeroEnable = metadataGm.GetValue(GetAttrAbsIndex(0, QLI_V2_CORE_ENABLE_INDEX, false));

    // LD 第一个workspace的索引
    uint32_t ldFirstWorkSpaceIndex = GetAttrAbsIndex(aiCoreIdx, QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX, false);
    ldInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t vecCoreIdx = tmpBlockIdx; // 此时代表vectoe核数
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_CORE_ENABLE_INDEX, true);
        ldInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!ldInfo.isLdCoreEnable) {
            return;
        }

        // LD 参数信息
        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumINDEX = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_NUM_INDEX, true);

        ldInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
        ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
        ldInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        ldInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        ldInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumINDEX);
        ldInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        ldInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        // mode=1 长度输出：取本 batch 的可见长度参数（与 CalcGS1LoopParams 同源）
        GetS1S2ActualSeqLen(ldInfo.bIdx, ldInfo.actS1Size, ldInfo.actS2Size, ldInfo.actS2SizeOrig);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(ldInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : ldInfo.bIdx * constInfo.qSeqSize;
        }
        // 搬出Topk的初始偏移地址
        ldInfo.indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.n2Idx) * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.mIdx) * constInfo.s1BaseSize * constInfo.kHeadNum * constInfo.sparseCount;
        // mode=1: 块级输出 candidate_topk_index 的初始偏移地址（与上面同构，行宽换成 candidateTopkBlocks）
        ldInfo.candidateOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.candidateTopkBlocks +
            static_cast<uint64_t>(ldInfo.n2Idx) * constInfo.candidateTopkBlocks +
            static_cast<uint64_t>(ldInfo.mIdx) * constInfo.s1BaseSize * constInfo.kHeadNum *
                constInfo.candidateTopkBlocks;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx);
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
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                    static_cast<uint64_t>(n2Idx) * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
    __gm__ uint8_t *metadata, __gm__ uint8_t *candidateTopkIndex, __gm__ uint8_t *candidateBlockLength,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues, __gm__ uint8_t *candidateTopkIndexOut,
    __gm__ uint8_t *candidateBlockLengthOut, __gm__ uint8_t *workspace,
    const QLIV2TilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK);

    // 可选输入：每行有效候选块数。AIC/AIV 都持有同一 GM 指针（D2 两侧各自现算，天然一致）。
    if (candidateBlockLength != nullptr) {
        candidateBlockLengthGm.SetGlobalBuffer((__gm__ int32_t *)candidateBlockLength);
        hasCandidateBlockLength_ = true;
    }

    if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
        // 计算分核
        SplitCore();
    }

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    GlobalTensor<half> weightWorkspaceGm; // v1阶段处理w*scale后的结果
    uint64_t weightMemSize = BLOCK_CUBE * constInfo.mBaseSize * WS_DOUBLE * sizeof(half);
    weightWorkspaceGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * weightMemSize));
    offset += GetBlockNum() * weightMemSize;

    // ld流程需要ws大小: [aicnum, 2, s1BaseSize, topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    GlobalTensor<float> vec1ResGm; // 存放TopK计算中间结果
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    // mode=1（candidate source，候选块来源）+ LD（跨核归约）：块级 topk 分段结果的备份区。
    // 与 vec1ResGm 同构（[核][S2 段][S1 行][value+index 对]），只是每行槽位宽从 topk 换成
    // 2 × candidateTopkBlocks（块级累加器的对宽）。位置级那条链已验证，块级这条链原缺
    // 溢写与归约，跨核切分后只剩头段结果（块级输出全错）。
    GlobalTensor<float> blockVec1ResGm;
    if (constInfo.candidateMode == CANDIDATE_MODE_SOURCE) {
        uint64_t singleCoreBlkVec1Res = static_cast<uint64_t>(constInfo.s1BaseSize) * WS_DOUBLE * WS_DOUBLE *
                                        constInfo.candidateTopkBlocks * 2ULL * sizeof(float);
        blockVec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset += GetBlockNum() * singleCoreBlkVec1Res;
    }

    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    // D3a: 候选 staging 区（条件申请：仅在传入 candidate_block_length 时占 workspace）
    if (candidateBlockLength != nullptr) {
        constexpr uint64_t CAND_STAGING_KEY_BYTES =
            QLIV2Common::CAND_RING_SLOTS * 256ULL * HEAD_DIM;     // 8 槽 × 256 token × 128B
        constexpr uint64_t CAND_STAGING_SCALE_BYTES =
            QLIV2Common::CAND_RING_SLOTS * 32ULL * 32ULL;         // 8 槽 × 32 块 × 32B(含填充)
        candStagingKeyGm.SetGlobalBuffer((__gm__ int8_t *)(workspace + offset + aiCoreIdx * CAND_STAGING_KEY_BYTES));
        offset += static_cast<uint64_t>(GetBlockNum()) * CAND_STAGING_KEY_BYTES;
        candStagingScaleGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * CAND_STAGING_SCALE_BYTES));
        offset += static_cast<uint64_t>(GetBlockNum()) * CAND_STAGING_SCALE_BYTES;
        // D3b: 候选分数区（每任务 4 行 × 64 tile × 256 float；按任务奇偶双缓冲 = 512KB/核）
        constexpr uint64_t CAND_SCORE_BYTES = 2ULL * QLIV2Common::CAND_SCORE_TILES_PER_TASK *
                                              QLIV2Common::CAND_TILE_TOKENS * sizeof(float);
        candScoreGm.SetGlobalBuffer((__gm__ float *)(workspace + offset + aiCoreIdx * CAND_SCORE_BYTES));
        offset += static_cast<uint64_t>(GetBlockNum()) * CAND_SCORE_BYTES;
        // v4: 候选 kscale 辅助区（每任务 256 tile × 1KB；奇偶双缓冲 = 512KB/核）
        constexpr uint64_t CAND_AUX_SCALE_BYTES = 2ULL * QLIV2Common::CAND_SCORE_TILES_PER_TASK *
                                                  QLIV2Common::CAND_AUX_SCALE_HALVES_PER_TILE * sizeof(half);
        candAuxScaleGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * CAND_AUX_SCALE_BYTES));
        offset += static_cast<uint64_t>(GetBlockNum()) * CAND_AUX_SCALE_BYTES;
    }
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ half *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ half *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ half *)keyScale);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key); // D3a: AIV 候选 gather 需要 K 源
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        if (constInfo.candidateMode == CANDIDATE_MODE_CONSUMER && candidateTopkIndex != nullptr) {
            candidateTopkIndexInGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndex);
        }
        if (constInfo.candidateMode == CANDIDATE_MODE_SOURCE && candidateTopkIndexOut != nullptr) {
            candidateTopkIndexOutGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndexOut);
        }
        if (constInfo.candidateMode == CANDIDATE_MODE_SOURCE && candidateBlockLengthOut != nullptr) {
            candidateBlockLengthOutGm.SetGlobalBuffer((__gm__ int32_t *)candidateBlockLengthOut);
        }
        if (outputIdxOffset != nullptr) {  // A15: 使能 output_idx_offset (参照 arch35)
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t *)outputIdxOffset);
        }
        // A15: 有效标志须与 tensor 一并传给 vector (InitParams 已先值拷贝 constInfo, 此处再改标志不生效)
        bool outputIdxOffsetValid = (outputIdxOffset != nullptr);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecCandidateTensor(candidateTopkIndexInGm, candidateTopkIndexOutGm,
                                             candidateBlockLengthOutGm, (candidateBlockLengthOut != nullptr),
                                             outputIdxOffsetGm, outputIdxOffsetValid);
        // D3a: 候选 gather 使能 = mode=2 且传入长度张量（V1 探针：cube 暂不消费 staging）
        vectorService.InitVecCandidateGatherTensor(
            keyGm, candStagingKeyGm, candStagingScaleGm, candidateBlockLengthGm, candScoreGm, candAuxScaleGm,
            (constInfo.candidateMode == CANDIDATE_MODE_CONSUMER && hasCandidateBlockLength_));
        vectorService.InitVecWorkspaceTensor(weightWorkspaceGm, mm1ResGm, vec1ResGm, blockVec1ResGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
        // D3b: 候选 QK 的 K 源与分数区（B1+B2 冒烟阶段仅写不读）
        matmulService.InitCandMm1Tensor(candStagingKeyGm, candScoreGm);
    }
    InitBuffers();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
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
    if (constInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.validS2Len % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;

    // ---- D2: 每任务候选统计（V1 候选驱动前置）----
    // AIC 与 AIV 都从同一 candidate_block_length 现算，公式相同 ⇒ 循环参数天然一致，
    // 不需要 GM 预扫描区/跨核握手（设计文档 D2 的简化实现）。
    // 注意：本步只计算、不改变现有循环行为；D3 才用 maxCandTiles 替换 S2 内层游标。
    tempLoopInfo.maxCandLen = 0U;
    tempLoopInfo.maxCandTiles = 0U;
    tempLoopInfo.candDriven = false;
    if (constInfo.candidateMode == CANDIDATE_MODE_CONSUMER && hasCandidateBlockLength_ &&
        !tempLoopInfo.isNeedLD) {
        uint32_t taskTokens = tempLoopInfo.actMBaseSize / static_cast<uint32_t>(constInfo.gSize);
        uint32_t rowBase = tempLoopInfo.actS1RowBase + tempLoopInfo.gS1Idx * constInfo.s1BaseSize;
        uint32_t maxL = 0U;
        for (uint32_t t = 0U; t < taskTokens; t++) {
            int32_t len = candidateBlockLengthGm.GetValue(rowBase + t);
            uint32_t l = (len > 0) ? static_cast<uint32_t>(len) : 0U;
            maxL = (l > maxL) ? l : maxL;
        }
        if (maxL > constInfo.candidateTopkBlocks) {
            maxL = constInfo.candidateTopkBlocks;
        }
        tempLoopInfo.maxCandLen = maxL;
        // V1 候选 tile = 32 块 = 256 token；上界按任务内最大 L 取 uniform 值。
        tempLoopInfo.maxCandTiles =
            (maxL * constInfo.candidateBlockSize + QLIV2_CAND_TILE_TOKENS - 1U) / QLIV2_CAND_TILE_TOKENS;
        // maxL=0 的任务仍需一次输出初始化（空行 -1），D3 用 maxCandTiles==0 的特殊分支处理。
        tempLoopInfo.candDriven = true;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    // candidate_block_length / candidate_topk_index 的行号基址：
    // TND 用 cu_seqlens_q 前缀，BSND 用 bIdx × qSeqSize（与 candidateOutOffset 的布局一致）
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        tempLoopInfo.actS1RowBase =
            (constInfo.actualLenQDims == 0) ? 0U : actualSeqLengthsGmQ.GetValue(tempLoopInfo.bIdx);
    } else {
        tempLoopInfo.actS1RowBase = tempLoopInfo.bIdx * static_cast<uint32_t>(constInfo.qSeqSize);
    }
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd =
        (bN2LoopIdx + 1 == constInfo.bN2End && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                         QLIV2Common::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx < tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    // B5/v4: 候选任务的全量扫输出写回被抑制（结果由 RunCandTask 的分数消费产生）
    runInfo.suppressOut = tempLoopInfo.candDriven;
    if (runInfo.isNeedLD && s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }

    if (!runInfo.isValid) {
        return;
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
    runInfo.actualSingleProcessSInnerSizeAlign = QLIV2Common::Align((uint32_t)runInfo.actualSingleProcessSInnerSize,
                                                                    QLIV2Common::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == constInfo.s2Start;
    runInfo.isLastS2InnerLoop = (s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx);
        } else { // BSND
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
        candidateOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.candidateTopkBlocks +
                                 runInfo.n2Idx * constInfo.candidateTopkBlocks;
        // A15: output_idx_offset 布局 [行, N2] (每行每 key 头一个 int32), batch 级前缀
        outputIdxOffsetCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum + runInfo.n2Idx;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D, cu_seqlens_k
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx);
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
    runInfo.candidateOutOffset = candidateOutCoreOffset;
    runInfo.outputIdxOffsetCoreOffset = outputIdxOffsetCoreOffset;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Process()
{
    // 没有计算任务，直接清理输出
    if (coreZeroEnable == 0) {
        ProcessInvalid();
        return;
    }
    ProcessMain();

    ProcessDecode();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QLIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessMain()
{
    // 无任务核直接返回
    if (constInfo.coreEnable == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        // v4: 候选 ring 的初始令牌 —— FREE 预置 RING_SLOTS 个（单 id 计数信号量，
        // 一次 set 广播给两个 AIV，每消费 1 tile 回补 1 个）；WFREE 预置 2 个（W 双槽）。
        if (constInfo.candidateMode == QLIV2Common::CANDIDATE_MODE_CONSUMER && hasCandidateBlockLength_) {
            for (uint32_t s = 0U; s < QLIV2Common::CAND_RING_SLOTS; s++) {
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
                    QLIV2Common::CAND_FREE_FLAG);
            }
            for (uint32_t s = 0U; s < 2U; s++) {
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
                    QLIV2Common::CAND_WFREE_FLAG);
            }
        }
    }

    QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];

    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
            if (constInfo.s2End != 0) {
                constInfo.gS1End += 1;
            }
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)) {
        constInfo.bN2End += 1;
        if (constInfo.s2End != 0) {
            constInfo.gS1End += 1;
        }
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
                    CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        constInfo.syncV1C1); // 反向同步 1
                }
            }
            continue;
        }
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            if (tempLoopInfo.candDriven) {
                uint32_t taskTokens = tempLoopInfo.actMBaseSize / static_cast<uint32_t>(constInfo.gSize);
                // 候选任务共用的 RunInfo：loop 决定 W 槽位，tensorWeightsOffset 指向本行块的权重起点
                QLIV2Common::RunInfo candRun{};
                candRun.bIdx = tempLoopInfo.bIdx;
                candRun.n2Idx = tempLoopInfo.n2Idx;
                candRun.gS1Idx = tempLoopInfo.gS1Idx;
                // W 双槽按候选任务序号交替（AIC/AIV 同一序号）；不用 gloop，避免与全量扫的
                // runInfo 双缓冲槽 / 延迟 ProcessVec1 排空相互干扰。
                candRun.loop = candTaskSeq_++;
                candRun.candTask = true;
                candRun.actS1Size = tempLoopInfo.actS1Size;
                candRun.actS2Size = tempLoopInfo.actS2Size;
                candRun.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
                candRun.actMBaseSize = tempLoopInfo.actMBaseSize;
                candRun.actualSingleProcessSInnerSize = QLIV2_CAND_TILE_TOKENS;
                candRun.actualSingleProcessSInnerSizeAlign = QLIV2_CAND_TILE_TOKENS;
                candRun.tensorQueryOffset =
                    static_cast<uint64_t>(tempLoopInfo.actS1RowBase) * constInfo.qHeadNum * constInfo.headDim +
                    static_cast<uint64_t>(gS1LoopIdx) * constInfo.mBaseSize * constInfo.headDim;
                candRun.tensorWeightsOffset = static_cast<uint64_t>(tempLoopInfo.actS1RowBase) * constInfo.qHeadNum +
                                              static_cast<uint64_t>(tempLoopInfo.n2Idx) * constInfo.gSize;
                if ASCEND_IS_AIV {
                    // 跳过全量扫前，先排空上一任务延迟的 ProcessVec1（否则它的最终输出会丢失）。
                    // 延迟槽 = 上一次 ProcessBaseBlock 的槽位 = (gloop-1)%2。
                    QLIV2Common::RunInfo &pending =
                        runInfo[(gloop + LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1) %
                                LI_QUANT_PRELOAD_TASK_CACHE_SIZE];
                    if (pending.isValid) {
                        CrossCoreWaitFlag(constInfo.syncC1V1);
                        vectorService.ProcessVec1(pending);
                        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                            constInfo.syncV1C1); // 反向同步 1
                        pending.isValid = false;
                    }
                    // B4: 先算一次本任务的 W（w×qscale 广播），AIC 候选 QK 依赖它
                    vectorService.ProcessVec0(candRun);
                    CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        QLIV2Common::CAND_WREADY_FLAG);
                    uint32_t rowBase = tempLoopInfo.actS1RowBase + gS1LoopIdx * constInfo.s1BaseSize;
                    uint32_t s1RowBase = gS1LoopIdx * constInfo.s1BaseSize;
                    uint32_t indiceOutOffset = static_cast<uint32_t>(
                        static_cast<uint64_t>(rowBase) * constInfo.kHeadNum * constInfo.sparseCount +
                        static_cast<uint64_t>(tempLoopInfo.n2Idx) * constInfo.sparseCount);
                    uint32_t outputIdxOffsetBase = static_cast<uint32_t>(
                        static_cast<uint64_t>(rowBase) * constInfo.kHeadNum + tempLoopInfo.n2Idx);
                    // v4: gather 与分数消费按 token 交错 —— 先 gather token t 的全部 tile，
                    // 再消费 token t-2 的分数（退两拍，保证分数写有落地余量）；
                    // 末尾两个 token 入 pending 队列，跨任务/核尾延迟消费
                    vectorService.RunCandTask(
                        tempLoopInfo.bIdx, tempLoopInfo.n2Idx, rowBase, taskTokens, tempLoopInfo.maxCandTiles,
                        candRun.loop, s1RowBase, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size,
                        tempLoopInfo.actS2SizeOrig, indiceOutOffset, outputIdxOffsetBase);
                } else {
                    // B1+B2+B3+B4：等 W 就绪后，按 per-token M=32 从 staging 读 K 算 QK
                    CrossCoreWaitFlag(QLIV2Common::CAND_WREADY_FLAG);
                    matmulService.ComputeMm1Candidate(candRun, taskTokens, tempLoopInfo.maxCandTiles);
                    // v4: 本任务的 W 槽读完后回收（AIV 最多领先 2 个任务，不会覆盖正在读的槽）
                    CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
                        QLIV2Common::CAND_WFREE_FLAG);
                }
                // 候选任务跳过全量扫（结果已由候选路径写出）
                constInfo.s2Start = 0;
                continue;
            }
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
            uint32_t extraLoop = isEnd ? LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1 : 0; // 只preload一轮

            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopEnd + extraLoop);
                 s2LoopIdx++) {
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
        // v4 核尾排空：候选任务内最后一个 token 的分数消费延迟到这里，
        // 先等 AIC 主循环结束令牌（其分数写已推进/落地），再读分数写输出。
        if (constInfo.candidateMode == QLIV2Common::CANDIDATE_MODE_CONSUMER && hasCandidateBlockLength_) {
            CrossCoreWaitFlag(QLIV2Common::CAND_AIC_DONE_FLAG);
            vectorService.DrainPendingCandScore();
        }
        vectorService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
        if (constInfo.candidateMode == QLIV2Common::CANDIDATE_MODE_CONSUMER && hasCandidateBlockLength_) {
            CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
                QLIV2Common::CAND_AIC_DONE_FLAG);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessBaseBlock(
    uint32_t loop, uint64_t s2LoopIdx, QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE;
    QLIV2Common::RunInfo &curRunInfo = runInfo[curTaskId];
    QLIV2Common::RunInfo &lastRunInfo = runInfo[1 - curTaskId];

    CalcRunInfo(loop, s2LoopIdx, curRunInfo);

    if (curRunInfo.isValid) {
        if ASCEND_IS_AIC {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            CrossCoreWaitFlag(constInfo.syncV1C1); // 反向同步 1
            matmulService.ComputeMm1(curRunInfo);
            CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            if (curRunInfo.isLastS2InnerLoop) {
                // 反向同步 0
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
            }
        } else {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncC1V0); // 反向同步 0
                vectorService.ProcessVec0(curRunInfo);
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }

    if (lastRunInfo.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            vectorService.ProcessVec1(lastRunInfo);
            CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1); // 反向同步 1
        }
        lastRunInfo.isValid = false;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (ldInfo.isLdCoreEnable) {
            vectorService.ProcessLD();
        }
    }
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
