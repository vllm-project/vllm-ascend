/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_KERNEL_ARCH35_H
#define POOL_KEY_INDEXER_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_service_vector_arch35.h"
#include "pool_key_indexer_service_cube_arch35.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bIdx = 0U;
    uint32_t bN2Idx = 0;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U; // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;  // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
};

template <typename LIT>
class PoolKeyIndexerKernel {
public:
    __aicore__ inline PoolKeyIndexerKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *poolKey, __gm__ uint8_t *weights,
                                __gm__ uint8_t *poolTailK, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable, __gm__ uint8_t *qDescale,
                                __gm__ uint8_t *kDescale, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                __gm__ uint8_t *workspace, const PoolKeyIndexerTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using SCORE_T = uint32_t;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    using W_T = typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t, entry 按
    // ORIG_DTYPE_WEIGHTS 分发), 优先于 PkiTypeTraits 推导
    using W_QUANT_T = typename LIT::weightsType;
    using W_ACT_T = typename std::conditional<std::is_same<W_QUANT_T, void>::value, W_T, W_QUANT_T>::type;

    PoolKeyIndexerServiceCube<LIT> matmulService;
    PoolKeyIndexerServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t qScaleCoreOffset = 0ULL; // 量化: q_descale GM 偏移(mode=0 vector 融合)
    uint64_t kScaleCoreOffset = 0ULL; // 量化: k_descale GM 偏移(mode=0 vector 融合)
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t valueOutCoreOffset = 0ULL;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_ACT_T> weightsGm;
    // 量化 descale: mode=0(float) 走 vector 融合; mode=1(E8M0 按 2 打包成
    // bf16 寻址) 走 Cube LoadData-Mx。两个模式均以 bf16 视角 SetGlobalBuffer。
    GlobalTensor<bfloat16_t> qScaleGmBf16;
    GlobalTensor<bfloat16_t> kScaleGmBf16;
    GlobalTensor<float> qScaleGmFp32;
    GlobalTensor<float> kScaleGmFp32;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<int64_t> actualSeqLengthsGmQ;
    GlobalTensor<int64_t> actualSeqLengthsGmKv;
    GlobalTensor<int64_t> poolTailKGm_;
    bool hasPoolTailK_ = false;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    PkiCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    PkiCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const PoolKeyIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, PkiCommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, PkiCommon::RunInfo runInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int64_t> &actualSeqLengthsGmKv, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, PkiCommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitTilingData(const PoolKeyIndexerTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.outputLayout = LAYOUT_T;  // 输出和输入形状一致
    constInfo.INVALID_VAL = 0xFF800000; // float -inf
    if constexpr (LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    if (constInfo.sparseCount > 2048) {
        constInfo.mBaseSize = S1_BASE_SIZE_SMALL * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
    } else {
        constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE;
    }
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.returnValue = tilingData->returnValue;
    constInfo.splitMFlag = (constInfo.gSize == 64 && constInfo.sparseCount <= 2048);
    if constexpr (LIT::isQuant) {
        // 量化路径禁用 splitM: splitM 的 LoadData2D dst 布局按 fp16 M 切分推导,
        // fp8 的 L1 NZ 行深变化后布局错位, 量化场景走常规单块路径
        constInfo.splitMFlag = false;
        // 禁用后 mBaseSize 必须钳到 L0 容量(128 行), 且能被 gSize 整除,
        // 否则 L0 越界或 Fixpipe 半区行与 Vec 读取行距错位
        if (constInfo.mBaseSize > 128) {
            uint32_t s1PerL0 = 128 / constInfo.gSize; // L0 128 行可容纳的 s1 行数(整除截断)
            if (s1PerL0 == 0) {
                s1PerL0 = 1; // gSize>128 时兜底(N1<=64, 正常不触达)
            }
            // 半区(mBaseSize/2)必须含整数个 s1 行(dualDstCtl 按半区拆分给
            // 两个 AIV, Vec 按 gSize 行距读取), s1PerL0 取偶
            if (s1PerL0 % 2 == 1) {
                s1PerL0 -= 1;
                if (s1PerL0 == 0) {
                    s1PerL0 = 1; // gSize>64 时只能 1 行(半区整行归属, mBaseSize=gSize<=64 无此分支)
                }
            }
            constInfo.mBaseSize = s1PerL0 * constInfo.gSize;
            constInfo.s1BaseSize = s1PerL0;
        }
    }
    constInfo.maskMode = tilingData->maskMode;
    constInfo.quantMode = tilingData->quantMode;
    constInfo.topk = tilingData->topk;
    constInfo.poolSize = tilingData->poolSize;
    constInfo.trunkLen = tilingData->trunkLen;
    constInfo.mBaseSizeMax = tilingData->mBaseSizeMax;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.wsOffScore = tilingData->wsOffScore;
    constInfo.wsOffLdScore = tilingData->wsOffLdScore;
    constInfo.wsOffLdIdx = tilingData->wsOffLdIdx;
    constInfo.qkScale = tilingData->qkScale;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount > 2048);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                   __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                      bool isAccumSeq,
                                                                      GlobalTensor<int64_t> &actualSeqLengthsGmKv,
                                                                      uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return static_cast<uint32_t>(actualSeqLengthsGmKv.GetValue(bIdx) - actualSeqLengthsGmKv.GetValue(bIdx - 1));
    } else {
        return static_cast<uint32_t>(actualSeqLengthsGmKv.GetValue(bIdx));
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                      uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    uint32_t actS2Pool = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGmKv,
                                         constInfo.kSeqSize);
    uint32_t tailK = hasPoolTailK_ ? static_cast<uint32_t>(poolTailKGm_.GetValue(bIdx)) : 0;
    actS2SizeOrig = actS2Pool * constInfo.poolSize + tailK;
    actS2Size = actS2SizeOrig / constInfo.poolSize;
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                              uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.poolSize == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len =
        (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
        static_cast<int32_t>(constInfo.poolSize);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / static_cast<int32_t>(constInfo.poolSize));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isLDOpen ? CeilDiv(actS2Size, constInfo.s2BaseSize) : (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isLDOpen ? GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) :
                                             (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIT>
__aicore__ void inline PoolKeyIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                            PkiCommon::SplitCoreInfo &info)
{
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        splitCoreInfo.isCoreEnable = true;
    } else {
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == PkiLayout::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2SizeOrig);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isLDOpen ? s2BaseNum : (actS2Size > 0 ? 1 : 0);
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End =
                        constInfo.isLDOpen ? s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1 : s2BaseNum - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.s2End = 0;
                            info.gS1End = 0;
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
__aicore__ inline void PoolKeyIndexerKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        // poolSize>1 时 indices 输出行宽为 outputLen(见 CalcRunInfo 中 idxOutStride 说明)
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        if (constInfo.outputLayout == PkiLayout::TND) {
            uint32_t tSize = static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1));
            uint32_t tBase = bIdx == 0 ? 0 : static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(bIdx - 1));
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset = (tBase + s1Idx) * constInfo.kHeadNum * idxOutStride + // T轴、s1轴偏移
                                           n2Idx * idxOutStride;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == PkiLayout::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * idxOutStride +
                                           s1Idx * constInfo.kHeadNum * idxOutStride + // B轴、S1轴偏移
                                           n2Idx * idxOutStride;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *poolKey, __gm__ uint8_t *weights, __gm__ uint8_t *poolTailK,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *qDescale, __gm__ uint8_t *kDescale, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
    __gm__ uint8_t *workspace, const PoolKeyIndexerTilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);
    if (poolTailK != nullptr) {
        poolTailKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(poolTailK), constInfo.batchSize);
        hasPoolTailK_ = true;
    }

    // 获取分核信息
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;

    uint64_t offset = 0;
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize = constInfo.s1BaseSize *
                                   PkiCommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize) *
                                   sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + aiCoreIdx * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ float *)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_ACT_T *)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        if constexpr (LIT::isFp8PerToken) {
            // mode=0: q/k_descale 为 float, vector 侧融合
            qScaleGmFp32.SetGlobalBuffer((__gm__ float *)qDescale);
            kScaleGmFp32.SetGlobalBuffer((__gm__ float *)kDescale);
            vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm, qScaleGmFp32,
                                             kScaleGmFp32);
        } else {
            vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm);
        }
        vectorService.InitVecWorkspaceTensor(scoreGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)poolKey);
        if constexpr (LIT::isMxFp8) {
            // mode=1: E8M0 scale 按 2 个打包成 1 个 bf16 寻址, 块数 = e8m0 元素数 / 2
            qScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)qDescale);
            kScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)kDescale);
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, kScaleGmBf16, qScaleGmBf16);
        } else {
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
        }
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
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
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
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
    if constexpr (LAYOUT_T == PkiLayout::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                              PkiCommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;

    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    runInfo.poolTailK = hasPoolTailK_ ? static_cast<int32_t>(poolTailKGm_.GetValue(tempLoopInfo.bIdx)) : 0;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        PkiCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, PkiCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (LAYOUT_T == PkiLayout::TND) {
            actualSeqQPrefixSum =
                (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1));
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // 量化 q_descale 偏移: mode=0 与 weights 同构直接复用其公式;
        // mode=1 为 (B,S1,N1,D/64,2), 每 s1 行 D/32 个 e8m0
        if constexpr (LIT::isFp8PerToken) {
            qScaleCoreOffset = weightsCoreOffset;
        } else if constexpr (LIT::isMxFp8) {
            qScaleCoreOffset =
                actualSeqQPrefixSum * constInfo.qHeadNum * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE) +
                runInfo.gS1Idx * constInfo.mBaseSize * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE) +
                runInfo.n2Idx * constInfo.gSize * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE);
        }
        // B,S1,N2,k/T,N2,k; poolSize>1 时 indices 行宽为 outputLen(≠sparseCount),
        // 否则 batch>0 行基址错位, 写入覆盖相邻行
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * idxOutStride + runInfo.n2Idx * idxOutStride;
        // values 输出行宽恒为 sparseCount(与 indices 的 outputLen 不同),
        // 不可复用 indiceOutCoreOffset, 否则错位写到相邻 batch
        valueOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == PkiLayout::TND) { // T N2 D
        actualSeqKPrefixSum =
            (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGmKv.GetValue(runInfo.bIdx - 1));
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    // 量化 k_descale 偏移: mode=0 每池 1 个 float(偏移=池前缀+s2Idx*s2BaseSize);
    // mode=1 每池 D/32 个 e8m0(偏移=(池前缀+s2Idx*s2BaseSize)*N2*(D/32))
    if constexpr (LIT::isFp8PerToken) {
        kScaleCoreOffset =
            actualSeqKPrefixSum * constInfo.kHeadNum + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum;
    } else if constexpr (LIT::isMxFp8) {
        uint64_t kScaleS2Offset = actualSeqKPrefixSum + static_cast<uint64_t>(runInfo.s2Idx) * constInfo.s2BaseSize;
        kScaleCoreOffset = kScaleS2Offset * constInfo.kHeadNum * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE);
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.tensorQScaleOffset = qScaleCoreOffset;
    runInfo.tensorKeyScaleOffset = kScaleCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.valueOutOffset = valueOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        // poolSize>1 时 indices 输出行宽为 outputLen(见 CalcRunInfo 中 idxOutStride 说明)
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        // TND qSeqSize already contains the total token count across batches.
        uint64_t queryTokenCount = constInfo.qSeqSize;
        if constexpr (LAYOUT_T == PkiLayout::BSND) {
            queryTokenCount *= constInfo.batchSize;
        }
        uint64_t totalOutputSize = queryTokenCount * constInfo.kHeadNum * idxOutStride;
        uint64_t singleCoreSize =
            PkiCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
        if (constInfo.returnValue) {
            // values 总大小与 indices 不同(行宽 sparseCount vs outputLen),
            // 需按自身总大小独立切分清理
            uint64_t totalValueSize =
                queryTokenCount * constInfo.kHeadNum * constInfo.sparseCount;
            uint64_t singleCoreValueSize =
                PkiCommon::Align((totalValueSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(uint32_t));
            uint64_t valueBase = tmpBlockIdx * singleCoreValueSize;
            if (valueBase < totalValueSize) {
                uint64_t valueDealSize = (valueBase + singleCoreValueSize <= totalValueSize) ?
                                             singleCoreValueSize :
                                             totalValueSize - valueBase;
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint32_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint32_t> valueOut = valueOutGmTmp[valueBase];
                AscendC::InitGlobalMemory(valueOut, valueDealSize, constInfo.INVALID_VAL);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    PkiCommon::RunInfo runInfo;
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
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_FIX>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_FIX>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                                   PkiCommon::RunInfo runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec1(runInfo);
        if (runInfo.isLastS2InnerLoop) { // 本核s2last
            vectorService.ProcessTopK(runInfo);
        }
    }
}

} // namespace PkiKernel
#endif // POOL_KEY_INDEXER_KERNEL_ARCH35_H
