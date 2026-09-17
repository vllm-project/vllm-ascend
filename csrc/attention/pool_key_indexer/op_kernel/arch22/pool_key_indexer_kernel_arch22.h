/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_KERNEL_ARCH22_H
#define POOL_KEY_INDEXER_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_service_vector_arch22.h"
#include "pool_key_indexer_service_cube_arch22.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace PkiServiceVec;
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
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;
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
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable, __gm__ uint8_t *qDescale,
                                __gm__ uint8_t *kDescale, __gm__ uint8_t *sparse_indices, __gm__ uint8_t *sparse_values,
                                __gm__ uint8_t *workspace, const PoolKeyIndexerTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    // 编译期条件选择模板第二个参数的类型，直接声明W_T
    // 第一个模板参数：固定为Q_T；第二个模板参数：编译期选float/void
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    using MM1_OUT_T = float;

    PoolKeyIndexerServiceCube<LIT> matmulService;
    PoolKeyIndexerServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;
    static constexpr uint32_t BLOCK_CUBE_SIZE = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    // values 输出行宽恒为 sparseCount, 与 indices(poolSize>1 时为 outputLen)
    // 不同, 必须单独维护基址偏移, 不可复用 indiceOutCoreOffset
    uint64_t valueOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<int64_t> actualSeqLengthsGmQ;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> poolTailKGm_;
    bool hasPoolTailK_ = false;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

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
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, PkiCommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, PkiCommon::RunInfo &runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int64_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
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
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.returnValue = tilingData->returnValue;
    constInfo.maskMode = tilingData->maskMode;
    constInfo.quantMode = tilingData->quantMode;
    constInfo.topk = tilingData->topk;
    constInfo.poolSize = tilingData->poolSize;
    constInfo.trunkLen = tilingData->trunkLen;
    constInfo.mBaseSizeMax = tilingData->mBaseSizeMax;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.wsOffScore = tilingData->wsOffScore;
    constInfo.wsOffLdScore = tilingData->wsOffLdScore;
    constInfo.wsOffLdIdx = tilingData->wsOffLdIdx;
    constInfo.qkScale = tilingData->qkScale;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    // arch22 切分常量以 kernel 侧硬编码为准, 覆盖 host tiling 字段
    // (host 下发字段供 arch35 消费); workspace 公式按 kernel 口径镜像推导
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;

    constInfo.s1BaseSize = constInfo.isSparseCountOver2K ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = PkiCommon::Align(constInfo.mBaseSize, BLOCK_CUBE_SIZE);
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
                                                                   __gm__ uint8_t *actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                      bool isAccumSeq,
                                                                      GlobalTensor<int64_t> &actualSeqLengthsGm,
                                                                      uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return static_cast<uint32_t>(actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    } else {
        return static_cast<uint32_t>(actualSeqLengthsGm.GetValue(bIdx));
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                      uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    uint32_t actS2Pool =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
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
            s2BaseNum =
                constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) :
                                                        GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline PoolKeyIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                            PkiCommon::SplitCoreInfo &info)
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
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    // 尾部残余 fill 的高水位终点跟踪(见函数末尾注释)
    uint32_t tailBN2End = 0;
    uint32_t tailGS1End = 0;
    uint32_t tailS2End = 0;
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
            s2Loop = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                // S2 跨核规避: 保证每个 (batch, gS1) 的 S2 块完整落在单核内
                // (LD 归并要求 S2 分段完整), 代价是多 batch 大 S2 场景核利用率下降。
                if (s2Idx == 0 && lastGS1RemainBlockCnt + s2RemainBaseNum > coreDealBlockCnt) {
                    coreDealBlockCnt = lastGS1RemainBlockCnt + s2RemainBaseNum;
                }
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isSparseCountOver2K ? s2BaseNum - 1 :
                                                                 s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

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
                    // 记录未完成 fill 的高水位终点(尾部残余 fill 用)
                    tailBN2End = bN2Idx;
                    tailGS1End = gS1Idx;
                    tailS2End = s2Loop - 1;
                    break;
                }
            }
        }
    }
    // 尾部残余 fill: 最后一段块数可能永远凑不满配额, 由当前核吃满剩余块, 防尾部块漏执行。
    if (!findLastCoreEnd && coreIdx == curCoreIdx) {
        info.bN2End = tailBN2End;
        info.gS1End = tailGS1End;
        info.s2End = tailS2End;
        if (info.bN2End != constInfo.batchSize - 1) {
            info.bN2End = constInfo.batchSize - 1;
            info.gS1End = 0;
            info.s2End = 0;
        }
        return;
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
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *qDescale, __gm__ uint8_t *kDescale, __gm__ uint8_t *sparse_indices, __gm__ uint8_t *sparse_values,
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
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);
    if (poolTailK != nullptr) {
        poolTailKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(poolTailK), constInfo.batchSize);
        hasPoolTailK_ = true;
    }

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 排布: |mm1ResGm(存S)|vec1ResGm(LD中间结果)|vec1ParamGm(LD参数)|
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程 ws: [aic, s1_cube, 头尾, idx/value, 2048] float
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    // ld参数 ws: [aic, s1_cube, 头尾, 16ele] int64
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparse_indices);
        valueOutGm.SetGlobalBuffer((__gm__ float *)sparse_values);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, indiceOutGm, valueOutGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)poolKey);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
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
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 当前 batch 的尾部 token 数(ExpandAndAppendIndices 尾块追加与 LD 参数均依赖)
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
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == PkiLayout::TND) {
            actualSeqQPrefixSum =
                (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1));
            actualSeqKPrefixSum =
                (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGm.GetValue(runInfo.bIdx - 1));
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
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset =
        keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
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
    ProcessDecode();
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

                uint32_t negInf = constInfo.INVALID_VAL;
                AscendC::InitGlobalMemory(valueOut, valueDealSize, negInf);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
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
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                                   PkiCommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessDecode()
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
} // namespace PkiKernel
#endif // POOL_KEY_INDEXER_KERNEL_ARCH22_H
