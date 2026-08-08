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
 * \file split_core.cpp
 * \brief
 */

#include "split_core.h"
#include <cstdio>
#include <math.h>
#include "log/log.h"


namespace optiling {

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    const auto &actualSeqS1 = baseInfo.actualSeqS1Size;

    if (actualSeqS1.empty()) {
        return baseInfo.s1Size;
    }

    if (baseInfo.actualLenQDims == 1U) {
        return static_cast<uint32_t>(actualSeqS1[0]);
    }

    if (!baseInfo.isAccumSeqS1) {
        return static_cast<uint32_t>(actualSeqS1[bIdx]);
    }

    return (bIdx == 0U) ? static_cast<uint32_t>(actualSeqS1[bIdx]) :
           static_cast<uint32_t>(actualSeqS1[bIdx] - actualSeqS1[bIdx - 1U]);
}

uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    const auto &actualSeqS2 = baseInfo.actualSeqS2Size;
    uint32_t prefix = static_cast<uint32_t>(baseInfo.actualSeqPrefixSize);
    if (actualSeqS2.empty()) {
        return prefix + baseInfo.s2Size;
    }

    if (baseInfo.actualLenKvDims == 1U) {
        return prefix + static_cast<uint32_t>(actualSeqS2[0]);
    }

    if (!baseInfo.isAccumSeqS2) {
        return prefix + static_cast<uint32_t>(actualSeqS2[bIdx]);
    }
    
    return (bIdx == 0U) ? prefix + static_cast<uint32_t>(actualSeqS2[bIdx]) :
           prefix + static_cast<uint32_t>(actualSeqS2[bIdx] - actualSeqS2[bIdx - 1U]);
}

int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case SparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
        case SparseMode::TREE:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        default:
            return baseInfo.nextToken;
    }
}

int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
    uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
    const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // no mask
    if (!baseInfo.attenMaskFlag) {
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) - 1;
    
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (!baseInfo.isS1G) {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    } else {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.gSize);
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size -1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size -1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize + 1U; // end of block index, +1 for Right-open interval

    return std::make_pair(s2Start, s2End);
}

void CalcSplitInfo(SplitContext &splitContext)
{
    const SplitParam &splitParam = splitContext.splitParam;
    const BaseInfo &baseInfo = splitContext.baseInfo;
    
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);

        splitInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.s1GTailSize[bIdx] = (s1Size * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    OP_LOGI ("SplitInfo", "========== SplitInfo ==========");
    OP_LOGI("SplitInfo", "splitInfo.isKvSeqAllZero: %u", splitInfo.isKvSeqAllZero);
    for (uint32_t i = 0; i < baseInfo.bSize; i++) {
        OP_LOGI("SplitInfo", "splitInfo.s1GBaseNum[%u]: %d", i, splitInfo.s1GBaseNum[i]);
        OP_LOGI("SplitInfo", "splitInfo.s1GTailSize[%u]: %d", i, splitInfo.s1GTailSize[i]);
        OP_LOGI("SplitInfo", "splitInfo.s2BaseNum[%u]: %d", i, splitInfo.s2BaseNum[i]);
        OP_LOGI("SplitInfo", "splitInfo.s2TailSize[%u]: %d", i, splitInfo.s2TailSize[i]);
    }
}

void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const SplitParam &splitParam = splitContext.splitParam;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, baseInfo, splitParam, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = (splitInfo.s2TailSize[batchCache.bIdx] != 0U &&
        s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];

        splitRes.fdRes.bN2IdxOfFdHead[i] = tmpRes.fdRes.bN2IdxOfFdHead[i];
        splitRes.fdRes.gS1IdxOfFdHead[i] = tmpRes.fdRes.gS1IdxOfFdHead[i];
        splitRes.fdRes.s2SplitNumOfFdHead[i] = tmpRes.fdRes.s2SplitNumOfFdHead[i];
        splitRes.fdRes.s2SplitStartIdxOfCore[i] = tmpRes.fdRes.s2SplitStartIdxOfCore[i];
        splitRes.fdRes.gS1SplitNumOfFdHead[i] = tmpRes.fdRes.gS1SplitNumOfFdHead[i];
        splitRes.fdRes.gS1LastPartSizeOfFdHead[i] = tmpRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }
}

void ClearTmpResult(SplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.usedVecNumOfFd = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.bN2IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitStartIdxOfCore[i] = 0U;
        tmpResult.fdRes.gS1SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1LastPartSizeOfFdHead[i] = 0U;
    }
}

void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0;

    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
    OP_LOGI("CostInfo", "========== CostInfo ==========");
    OP_LOGI("CostInfo", "costInfo.totalCost: %ld", costInfo.totalCost);
    OP_LOGI("CostInfo", "costInfo.totalBlockNum: %u", costInfo.totalBlockNum);
    for (uint32_t i = 0; i < baseInfo.bSize; i++) {
        OP_LOGI("CostInfo", "costInfo.bN2CostOfEachBatch[%u]: %d", i, costInfo.bN2CostOfEachBatch[i]);
        OP_LOGI("CostInfo", "costInfo.bN2LastBlockCostOfEachBatch[%u]: %d", i, costInfo.bN2LastBlockCostOfEachBatch[i]);
        OP_LOGI("CostInfo", "costInfo.bN2BlockOfEachBatch[%u]: %d", i, costInfo.bN2BlockOfEachBatch[i]);
    }
}

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {    // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {  // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const BaseInfo &baseInfo = splitContext.baseInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.isFinished = true;
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void AssignByBlock(AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO, 
            assignContext.coreCache.cost + curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext); 
}

bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * baseInfo.gSize - splitS1GIdx * splitParam.mBaseSize) : splitParam.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + splitParam.gS1BaseSizeOfFd - 1U) / splitParam.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (splitParam.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}

void LogAssignContext(const char* phase, const AssignContext &assignContext) {
    OP_LOGD("assignContext", "[%s] curCoreIdx: %u, unassignedCost: %ld, "
    "bIdx: %u, bN2Idx: %u, s1GIdx: %u, s2Idx: %u, "
    "bN2Cost: %ld, bN2Block: %u, "
    "s1GBlock: %u, s2Start: %u, s2End: %u",
    phase, assignContext.curCoreIdx, assignContext.unassignedCost,
    assignContext.curBIdx, assignContext.curBN2Idx, assignContext.curS1GIdx, assignContext.curS2Idx,
    assignContext.bN2Cost, assignContext.bN2Block,
    assignContext.s1GCache.s1GBlock, assignContext.s1GCache.s2Start, assignContext.s1GCache.s2End);
}

void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        LogAssignContext("START", assignContext);
        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        LogAssignContext("BATCH", assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        LogAssignContext("ROW", assignContext);
        // 3、按块分配
        AssignByBlock(assignContext);
        LogAssignContext("BLOCK", assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
            LogAssignContext("FORCE", assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;

        // 对之前的归约信息进行记录并清理
        if (IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(splitContext, assignContext, result);
            assignContext.curKvSplitPart = 1U;
        }

        // 更新S2切分信息
        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector);  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);  // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

void LogSplitCoreInput(const BaseInfo &baseInfo, const SplitParam &param)
{
    OP_LOGI("BaseInfo", "========== BaseInfo ==========");
    OP_LOGI("BaseInfo", "bSize: %u", baseInfo.bSize);
    OP_LOGI("BaseInfo", "n2Size: %u", baseInfo.n2Size);
    OP_LOGI("BaseInfo", "gSize: %u", baseInfo.gSize);
    OP_LOGI("BaseInfo", "s1Size: %u", baseInfo.s1Size);
    OP_LOGI("BaseInfo", "s2Size: %u", baseInfo.s2Size);
    OP_LOGI("BaseInfo", "isS1G: %u", baseInfo.isS1G);
    OP_LOGI("BaseInfo", "isAccumSeqS1: %u", baseInfo.isAccumSeqS1);
    OP_LOGI("BaseInfo", "isAccumSeqS2: %u", baseInfo.isAccumSeqS2);
    OP_LOGI("BaseInfo", "actualLenQDims: %u", baseInfo.actualLenQDims);
    OP_LOGI("BaseInfo", "actualLenKvDims: %u", baseInfo.actualLenKvDims);
    OP_LOGI("BaseInfo", "attenMaskFlag: %u", baseInfo.attenMaskFlag);
    OP_LOGI("BaseInfo", "sparseMode: %u", baseInfo.sparseMode);
    OP_LOGI("BaseInfo", "preToken: %ld", baseInfo.preToken);
    OP_LOGI("BaseInfo", "nextToken: %ld", baseInfo.nextToken);
    OP_LOGI("BaseInfo", "actualSeqPrefixSize: %ld", baseInfo.actualSeqPrefixSize);
    OP_LOGI("SplitParam", "mBaseSize: %u", param.mBaseSize);
    OP_LOGI("SplitParam", "s2BaseSize: %u", param.s2BaseSize);
}
void LogSplitCoreResult(const SplitResult &result)
{
    OP_LOGI("SplitRes", "splitRes.usedCoreNum: %d", result.usedCoreNum);
    OP_LOGI("SplitRes", "splitRes.maxCost: %d", result.maxCost);
    OP_LOGI("SplitRes", "splitRes.numOfFdHead: %d", result.numOfFdHead);
    OP_LOGI("SplitRes", "splitRes.maxS2SplitNum: %d", result.maxS2SplitNum);
    OP_LOGI("SplitRes", "splitRes.usedVecNumOfFd: %d", result.usedVecNumOfFd);
    for (uint32_t i = 0; i < result.usedCoreNum; i++) {
        OP_LOGI("SplitRes", "outerSplitParams.bN2End[%u]: %d", i, result.bN2End[i]);
        OP_LOGI("SplitRes", "outerSplitParams.gS1End[%u]: %d", i, result.gS1End[i]);
        OP_LOGI("SplitRes", "outerSplitParams.s2End[%u]: %d", i, result.s2End[i]);
        OP_LOGI("fdRes", "fDParams.s2SplitStartIdxOfCore[%u]: %d", i, result.fdRes.s2SplitStartIdxOfCore[i]);
    }
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        OP_LOGI("fdRes", "fDParams.bN2IdxOfFdHead[%u]: %d", i, result.fdRes.bN2IdxOfFdHead[i]);
        OP_LOGI("fdRes", "fDParams.gS1IdxOfFdHead[%u]: %d", i, result.fdRes.gS1IdxOfFdHead[i]);
        OP_LOGI("fdRes", "fDParams.s2SplitNumOfFdHead[%u]: %d", i, result.fdRes.s2SplitNumOfFdHead[i]);
        OP_LOGI("fdRes", "fDParams.gS1SplitNumOfFdHead[%u]: %d", i, result.fdRes.gS1SplitNumOfFdHead[i]);
        OP_LOGI("fdRes", "fDParams.gS1LastPartSizeOfFdHead[%u]: %d", i, result.fdRes.gS1LastPartSizeOfFdHead[i]);
    }
    for (uint32_t i = 0; i < result.usedVecNumOfFd; i++) {
        OP_LOGI("fdRes", "fDParams.gS1IdxEndOfFdHead[%u]: %d", i, result.fdRes.gS1IdxEndOfFdHead[i]);
        OP_LOGI("fdRes", "fDParams.gS1IdxEndOfFdHeadSplit[%u]: %d", i, result.fdRes.gS1IdxEndOfFdHeadSplit[i]);
    }
}

void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, SplitResult &result)
{
    // 打印baseInfo,param信息
    LogSplitCoreInput(baseInfo, param);
    SplitContext splitContext(baseInfo, param);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.gS1End[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }

    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(
        std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;

    SplitResult tmpResult {coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        OP_LOGD("SplitRes", "CoreNum: %u", i);
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    // 3、存在FD任务，对FD进行负载均衡分配
    if (result.numOfFdHead > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);  // 至少使用1个core

    // 打印分核结果信息
    LogSplitCoreResult(result);
}

}
