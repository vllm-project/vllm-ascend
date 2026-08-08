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
 * \file split_core_v2.cpp
 * \brief
*/

#include "split_core_v2.h"
#include <cstdio>
#include <math.h>
#include "log/log.h"

namespace split_core_v2 {

uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    if (baseInfo.actualSeqS1Size.empty()) {
        return baseInfo.s1Size;
    }

    if (baseInfo.actualLenQDims == 1U) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[0]);
    }

    if (!baseInfo.isAccumSeqS1) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]);
    }

    return (bIdx == 0U) ? static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]) :
                         static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1U]);
}

uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    uint32_t prefixSize = static_cast<uint32_t>(baseInfo.actualSeqPrefixSize);
    if (baseInfo.actualSeqS2Size.empty()) {
        return prefixSize + baseInfo.s2Size;
    }

    if (baseInfo.actualLenKvDims == 1U) {
        return prefixSize + static_cast<uint32_t>(baseInfo.actualSeqS2Size[0]);
    }

    if (!baseInfo.isAccumSeqS2) {
        return prefixSize + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]);
    }

    return (bIdx == 0U) ?
               prefixSize + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]) :
               prefixSize + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1U]);
}

int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        default:
            return baseInfo.nextToken;
    }
}

int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t mTailSize,
                                 uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (mTailSize == 0U) ? 0U : CalcCost(mTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (mTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(mTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> CalcS2Range(uint32_t mIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
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
        s2Start = 0U;
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of mFirstToken, mLastToken
    int64_t mFirstToken = static_cast<int64_t>(mIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t mLastToken = std::min(mFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
                                  static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) - 1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (baseInfo.isS1G) {
        s1FirstToken = mFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = mLastToken / static_cast<int64_t>(baseInfo.gSize);
    } else {
        if (mFirstToken / batchCache.s1Size == mLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = mFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = mLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2LastToken < 0 || s2FirstToken >= batchCache.s2Size || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize +
            1U; // end of block index, +1 for Right-open interval

    return std::make_pair(s2Start, s2End);
}

void CalcSplitInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;

    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);

        splitInfo.mBaseNum[bIdx] = (s1Size * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.mTailSize[bIdx] = (s1Size * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.mBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.mTailSize[bIdx],
                                        splitInfo.s2TailSize[bIdx]);
}

void CalcMCache(uint32_t mIdx, const SplitContext &splitContext, const BatchCache &batchCache, MCache &mCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    mCache.bIdx = batchCache.bIdx;
    mCache.mIdx = mIdx;

    auto s2Range = CalcS2Range(mIdx, baseInfo, splitParam, batchCache);
    mCache.s2Start = s2Range.first;
    mCache.s2End = s2Range.second;

    if (mCache.s2Start >= mCache.s2End) {
        mCache.mBlock = 0;
        mCache.mCost = 0;
        mCache.mLastBlockCost = 0;
        mCache.mNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    mCache.mBlock = mCache.s2End - mCache.s2Start;
    uint32_t curTailS2Num =
        (splitInfo.s2TailSize[batchCache.bIdx] != 0U && mCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U :
                                                                                                                  0U;
    uint32_t curNormalS2Num = mCache.mBlock - curTailS2Num;
    if (splitInfo.mBaseNum[batchCache.bIdx] == 0) {
        mCache.mCost = 0;
        mCache.mLastBlockCost = 0;
        mCache.mNormalBlockCost = 0;
    } else if (mIdx == (splitInfo.mBaseNum[batchCache.bIdx] - 1U) && splitInfo.mTailSize[batchCache.bIdx] != 0U) {
        mCache.mCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                            batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        mCache.mLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        mCache.mNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        mCache.mCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        mCache.mLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                                        batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        mCache.mNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.fdRes.fdNum = tmpRes.fdRes.fdNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.mEnd[i] = tmpRes.mEnd[i];
        splitRes.s2End[i] = tmpRes.s2End[i];
        splitRes.fdRes.fdBN2Idx[i] = tmpRes.fdRes.fdBN2Idx[i];
        splitRes.fdRes.fdMIdx[i] = tmpRes.fdRes.fdMIdx[i];
        splitRes.fdRes.fdS2SplitNum[i] = tmpRes.fdRes.fdS2SplitNum[i];
        splitRes.fdRes.fdWorkspaceIdx[i] = tmpRes.fdRes.fdWorkspaceIdx[i];
        splitRes.firstFdDataWorkspaceIdx[i] = tmpRes.firstFdDataWorkspaceIdx[i];
        splitRes.fdRes.mSize[i] = tmpRes.fdRes.mSize[i];
    }
}

void ClearTmpResult(FAMetaData &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.fdRes.fdNum = 0U;
    tmpResult.fdRes.fdUsedVecNum = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.mEnd[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.fdBN2Idx[i] = 0U;
        tmpResult.fdRes.fdMIdx[i] = 0U;
        tmpResult.fdRes.fdS2SplitNum[i] = 0U;
        tmpResult.fdRes.fdWorkspaceIdx[i] = 0U;
        tmpResult.firstFdDataWorkspaceIdx[i] = 0U;
        tmpResult.fdRes.mSize[i] = 0U;
    }
}

void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0LL;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0LL;

    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BatchCache bCache;
    MCache mCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t mIdx = 0; mIdx < splitInfo.mBaseNum[bIdx]; mIdx++) {
        CalcMCache(mIdx, splitContext, bCache, mCache);
        costInfo.bN2CostOfEachBatch[bIdx] += mCache.mCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += mCache.mBlock;
        if (mCache.mCost > costInfo.maxMCost) {
            costInfo.maxMCost = mCache.mCost;
        }
        if (mCache.mBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = mCache.mLastBlockCost;
        }
    }
}

void CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalBlockNum = 0U;
        costInfo.totalCost = 0;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
}

void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateM = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.mCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curMIdx++;
        UpdateM = true;
    }

    // Update M
    if (assignContext.curMIdx >= splitInfo.mBaseNum[assignContext.curBIdx]) {
        assignContext.curMIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx ==
        baseInfo.bSize * baseInfo.n2Size) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curMIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curMIdx = 0U;
        UpdateBatch = true;
        UpdateM = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateM) {
        CalcMCache(assignContext.curMIdx, splitContext, assignContext.batchCache, assignContext.mCache);
        assignContext.curS2Idx = assignContext.mCache.s2Start;
    }
}

void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const BaseInfo &baseInfo = splitContext.baseInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.curMIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curMIdx = 0U;
        CalcMCache(assignContext.curMIdx, splitContext, assignContext.batchCache, assignContext.mCache);
        assignContext.curS2Idx = assignContext.mCache.s2Start;
    }
}

void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.mCache.mLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.mCache.mCost)) {
        assignContext.coreCache.cost += assignContext.mCache.mCost;
        assignContext.coreCache.block += assignContext.mCache.mBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.mCache.mCost ?
                                    assignContext.bN2Cost - assignContext.mCache.mCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.mCache.mBlock ?
                                     assignContext.bN2Block - assignContext.mCache.mBlock :
                                     0U;
        // 计算新一行的信息
        do {
            assignContext.curMIdx++;
            CalcMCache(assignContext.curMIdx, splitContext, assignContext.batchCache, assignContext.mCache);
        } while (assignContext.mCache.mBlock == 0);
        assignContext.curS2Idx = assignContext.mCache.s2Start;
    }
}

void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !splitContext.splitParam.streamK) {
        return;
    }

    int64_t curCost = assignContext.mCache.mNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.mCache.s2End - 1U)) {
        curCost = assignContext.mCache.mLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost +
                             curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.mCache.mCost = assignContext.mCache.mCost - curCost;
        assignContext.bN2Block--;
        assignContext.mCache.mBlock--;
    }
}

void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.mCache.mNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.mCache.s2End - 1U)) {
        curCost = assignContext.mCache.mLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.mCache.mCost = assignContext.mCache.mCost - curCost;
    assignContext.mCache.mBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool IsNeedRecordFDInfo(const AssignContext &assignContext, const FAMetaData &splitRes)
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
        assignContext.curMIdx == splitRes.mEnd[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, FAMetaData &result)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitMIdx = result.mEnd[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitMIdx == splitInfo.mBaseNum[splitBIdx] - 1U) ?
                                (s1Size * baseInfo.gSize - splitMIdx * splitParam.mBaseSize) :
                                splitParam.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + splitParam.gS1BaseSizeOfFd - 1U) / splitParam.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (splitParam.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    // 记录
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.fdBN2Idx[result.fdRes.fdNum] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.fdRes.fdNum] = result.mEnd[assignContext.curCoreIdx - 1U];
    result.fdRes.fdS2SplitNum[result.fdRes.fdNum] = assignContext.curKvSplitPart;
    result.fdRes.fdWorkspaceIdx[result.fdRes.fdNum] = assignContext.preFdDataNUM;
    result.fdRes.mSize[result.fdRes.fdNum] = curFdS1gSize;
    result.fdRes.fdNum++;
}

void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curMIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcMCache(assignContext.curMIdx, splitContext, assignContext.batchCache, assignContext.mCache);
    assignContext.curS2Idx = assignContext.mCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
            assignContext.preFdDataNUM + assignContext.curKvSplitPart - 1U;

        int64_t avgCost = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);
        assignContext.coreCache = {};
        if (!splitContext.splitParam.streamK) {
            assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxMCost);
        } else {
            assignContext.coreCache.costLimit = avgCost;
        }

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        // 3、按块分配
        AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0 && splitContext.splitParam.streamK) {
            ForceAssign(splitContext, assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.mEnd[i] = assignContext.curMIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;

        // 对之前的归约信息进行记录并清理
        if (IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(splitContext, assignContext, result);
            assignContext.preFdDataNUM += assignContext.curKvSplitPart;
            assignContext.curKvSplitPart = 1U;
        }

        // 更新S2切分信息
        if (assignContext.curS2Idx > assignContext.mCache.s2Start &&
            assignContext.curS2Idx <= assignContext.mCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SplitFD(FAMetaData &result)
{
    FlashDecodeResult &fdRes = result.fdRes;
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < result.fdRes.fdNum; i++) {
        totalFDLoad += fdRes.fdS2SplitNum[i] * fdRes.mSize[i];
    }
    // 计算每个核处理的load
    uint32_t maxVectorNum = result.usedCoreNum * result.vecCubeRatio;
    uint32_t emptyVectorNum = maxVectorNum - result.fdRes.fdNum;
    uint64_t averageLoad = (totalFDLoad + maxVectorNum - 1U) / maxVectorNum; // 向上取整，避免核负载为0
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < result.fdRes.fdNum; i++) {
        if (emptyVectorNum == 0U) {
            fdRes.taskIdx[curCoreIndex] = i;
            fdRes.mStart[curCoreIndex] = 0U;
            fdRes.mLen[curCoreIndex] = fdRes.mSize[i];
            curCoreIndex++;
            continue;
        }

        uint32_t curFDVectorNum =
            std::max(fdRes.fdS2SplitNum[i] * fdRes.mSize[i] / averageLoad,
                     1UL); // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数且最少分一个核
        uint32_t curAvgMSize = (fdRes.mSize[i] + curFDVectorNum - 1U) /
                               curFDVectorNum; // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        curFDVectorNum =
            (fdRes.mSize[i] + curAvgMSize - 1U) / curAvgMSize; // 重新计算正确的核数，避免因为向上取整导致有核为空
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            fdRes.taskIdx[curCoreIndex] = i;
            fdRes.mStart[curCoreIndex] = vid * curAvgMSize;
            fdRes.mLen[curCoreIndex] = (vid < curFDVectorNum - 1) ? curAvgMSize : (fdRes.mSize[i] - vid * curAvgMSize);
            curCoreIndex++;
        }
        emptyVectorNum -= (curFDVectorNum - 1U);    // 1: 空余核不包含FD自身的核，要-1
    }
    fdRes.fdUsedVecNum = curCoreIndex;
}

bool CheckChooseWithFd(int64_t noFdCost, int64_t withFdCost, uint32_t actualMFdsize, const SplitParam &param)
{
    const int64_t fullBlockCost = CalcCost(actualMFdsize, param.s2BaseSize);
    if (noFdCost <= param.fdLeastBlock * fullBlockCost) {
        return false;
    }
    int64_t fdTolerance = param.fdTolerance * fullBlockCost;
    return noFdCost - fdTolerance > withFdCost;        // using minus in case overflow
}

void LogSplitCoreInput(const BaseInfo &baseInfo, const SplitParam &param)
{
    OP_LOGI("FusedInferAttentionScore", "========== BaseInfo ==========\n");
    OP_LOGI("FusedInferAttentionScore", "bSize: %u\n", baseInfo.bSize);
    OP_LOGI("FusedInferAttentionScore", "n2Size: %u\n", baseInfo.n2Size);
    OP_LOGI("FusedInferAttentionScore", "gSize: %u\n", baseInfo.gSize);
    OP_LOGI("FusedInferAttentionScore", "s1Size: %u\n", baseInfo.s1Size);
    OP_LOGI("FusedInferAttentionScore", "s2Size: %u\n", baseInfo.s2Size);
    OP_LOGI("FusedInferAttentionScore", "isS1G: %u\n", baseInfo.isS1G);
    OP_LOGI("FusedInferAttentionScore", "isAccumSeqS1: %u\n", baseInfo.isAccumSeqS1);
    OP_LOGI("FusedInferAttentionScore", "isAccumSeqS2: %u\n", baseInfo.isAccumSeqS2);
    OP_LOGI("FusedInferAttentionScore", "actualLenQDims: %u\n", baseInfo.actualLenQDims);
    OP_LOGI("FusedInferAttentionScore", "actualLenKvDims: %u\n", baseInfo.actualLenKvDims);
    OP_LOGI("FusedInferAttentionScore", "attenMaskFlag: %u\n", baseInfo.attenMaskFlag);
    OP_LOGI("FusedInferAttentionScore", "sparseMode: %u\n", baseInfo.sparseMode);
    OP_LOGI("FusedInferAttentionScore", "preToken: %lld\n", baseInfo.preToken);
    OP_LOGI("FusedInferAttentionScore", "nextToken: %lld\n", baseInfo.nextToken);
    OP_LOGI("FusedInferAttentionScore", "actualSeqPrefixSize: %lld\n", baseInfo.actualSeqPrefixSize);
    if (baseInfo.actualSeqS1Size.empty()) {
        OP_LOGI("FusedInferAttentionScore", "actualSeqS1Size: nullptr\n");
    } else {
        OP_LOGI("FusedInferAttentionScore", "actualSeqS1Size: [");
        for (size_t i = 0; i < baseInfo.actualSeqS1Size.size(); ++i) {
            OP_LOGI("FusedInferAttentionScore", "%lld", baseInfo.actualSeqS1Size[i]);
            if (i < baseInfo.actualSeqS1Size.size() - 1) {
                OP_LOGI("FusedInferAttentionScore", ", ");
            }
        }
        OP_LOGI("FusedInferAttentionScore", "]\n");
    }
    if (baseInfo.actualSeqS2Size.empty()) {
        OP_LOGI("FusedInferAttentionScore", "actualSeqS2Size: nullptr\n");
    } else {
        OP_LOGI("FusedInferAttentionScore", "actualSeqS2Size: [");
        for (size_t i = 0; i < baseInfo.actualSeqS2Size.size(); ++i) {
            OP_LOGI("FusedInferAttentionScore", "%lld", baseInfo.actualSeqS2Size[i]);
            if (i < baseInfo.actualSeqS2Size.size() - 1) {
                OP_LOGI("FusedInferAttentionScore", ", ");
            }
        }
        OP_LOGI("FusedInferAttentionScore", "]\n");
    }
    OP_LOGI("FusedInferAttentionScore", "mBaseSize: %u\n", param.mBaseSize);
    OP_LOGI("FusedInferAttentionScore", "s2BaseSize: %u\n", param.s2BaseSize);
    OP_LOGI("FusedInferAttentionScore", "gS1BaseSizeOfFd: %u\n", param.gS1BaseSizeOfFd);
}

void LogSplitCoreResult(const FAMetaData &result)
{
    OP_LOGI("FusedInferAttentionScore", "========== start print split core res ==========\n");
    OP_LOGI("FusedInferAttentionScore", "result.usedCoreNum: %d \n", result.usedCoreNum);
    OP_LOGI("FusedInferAttentionScore", "result.maxCost: %lld \n", result.maxCost);
    OP_LOGI("FusedInferAttentionScore", "result.fdRes.fdNum: %d \n", result.fdRes.fdNum);
    OP_LOGI("FusedInferAttentionScore", "result.fdRes.fdUsedVecNum: %d \n", result.fdRes.fdUsedVecNum);
    for (uint32_t i = 0; i < result.usedCoreNum; i++) {
        OP_LOGI("FusedInferAttentionScore", "outerSplitParams.bN2End[%d]: %d \n", i, result.bN2End[i]);
        OP_LOGI("FusedInferAttentionScore", "outerSplitParams.mEnd[%d]: %d \n", i, result.mEnd[i]);
        OP_LOGI("FusedInferAttentionScore", "outerSplitParams.s2End[%d]: %d \n", i, result.s2End[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.firstFdDataWorkspaceIdx[%d]: %d \n",
                i, result.firstFdDataWorkspaceIdx[i]);
    }
    for (uint32_t i = 0; i < result.fdRes.fdNum; i++) {
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.fdBN2Idx[%d]: %d \n", i, result.fdRes.fdBN2Idx[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.fdMIdx[%d]: %d \n", i, result.fdRes.fdMIdx[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.fdS2SplitNum[%d]: %d \n", i, result.fdRes.fdS2SplitNum[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.fdWorkspaceIdx[%d]: %d \n",
                i, result.fdRes.fdWorkspaceIdx[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.mSize[%d]: %d \n", i, result.fdRes.mSize[i]);
    }
    for (uint32_t i = 0; i < result.fdRes.fdUsedVecNum; i++) {
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.taskIdx[%d]: %d \n", i, result.fdRes.taskIdx[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.mStart[%d]: %d \n", i, result.fdRes.mStart[i]);
        OP_LOGI("FusedInferAttentionScore", "fDParams.fdRes.mLen[%d]: %d \n", i, result.fdRes.mLen[i]);
    }
}

void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, FAMetaData &result)
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
        result.mEnd[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;

    FAMetaData tmpResult{coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    // 根据shape判断是否要开FD
    if (param.streamK && result.fdRes.fdNum > 0) {
        splitContext.splitParam.streamK = false;
        CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
        uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
        if (!CheckChooseWithFd(tmpResult.maxCost, result.maxCost, std::min(maxFdSize, param.mBaseSize), param)) {
            CopyTmpResult(tmpResult, result);
        }
    }
    // 3、存在FD任务，对FD进行负载均衡分配
    if (result.fdRes.fdNum > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U); // 至少使用1个core

    // 打印分核结果信息
    LogSplitCoreResult(result);
}
} // namespace split_core_v2
