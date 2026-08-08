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
 * \file split_core_v2.h
 * \brief
*/

#ifndef SPLIT_CORE_V2_H
#define SPLIT_CORE_V2_H

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace split_core_v2 {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
T Clip(T value, T min, T max)
{
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

template <typename T>
inline bool IsWithinTolerance(T limit, T toler, T value)
{
    return limit + toler >= value;
}

// 分核功能模块输入：输入case的基本信息
struct BaseInfo {
    uint32_t bSize{0U};
    uint32_t n2Size{0U};
    uint32_t gSize{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    bool isS1G{true};
    bool isAccumSeqS1{false};
    bool isAccumSeqS2{false};
    std::vector<int64_t> actualSeqS1Size{};
    std::vector<int64_t> actualSeqS2Size{};
    uint32_t actualLenQDims{0U};
    uint32_t actualLenKvDims{0U};
    bool attenMaskFlag{false};
    int32_t sparseMode{0U};
    int64_t preToken{0};
    int64_t nextToken{0};
    int64_t actualSeqPrefixSize{0};
};

// 分核功能模块输入：切分属性，预留接口，可作为切分方案的参数入口
struct SplitParam {
    uint32_t mBaseSize{ 1U };
    uint32_t s2BaseSize{ 1U };
    uint32_t gS1BaseSizeOfFd{ 8U }; // FD阶段分核，m轴切分基本块大小
    bool streamK = true;
    int64_t fdTolerance { 0U };             // if fd - full_block_cost * fdTolerance <= fd, then choose no fd,
                                            // full_block_cost is costFunc(mBaseSize, s2BaseSize)
    int64_t fdLeastBlock { 0U };            // if noFd.maxCost <= fdLeastBlock * full_block_cost, then choose no fd
};


// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    // 1、归约任务的索引信息
    uint32_t fdNum{0U};               // 归约任务数量
    std::vector<uint32_t> fdBN2Idx{}; // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> fdMIdx{};       // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> fdS2SplitNum{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    std::vector<uint32_t> fdWorkspaceIdx{}; // 每个归约任务在workspace中的存放位置
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    uint32_t fdUsedVecNum{0U};     // 归约过程使用的vector数量
    std::vector<uint32_t> mSize{}; // 每个归约任务的M轴大小，脚标为归约任务的序号
    // 2、FD kernel阶段，归约任务的分核信息
    std::vector<uint32_t> taskIdx{}; // 每个vector处理的归约任务对应ID
    std::vector<uint32_t> mStart{};  // 每个vector处理的归约任务的M轴起点
    std::vector<uint32_t> mLen{};    // 每个vector处理的归约任务的M轴行数
    // 3、每个core处理的第1个归约任务的数据应存放的workspace位置
    FlashDecodeResult(uint32_t coreNum, uint32_t vecCubeRatio)
        : fdBN2Idx(coreNum), fdMIdx(coreNum), fdS2SplitNum(coreNum), fdWorkspaceIdx(coreNum), mSize(coreNum),
          taskIdx(coreNum * vecCubeRatio), mStart(coreNum * vecCubeRatio), mLen(coreNum * vecCubeRatio)
    {
    }
};

// 分核功能模块输出：FA阶段的核间分核信息
struct FAMetaData {
    uint32_t usedCoreNum{0U};       // 使用的核数量
    uint32_t vecCubeRatio{0U};      // vec 与 cube 核数比例
    std::vector<uint32_t> bN2End{}; // 每个核处理数据的BN2结束点
    std::vector<uint32_t> mEnd{};   // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End{};  // 每个核处理数据的S2结束点
    int64_t maxCost{0};             // 慢核开销
    std::vector<uint32_t> firstFdDataWorkspaceIdx{}; // 每个core处理的第1个归约任务的数据应存放的workspace位置,绝对偏移
    FlashDecodeResult fdRes{0U, 0U}; // FD信息
    FAMetaData(uint32_t coreNum, uint32_t ratio)
        : vecCubeRatio(ratio), bN2End(coreNum), mEnd(coreNum), s2End(coreNum), firstFdDataWorkspaceIdx(coreNum),
          fdRes(coreNum, ratio){};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> mBaseNum{};  // M方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum{};   // S2方向，切了多少个基本块
    std::vector<uint32_t> mTailSize{}; // M方向，尾块size
    std::vector<uint32_t> s2TailSize{};  // S2方向，尾块size
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : mBaseNum(batchSize), s2BaseNum(batchSize), mTailSize(batchSize), s2TailSize(batchSize)
    {
    }
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};          // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch{};        // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{}; // batch最后一块的开销
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxMCost{0};

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize), bN2BlockOfEachBatch(batchSize), bN2LastBlockCostOfEachBatch(batchSize)
    {
    }
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    const BaseInfo &baseInfo{0U};
    SplitParam splitParam{};
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(const BaseInfo &info, SplitParam param)
        : baseInfo(info), splitParam(param), splitInfo(info.bSize), costInfo(info.bSize)
    {
    }
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    int64_t preTokenLeftUp{0};
    int64_t nextTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

// 分核功能模块内部使用：记录当前行（M）的临时信息
struct MCache {
    uint32_t bIdx{0U};
    uint32_t mIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    int64_t mCost{0};
    int64_t mLastBlockCost{0};
    uint32_t mBlock{0U};
    int64_t mNormalBlockCost{0};
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit{0}; // 负载上限
    int64_t cost{0};      // 已分配负载
    uint32_t block{0U};   // 已分配块数
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curMIdx{0U};
    uint32_t curS2Idx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t usedCoreNum{0U};
    uint32_t curKvSplitPart{1U};
    uint32_t preFdDataNUM{0U};

    int64_t bN2Cost{0};
    uint32_t bN2Block{0U};
    bool isFinished{false};
    BatchCache batchCache{};
    MCache mCache{};
    CoreCache coreCache{};
};

// util
uint32_t GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
uint32_t GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo);
int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size, const BaseInfo &baseInfo);
Range<uint32_t> CalcS2Range(uint32_t mIdx, const BaseInfo &baseInfo, const SplitParam &splitParam,
                            const BatchCache &batchCache);
int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t mTailSize,
                                 uint32_t s2TailSize);

// cache calculation
void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
void CalcMCache(uint32_t mIdx, const SplitContext &splitContext, const BatchCache &batchCache, MCache &mCache);
void CopyTmpResult(FAMetaData &tmpRes, FAMetaData &splitRes);
void ClearTmpResult(FAMetaData &tmpResult);

// preprocess
void CalcSplitInfo(SplitContext &splitContext);
void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
void CalcCostInfo(SplitContext &splitContext);

// assign
void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

// FD
bool IsNeedRecordFDInfo(const AssignContext &assignContext, const FAMetaData &splitRes);
void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, FAMetaData &result);

// main
void SplitFD(FAMetaData &result);
void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, FAMetaData &result);
bool CheckChooseWithFd(int64_t noFdCost, int64_t withFdCost, uint32_t actualMFdsize, const SplitParam &param);
void LogSplitCoreInput(const BaseInfo &baseInfo, const SplitParam &param);
void LogSplitCoreResult(const FAMetaData &result);
void SplitCore(uint32_t coreNum, const BaseInfo &baseInfo, const SplitParam &param, FAMetaData &result);
} // namespace split_core_v2
#endif
