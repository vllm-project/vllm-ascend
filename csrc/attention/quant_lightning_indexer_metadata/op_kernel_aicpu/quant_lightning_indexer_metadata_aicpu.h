/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file quant_lightning_indexer_metadata_aicpu.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H
#define QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H

#include <string>
#include <vector>
#include <array>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;

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

enum class ValidSocVersion {
    ASCEND910B = 0,
    ASCEND910D,
    RESERVED_VERSION = 99999
};

template<class T>
using Range = std::pair<T, T>;

template<class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template<typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    uint32_t fdUsedVecNum { 0U };             // 归约过程使用的vector数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> fdBN2Idx {};          // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> fdMIdx {};            // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> fdWorkspaceIdx {};    // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> fdS2SplitNum {};      // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    std::vector<uint32_t> fdMSize {};           // 每个归约任务m轴大小，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> fdIdx {};             // FD负载均衡阶段，每个vector处理的归约任务对应ID
    std::vector<uint32_t> fdMStart {};          // FD负载均衡阶段，每个vector处理的归约任务的m轴起点
    std::vector<uint32_t> fdMNum {};            // FD负载均衡阶段，每个vector处理的归约任务的m轴行数

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum) :
        fdBN2Idx(aicNum),
        fdMIdx(aicNum),
        fdWorkspaceIdx(aicNum),
        fdS2SplitNum(aicNum),
        fdMSize(aicNum),
        fdIdx(aivNum),
        fdMStart(aivNum),
        fdMNum(aivNum) {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };        // 使用的核数量
    std::vector<uint32_t> bN2End {};    // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};    // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};     // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};     // 每个核第一份归约任务的存放位置
    int64_t maxCost { 0 };            // 慢核开销
    uint32_t numOfFdHead { 0U };        // 归约任务数量
    uint32_t maxS2SplitNum { 0U };      // 单个归约任务最大分核数量
    FlashDecodeResult fdRes { 0U, 0U };     // FD信息

    SplitResult(uint32_t aicNum, uint32_t aivNum) :
        bN2End(aicNum),
        gS1End(aicNum),
        s2End(aicNum),
        firstFdDataWorkspaceIdx(aicNum),
        fdRes(aicNum, aivNum) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum {};                   // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum {};                    // S2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize {};                  // S1G方向，尾块size
    std::vector<uint32_t> s2TailSize {};                   // S2方向，尾块size
    bool isKvSeqAllZero { true };

    explicit SplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize) {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};           // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch {};          // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};  // batch最后一块的开销
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };
    uint64_t maxS1GCost { 0 }; //新增

    explicit CostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    SplitInfo splitInfo { 0U };
    CostInfo costInfo { 0U };

    explicit SplitContext(uint32_t batchSize) :
        splitInfo(batchSize),
        costInfo(batchSize) {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    BlockCost<int64_t> typeCost {};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx { 0U };
    uint32_t s1GIdx { 0U };
    uint32_t s2Start { 0U };
    uint32_t s2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit { 0 };  // 负载上限
    int64_t cost { 0 };       // 已分配负载
    uint32_t block { 0U };      // 已分配块数
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t usedCoreNum { 0U };
    uint32_t curKvSplitPart { 1U };
    uint32_t curFdDataNum { 1U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchCache {};
    S1GCache s1GCache {};
    CoreCache coreCache {};
};
class QuantLightningIndexerMetadataCpuKernel : public CpuKernel {
public:
    QuantLightningIndexerMetadataCpuKernel() = default;
    ~QuantLightningIndexerMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool ParamsInit();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetaData(SplitResult &splitRes);
    ValidSocVersion ProcessSocVersion();

  // util
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    uint32_t GetSparseSeqSize(uint32_t bIdx);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    Range<uint32_t> CalcS2Range(uint32_t s1GIdx,const BatchCache &batchCache);
    int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
        uint32_t s2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);

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
    void AssignBlockToCore(uint32_t coreNum, const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result);
                                                               
    // FD
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    // main
    void SplitFD(SplitResult &splitRes);
    void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
    void SplitCore();

private:
    CpuKernelContext* context_ = nullptr;
    // input
    Tensor *actSeqLenQ_ = nullptr;
    Tensor *actSeqLenKV_ = nullptr;
    // output
    Tensor *metaData_ = nullptr;
    // attributes
    std::string socVersion_ = "";//新增
    bool supportFd_ = false; //新增。是否开启LD
    uint32_t cmpRatio_ = 4; //新增,LIQ压缩率
    uint32_t aicCoreNum_ = 24U;
    uint32_t aivCoreNum_ = 48U;
    uint32_t batchSize_ = 0;
    uint32_t maxSeqlenQ_ = 0;
    uint32_t maxSeqlenK_ = 0;
    uint32_t numHeadsQ_ = 0;
    uint32_t numHeadsK_ = 0;
    uint32_t headDim_ = 0;
    uint32_t topKSize_ = 0;
    uint32_t queryQuantMode_ = 0;
    uint32_t keyQuantMode_ = 0;
    uint32_t sparseCount_ = 0;
    uint32_t sparseBlockSize_ = 0;
    uint32_t sparseBlockCount_ = 0; // new
    std::string layoutQuery_ = "BSND";
    std::string layoutKV_ = "BSND";
    uint32_t sparseMode_ = 0;
    uint32_t attentionMode_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t sparseSharedSize_ = 0;

    // SplitParams
    uint32_t coreNum_ = 24U; // new
    int64_t  preToken_ = INT64_MAX;
    int64_t  nextToken_ = INT64_MAX;
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t s2BaseSize_ = 0;
    uint32_t gS1BaseSizeOfFd_ = 0;
    bool isS1G_ = true;

private:
    enum class ParamId : uint32_t {
        // input
        actSeqLenQ = 0,
        actSeqLenKV = 1,
        // output
        metaData = 0,
    };
};
} // namespace aicpu
#endif