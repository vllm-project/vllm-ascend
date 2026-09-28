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
 * \file sparse_flash_mla_metadata_aicpu.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_METADATA_AICPU_H
#define SPARSE_FLASH_MLA_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h"
#include "../../common/op_kernel/aicpu_common.h"

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t COST_WEIGHT_M = 6U;
constexpr uint32_t COST_WEIGHT_S2 = 10U;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32U;
constexpr bool ORI_KV = false;
constexpr bool CMP_KV = true;
constexpr uint32_t NO_MASK = 0;
constexpr uint32_t HAS_MASK = 1;

enum BlockType : uint32_t {
    ORI_NORMAL_BLOCK = 0,
    ORI_TAIL_BLOCK,
    CMP_NORMAL_BLOCK,
    CMP_TAIL_BLOCK,
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
    ASCEND910 = 0,
    ASCEND950
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
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

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    uint32_t fdUsedVecNum{0U}; // 归约过程使用的vector数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> fdBN2Idx{}; // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> fdMIdx{};         // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> fdWorkspaceIdx{}; // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> fdS2SplitNum{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    std::vector<uint32_t> fdMSize{};      // 每个归约任务m轴大小，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> fdIdx{};    // FD负载均衡阶段，每个vector处理的归约任务对应ID
    std::vector<uint32_t> fdMStart{}; // FD负载均衡阶段，每个vector处理的归约任务的m轴起点
    std::vector<uint32_t> fdMNum{};   // FD负载均衡阶段，每个vector处理的归约任务的m轴行数

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum)
        : fdBN2Idx(aicNum),
          fdMIdx(aicNum),
          fdWorkspaceIdx(aicNum),
          fdS2SplitNum(aicNum),
          fdMSize(aicNum),
          fdIdx(aivNum),
          fdMStart(aivNum),
          fdMNum(aivNum)
    {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum{0U};                        // 使用的核数量
    std::vector<uint32_t> bN2End{};                  // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End{};                  // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End{};                   // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx{}; // 每个核第一份归约任务的存放位置
    int64_t maxCost{0};                              // 慢核开销
    uint32_t numOfFdHead{0U};                        // 归约任务数量
    uint32_t maxS2SplitNum{0U};                      // 单个归约任务最大分核数量
    uint32_t maxS2LoopNum{0U};                       // 单个核最大s2计算轮次数量
    FlashDecodeResult fdRes{0U, 0U};                 // FD信息

    SplitResult(uint32_t aicNum, uint32_t aivNum)
        : bN2End(aicNum),
          gS1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum),
          fdRes(aicNum, aivNum) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum{};    // S1G方向，切了多少个基本块
    std::vector<uint32_t> oriS2BaseNum{};  // oriS2方向，切了多少个基本块
    std::vector<uint32_t> cmpS2BaseNum{};  // cmpS2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize{};   // S1G方向，尾块size
    std::vector<uint32_t> oriS2TailSize{}; // oriS2方向，尾块size
    std::vector<uint32_t> cmpS2TailSize{}; // cmpS2方向，尾块size
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : s1GBaseNum(batchSize),
          oriS2BaseNum(batchSize),
          cmpS2BaseNum(batchSize),
          s1GTailSize(batchSize),
          oriS2TailSize(batchSize),
          cmpS2TailSize(batchSize)
    {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};          // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch{};        // 整个batch的开销
    std::vector<uint32_t> bN2S2LoopOfEachBatch{};       // 整个batch的s2计算轮次数量
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{}; // batch最后一块的开销
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxS1GCost{0}; // 记录所有S1G行中的最大开销

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize),
          bN2BlockOfEachBatch(batchSize),
          bN2S2LoopOfEachBatch(batchSize),
          bN2LastBlockCostOfEachBatch(batchSize)
    {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(uint32_t batchSize)
        : splitInfo(batchSize),
          costInfo(batchSize)
    {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t oriS2Size{0U};
    uint64_t cmpRevertS2Size{0U};
    int64_t oriPreTokenLeftUp{0};
    int64_t oriNextTokenLeftUp{0};
    int64_t cmpPreTokenLeftUp{0};
    int64_t cmpNextTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx{0U};
    uint32_t s1GIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    uint32_t oriS2Start{0U};
    uint32_t oriS2End{0U};
    uint32_t cmpS2Start{0U}; // win部分与cmp部分的切分点
    uint32_t cmpS2End{0U};
    int64_t s1GCost{0};
    int64_t s1GLastBlockCost{0};
    uint32_t s1GBlock{0U};
    uint32_t s2Loop{0U};
    uint32_t oriS1GBlock{0U};
    int64_t oriS1GCost{0};
    int64_t oriS1GLastBlockCost{0};
    int64_t oriS1GNormalBlockCost{0};
    uint32_t cmpS1GBlock{0U};
    int64_t cmpS1GCost{0};
    int64_t cmpS1GLastBlockCost{0};
    int64_t cmpS1GNormalBlockCost{0};
    int64_t oriS2TailSize{0};
    int64_t cmpS2TailSize{0};
    uint32_t actOriS2Size{0U};
    uint32_t actCmpS2Size{0U};
    uint32_t reductionBlockSize{0U};
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit{0}; // 负载上限
    int64_t cost{0};      // 已分配负载
    uint32_t block{0U};   // 已分配块数
    uint32_t s2Loop{0U};  // 已分配s2计算轮次数量
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curS1GIdx{0U};
    uint32_t curS2Idx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t curKvSplitPart{1U};
    uint32_t preFdDataNum{0U};

    int64_t bN2Cost{0};
    uint32_t bN2Block{0U};
    uint32_t bN2S2Loop{0U};
    bool isFinished{false};
    BatchCache batchCache{};
    S1GCache s1GCache{};
    CoreCache coreCache{};
};

class SparseFlashMlaMetadataCpuKernel : public CpuKernel {
public:
    SparseFlashMlaMetadataCpuKernel() = default;
    ~SparseFlashMlaMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    int32_t GetQueryBatchSize();
    void CalcOriMaskMode();
    void CalcCmpMaskMode();
    ValidSocVersion ProcessSocVersion();
    bool ParamsCheck();
    int32_t GetSumOfQuerySeq();
    bool ParamsInit();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetadata(SplitResult &splitRes);
    // util
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetOriS2SeqSize(uint32_t bIdx);
    uint32_t GetCmpS2SeqSize(uint32_t bIdx);
    uint64_t GetRevertS2Size(uint32_t bIdx);
    uint32_t GetS1Idx(uint32_t s1Size, uint32_t s1GIdx);
    uint32_t GetBsStride(uint32_t bIdx, uint32_t s1Idx);
    uint32_t GetOriTopkLength(uint32_t bsStride);
    uint32_t GetOriTopkLength(uint32_t s1GIdx, const BatchCache &batchCache) const;
    uint32_t ReadOriTopkLengthAtRow(uint32_t bIdx, uint32_t s1Idx) const;
    uint32_t GetCmpTopkLength(uint32_t bsStride);
    int64_t CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
    int64_t CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
    Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache, bool isCmpKv);
    int64_t OriCalcCost(uint32_t basicM, uint32_t basicS2);
    int64_t CmpCalcCost(uint32_t basicM, uint32_t basicS2);
    void CalcCostTable(uint32_t s1GTailSize, uint32_t reductionBlockSize, uint32_t oriS2TailSize,
                       uint32_t cmpS2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcOriBlockRange(const Range<int64_t> &oriS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
    void CalcCmpBlockRange(const Range<int64_t> &cmpS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
    void CalcOriS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void GatherOriAndCmpCache(S1GCache &s1GCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache,
                      S1GCache &s1GCache);

    // preprocess
    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    // assign
    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    int64_t CalcCurBlockCost(const AssignContext &assignContext);
    uint32_t CalcCurBlockS2Loop(const AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result);

    // FD
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    bool IsFirstReductionBlock(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    // main
    void SplitFD(SplitResult &splitRes);
    void CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, SplitResult &result);

private:
    // input
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensOriKv_ = nullptr;
    Tensor *cuSeqlensCmpKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedOriKv_ = nullptr;
    Tensor *sequsedCmpKv_ = nullptr;
    Tensor *cmpResidualKv_ = nullptr;
    Tensor *oriTopkLength_ = nullptr;
    Tensor *cmpTopkLength_ = nullptr;

    // output
    Tensor *metadata_ = nullptr;

    // attributes
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t maxSeqlenOriKv_ = 0;
    int32_t maxSeqlenCmpKv_ = 0;
    int32_t numHeadsKv_ = 1;
    int32_t headDim_ = 0;
    int32_t oriTopK_ = 0;
    int32_t cmpTopK_ = 0;
    int32_t cmpRatio_ = 0;
    int32_t oriMaskMode_ = static_cast<int32_t>(SparseMode::BAND);
    int32_t cmpMaskMode_ = static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    int64_t oriWinLeft_ = 127;
    int64_t oriWinRight_ = 0;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "PA_BBND";
    bool hasOriKv_ = true;
    bool hasCmpKv_ = true;
    uint32_t aicCoreNum_ = optiling::AIC_CORE_MAX_NUM;
    uint32_t aivCoreNum_ = optiling::AIV_CORE_MAX_NUM;
    bool isBatchConsistency_ = false;

    // attr
    std::string socVersion_ = "Ascend950";
    int64_t oriPreToken_ = 0;
    int64_t oriNextToken_ = 0;
    int64_t cmpPreToken_ = 0;
    int64_t cmpNextToken_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 0;
    uint32_t s2BaseSize_ = 128U;
    bool isS1G_ = true;
    bool supportFd_ = false;
    uint32_t oriAttentionMode_ = HAS_MASK;
    uint32_t cmpAttentionMode_ = HAS_MASK;
    BlockCost<int64_t> typeCost_ = {};
    bool isSplitG_ = false;
    bool isSparseOriKv_ = false;
    bool isSparseCmpKv_ = false;
    bool hasOriTopkLength_ = false;
    uint32_t remainedBlockNum_ = 0U;

private:
    enum class ParamId : uint32_t {
        // input
        cuSeqlensQ = 0,
        cuSeqlensOriKv = 1,
        cuSeqlensCmpKv = 2,
        sequsedQ = 3,
        sequsedOriKv = 4,
        sequsedCmpKv = 5,
        cmpResidualKv = 6,
        oriTopkLength = 7,
        cmpTopkLength = 8,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
