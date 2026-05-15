/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_metadata_aicpu.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum BlockType : uint32_t {
    WIN_NORMAL_BLOCK = 0,
    WIN_TAIL_BLOCK,
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
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2IdxOfFdHead {};           // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> gS1IdxOfFdHead {};           // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> s2SplitNumOfFdHead {};       // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> gS1SplitNumOfFdHead {};      // 每个归约任务m轴切分份数，脚标为归约任务的序号
    std::vector<uint32_t> gS1LastPartSizeOfFdHead {};  // 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
    std::vector<uint32_t> gS1IdxEndOfFdHead {};        // FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
    std::vector<uint32_t> gS1IdxEndOfFdHeadSplit {};   // FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
    // 3、每个core处理的第1个归约任务的数据应存放的workspace位置
    std::vector<uint32_t> s2SplitStartIdxOfCore {};

    FlashDecodeResult(uint32_t coreNum, uint32_t vecCubeRatio) :
        bN2IdxOfFdHead(coreNum),
        gS1IdxOfFdHead(coreNum),
        s2SplitNumOfFdHead(coreNum),
        gS1SplitNumOfFdHead(coreNum),
        gS1LastPartSizeOfFdHead(coreNum),
        gS1IdxEndOfFdHead(coreNum * vecCubeRatio),
        gS1IdxEndOfFdHeadSplit(coreNum * vecCubeRatio),
        s2SplitStartIdxOfCore(coreNum) {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };        // 使用的核数量
    uint32_t vecCubeRatio { 0U };        // vec 与 cube 核数比例
    std::vector<uint32_t> bN2End {};    // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};    // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};     // 每个核处理数据的S2结束点
    int64_t maxCost { 0 };            // 慢核开销
    uint32_t numOfFdHead { 0U };        // 归约任务数量
    uint32_t maxS2SplitNum { 0U };      // 单个归约任务最大分核数量
    uint32_t usedVecNumOfFd { 0U };     // 归约过程使用的vector数量
    FlashDecodeResult fdRes { 0U, 0U };     // FD信息

    SplitResult(uint32_t coreNum, uint32_t ratio) :
        bN2End(coreNum),
        vecCubeRatio(ratio),
        gS1End(coreNum),
        s2End(coreNum),
        fdRes(coreNum, ratio) {};
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
    int64_t maxS1GCost { 0 };  // 记录所有S1G行中的最大开销

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
    uint32_t winS2Start { 0U };
    uint32_t winS2End { 0U };
    uint32_t cmpS2Start { 0U };  // win部分与cmp部分的切分点
    uint32_t cmpS2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
    uint32_t winS1GBlock { 0U };
    int64_t winS1GCost { 0 };
    int64_t winS1GLastBlockCost { 0 };
    int64_t winS1GNormalBlockCost { 0 };
    uint32_t cmpS1GBlock { 0U };
    int64_t cmpS1GCost { 0 };
    int64_t cmpS1GLastBlockCost { 0 };
    int64_t cmpS1GNormalBlockCost { 0 };
    int64_t cmpS2TailSize {0};
    int64_t winS2TailSize {0};
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

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchCache {};
    S1GCache s1GCache {};
    CoreCache coreCache {};
};

class KvQuantSparseAttnSharedkvMetadataCpuKernel : public CpuKernel {
public:
  KvQuantSparseAttnSharedkvMetadataCpuKernel() = default;
  ~KvQuantSparseAttnSharedkvMetadataCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  bool Prepare(CpuKernelContext &ctx);
  bool ParamsCheck();
  bool ParamsInit();
  bool BalanceSchedule();
  bool GenMetaData();
  ValidSocVersion ProcessSocVersion();
  // util
  uint32_t GetS1SeqSize(uint32_t bIdx);
  uint32_t GetS2SeqSize(uint32_t bIdx);
  int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
  int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
  Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache);
  int64_t WinCalcCost(uint32_t basicM, uint32_t basicS2);
  int64_t CmpCalcCost(uint32_t basicM, uint32_t basicS2);
  void CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
  uint32_t winS2TailSize, uint32_t cmpS2TailSize);

  // cache calculation
  void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
  void CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
  void CalcWinS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
  void CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
  void GatherWinAndCmpCache(S1GCache &s1GCache);
  void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);

  // preprocess
  void CalcSplitInfo(SplitContext &splitContext);
  void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
  void CalcCostInfo(SplitContext &splitContext);

  // assign
  void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
  void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
  void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
  int64_t CalcCurBlockCost(AssignContext &assignContext);
  void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
  void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);
  void AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result);

  // FD
  bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
  void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

  // main
  void SplitFD(SplitResult &result);
  void CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
  void SplitCore();

private:
  // input
  Tensor *actSeqLenQ_ = nullptr;
  Tensor *actSeqLenOriKV_ = nullptr;
  Tensor *actSeqLenCmpKV_ = nullptr;
  Tensor *SeqUsedQ_ = nullptr;
  Tensor *SeqUsedKV_ = nullptr;

  // output
  Tensor *metaData_ = nullptr;

  // attributes
  uint32_t batchSize_ = 0;
  uint32_t querySeqSize_ = 0;
  uint32_t queryHeadNum_ = 0;
  uint32_t kvSeqSize_ = 0;
  uint32_t kvHeadNum_ = 0;
  uint32_t headDim_ = 0;
  uint32_t oriTopK_ = 0;
  uint32_t cmpTopK_ = 0;
  uint32_t cmpRatio_ = -1;
  uint32_t winMaskMode_ = 4;
  uint32_t cmpMaskMode_ = 3;
  int64_t winLeft_ = 127;
  int64_t winRight_ = 0;
  std::string layoutQuery_ = "BSND";
  std::string layoutKV_ = "PA_ND";
  bool hasOriKV_ = true;
  bool hasCmpKV_ = true;
  uint32_t aicCoreNum_ = 24U;
  uint32_t aivCoreNum_ = 48U;

  // attr
  uint32_t coreNum_ = 24U; // new
  std::string socVersion_ = "ascend910B";
  int64_t preToken_ = 0; // new
  int64_t nextToken_ = 0; // new
  uint32_t groupSize_ = 0;
  uint32_t mBaseSize_ = 0;
  uint32_t s2BaseSize_ = 0;
  uint32_t gS1BaseSizeOfFd_ = 0;
  bool isS1G_ = true;
  SplitResult splitRes_ {36, 2};
  bool isCFA = false;
  bool isSCFA = false;
  bool supportFd = false;
  uint32_t sparseMode_ = 0;
  uint32_t attentionMode_ = 1;
  BlockCost<int64_t> typeCost_;
private:
  enum class ParamId : uint32_t {
    // input
    actSeqLenQ = 0,
    actSeqLenOriKV = 1,
    actSeqLenCmpKV = 2,
    SeqUsedQ = 3,
    SeqUsedKV = 4,
    // output
    metaData = 0,
  };
};
} // namespace aicpu

#endif