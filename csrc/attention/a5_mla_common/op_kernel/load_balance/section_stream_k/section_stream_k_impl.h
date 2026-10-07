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
 * \file section_stream_k_impl.h
 * \brief SectionStreamK具体实现
 */

#ifndef SECTION_STREAM_K_IMPL_H
#define SECTION_STREAM_K_IMPL_H

#include <cmath>
#include <algorithm>
#include "section_stream_k_def.h"
#include "../load_balance_common.h"

namespace load_balance {

class SectionStreamKImpl {
public:
    struct SectionStreamKImplResult {
        int64_t maxCost{0}; // 慢核开销

        // FA
        uint32_t usedCoreNum{0U};                        // 使用的核数量
        std::vector<uint32_t> bNEnd{};                   // 每个核处理数据的BN结束点
        std::vector<uint32_t> mEnd{};                    // 每个核处理数据的M结束点
        std::vector<uint32_t> s2End{};                   // 每个核处理数据的S2结束点
        uint32_t maxS2SplitNum{0U};                      // 单个归约任务最大分核数量
        uint32_t fdTaskNum{0U};                          // 归约任务数量
        std::vector<uint32_t> firstFdDataWorkspaceIdx{}; // 每个核第一份归约任务的存放地址

        // FD
        uint32_t usedVecNum{0U}; // 归约过程使用的vector数量
        // 1、归约任务的索引信息
        std::vector<uint32_t> bNIdx{}; // 每个归约任务的BN索引，脚标为归约任务的序号，最大为核数-1
        std::vector<uint32_t> mIdx{};         // 每个归约任务的m索引，脚标为归约任务的序号
        std::vector<uint32_t> workspaceIdx{}; // 每个归约任务在workspace中的存放位置
        std::vector<uint32_t> s2SplitNum{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        std::vector<uint32_t> mSize{};      // 每个归约任务的M轴大小，脚标为归约任务的序号
        // 2、FD kernel阶段，归约任务的分核信息
        std::vector<uint32_t> taskIdx{}; // 每个vector处理的归约任务对应ID
        std::vector<uint32_t> mStart{};  // 每个vector处理的归约任务的M轴起点
        std::vector<uint32_t> mLen{};    // 每个vector处理的归约任务的M轴行数

        explicit SectionStreamKImplResult(uint32_t aicNum, uint32_t aivNum)
            : bNEnd(aicNum),
              mEnd(aicNum),
              s2End(aicNum),
              firstFdDataWorkspaceIdx(aicNum),
              bNIdx(aicNum),
              mIdx(aicNum),
              workspaceIdx(aicNum),
              s2SplitNum(aicNum),
              mSize(aicNum),
              taskIdx(aivNum),
              mStart(aivNum),
              mLen(aivNum)
        {}

        inline void Clear()
        {
            maxCost = INT64_MIN;
            usedCoreNum = 0U;
            maxS2SplitNum = 0U;
            fdTaskNum = 0U;
            std::fill(bNEnd.begin(), bNEnd.end(), 0U);
            std::fill(mEnd.begin(), mEnd.end(), 0U);
            std::fill(s2End.begin(), s2End.end(), 0U);
            std::fill(firstFdDataWorkspaceIdx.begin(), firstFdDataWorkspaceIdx.end(), 0U);

            usedVecNum = 0U;
            std::fill(bNIdx.begin(), bNIdx.end(), 0U);
            std::fill(mIdx.begin(), mIdx.end(), 0U);
            std::fill(workspaceIdx.begin(), workspaceIdx.end(), 0U);
            std::fill(s2SplitNum.begin(), s2SplitNum.end(), 0U);
            std::fill(mSize.begin(), mSize.end(), 0U);
            std::fill(taskIdx.begin(), taskIdx.end(), 0U);
            std::fill(mStart.begin(), mStart.end(), 0U);
            std::fill(mLen.begin(), mLen.end(), 0U);
        }
    };

public:
    inline std::vector<SectionStreamKImplResult> Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo);
    inline uint32_t SetParam(const SectionStreamKParam &param);

private:
    enum BlockType : uint32_t {
        NORMAL_BLOCK = 0,
        TAIL_BLOCK,
        BLOCK_MAX_TYPE
    };

    struct GridInfo {
        uint32_t sectionNum{0U};              // section
        std::vector<uint32_t> sectionBnIdx{}; // OutputLayout对应BN方向，section的切分点
        std::vector<uint32_t> mBaseNum{};     // M方向，切了多少个基本块
        std::vector<uint32_t> s2BaseNum{};    // S2方向，切了多少个基本块
        std::vector<uint32_t> mTailSize{};    // M方向，尾块size
        std::vector<uint32_t> s2TailSize{};   // S2方向，尾块size
        bool isEmpty{true};

        explicit GridInfo(uint32_t batchSize)
            : mBaseNum(batchSize),
              s2BaseNum(batchSize),
              mTailSize(batchSize),
              s2TailSize(batchSize)
        {}
    };

    struct CostInfo {
        std::vector<int64_t> bNCostOfEachBatch{};          // 整个batch的开销
        std::vector<uint32_t> bNBlockOfEachBatch{};        // 整个batch的开销
        std::vector<int64_t> bNLastBlockCostOfEachBatch{}; // batch最后一块的开销
        std::vector<uint32_t> sectionBlockNum{};
        std::vector<int64_t> sectionCost{};

        explicit CostInfo(uint32_t batchSize)
            : bNCostOfEachBatch(batchSize),
              bNBlockOfEachBatch(batchSize),
              bNLastBlockCostOfEachBatch(batchSize)
        {}
    };

    struct ComputeContext {
        const DeviceInfo &deviceInfo;
        const IBaseInfo &baseInfo;
        GridInfo gridInfo;
        CostInfo costInfo;

        explicit ComputeContext(const DeviceInfo &dInfo, const IBaseInfo &bInfo)
            : deviceInfo(dInfo),
              baseInfo(bInfo),
              gridInfo(bInfo.GetBatchSize()),
              costInfo(bInfo.GetBatchSize())
        {}
    };

    // 分核功能模块内部使用：记录batch相关的临时信息
    struct BatchCache {
        uint32_t bIdx{0U};
        uint32_t s1Size{0U};
        uint32_t s2Size{0U};
        int64_t preTokenLeftUp{0};
        int64_t nextTokenLeftUp{0};
        Table<int64_t, BLOCK_MAX_TYPE, BLOCK_MAX_TYPE> typeCost{};
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

    struct AssignContext {
        uint32_t curSectionIdx{0U};
        uint32_t curBIdx{0U};
        uint32_t curBNIdx{0U};
        uint32_t curMIdx{0U};
        uint32_t curS2Idx{0U};
        uint32_t curCoreIdx{0U};
        int64_t unassignedCost{0};
        uint32_t usedCoreNum{0U};
        uint32_t curKvSplitPart{1U};
        uint32_t preFdDataNum{0U};
        int64_t bNCost{0};
        uint32_t bNBlock{0U};
        bool isFinished{false};
        BatchCache batchCache{};
        MCache mCache{};
        CoreCache coreCache{};
    };

    struct FaConfig {
        bool FdOn{true};
        uint32_t coreNum{0U};
        int64_t coreLimit{0U};
        int64_t costLimit{INT64_MIN};
        uint32_t sectionIdx{0U};
    };

    using CostTable = Table<int64_t, BLOCK_MAX_TYPE, BLOCK_MAX_TYPE>;

private:
    inline static int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    inline uint32_t GetHeadNum(const IBaseInfo &baseInfo) const;
    inline uint32_t GetMSize(uint32_t batchIdx, const IBaseInfo &baseInfo) const;
    inline uint32_t ToOutputLayoutBnIdx(uint32_t bn2Idx, const IBaseInfo &baseInfo) const;
    inline CostTable CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t mTailSize,
                                   uint32_t s2TailSize);
    static inline Range<uint32_t> CalcCoreRange(uint32_t sectionIdx, const ComputeContext &computeContext);
    inline Range<uint32_t> CalcS2Range(uint32_t mIdx, const IBaseInfo &baseInfo, const BatchCache &batchCache);
    inline void CalcGridInfo(ComputeContext &computeContext);
    inline void CalcGridInfoSection(ComputeContext &computeContext);
    inline void CalcCostInfo(ComputeContext &computeContext);
    inline void CalcBatchCost(uint32_t bIdx, const ComputeContext &computeContext, CostInfo &costInfo);
    inline void CalcBatchCache(uint32_t bIdx, const ComputeContext &computeContext, BatchCache &batchCache);
    inline void CalcMCache(uint32_t mIdx, const ComputeContext &computeContext, const BatchCache &batchCache,
                           MCache &mCache);
    inline SectionStreamKImplResult ScheduleSection(uint32_t sectionIdx, const ComputeContext &computeContext);
    inline void ScheduleFa(const FaConfig &faConfig, const ComputeContext &computeContext,
                           SectionStreamKImplResult &result);
    static inline void ScheduleFd(uint32_t aivNum, SectionStreamKImplResult &result);
    static inline bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SectionStreamKImplResult &result);
    inline void RecordFDInfo(const ComputeContext &computeContext, const AssignContext &assignContext,
                             SectionStreamKImplResult &result);
    inline bool CheckChooseWithFd(uint32_t sectionNum, const SectionStreamKImplResult &noFd,
                                  const SectionStreamKImplResult &withFd);

    // assign
    inline void AssignByBatch(const ComputeContext &computeContext, AssignContext &assignContext);
    inline void AssignByRow(const ComputeContext &computeContext, AssignContext &assignContext);
    inline void AssignByBlock(AssignContext &assignContext);

private:
    SectionStreamKParam m_param{};
};

inline std::vector<SectionStreamKImpl::SectionStreamKImplResult> SectionStreamKImpl::Compute(
    const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo)
{
    ComputeContext computeContext{deviceInfo, baseInfo};
    std::vector<SectionStreamKImplResult> result{};

    CalcGridInfo(computeContext);
    // 全空case
    if (computeContext.gridInfo.isEmpty) {
        result.emplace_back(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        result[0].usedCoreNum = 1U;
        result[0].bNEnd[0] = baseInfo.GetBatchSize() * GetHeadNum(baseInfo);
        result[0].mEnd[0] = 0U;
        result[0].s2End[0] = 0U;
        return result;
    }
    CalcCostInfo(computeContext);

    for (uint32_t sectionIdx = 0; sectionIdx < computeContext.gridInfo.sectionNum; ++sectionIdx) {
        // 全mask case
        if (computeContext.costInfo.sectionBlockNum[sectionIdx] == 0U) {
            result.emplace_back(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
            result[sectionIdx].usedCoreNum = 1U;
            result[sectionIdx].bNEnd[0] = computeContext.gridInfo.sectionBnIdx[sectionIdx];
            result[sectionIdx].mEnd[0] = 0U;
            result[sectionIdx].s2End[0] = 0U;
            continue;
        }

        auto res = ScheduleSection(sectionIdx, computeContext);
        result.emplace_back(res);
    }

    return result;
}

inline uint32_t SectionStreamKImpl::SetParam(const SectionStreamKParam &param)
{
    if (param.faToleranceRatio == 0U || param.mBaseSize == 0U || param.s2BaseSize == 0U) {
        return SECTION_STREAM_K_ERROR_INVALID_PARAM;
    }

    m_param = param;
    if (m_param.costFunc == nullptr) {
        m_param.costFunc = CalcCost;
    }
    return SECTION_STREAM_K_SUCCESS;
}

inline void SectionStreamKImpl::CalcGridInfo(ComputeContext &computeContext)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;

    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    GridInfo &gridInfo = computeContext.gridInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        uint32_t s2Size = baseInfo.GetKvSeqSize(bIdx);
        uint32_t mSize = GetMSize(bIdx, baseInfo);

        gridInfo.mBaseNum[bIdx] = CeilDiv(mSize, m_param.mBaseSize);
        gridInfo.mTailSize[bIdx] = mSize % m_param.mBaseSize;
        gridInfo.s2BaseNum[bIdx] = CeilDiv(s2Size, m_param.s2BaseSize);
        gridInfo.s2TailSize[bIdx] = s2Size % m_param.s2BaseSize;
        if (gridInfo.mBaseNum[bIdx] != 0U && gridInfo.s2BaseNum[bIdx] != 0U) {
            gridInfo.isEmpty = false;
        }
    }
    CalcGridInfoSection(computeContext);
}

inline void SectionStreamKImpl::CalcGridInfoSection(ComputeContext &computeContext)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    GridInfo &gridInfo = computeContext.gridInfo;

    // 如果L2未设置，则不进行section切分
    if (m_param.l2Byte == 0U) {
        gridInfo.sectionBnIdx.emplace_back(baseInfo.GetBatchSize() * GetHeadNum(baseInfo));
        gridInfo.sectionNum = 1;
        return;
    }

    uint32_t bn2Idx = 0U;
    int64_t tokenLimit = static_cast<int64_t>(m_param.l2Byte);
    int64_t tokenSize = 0L;
    uint32_t maxMSize = 0U;
    int64_t maxSingleHeadTokenCost = 0U;
    int64_t headDimQk = static_cast<int64_t>(baseInfo.GetHeadDimQk());
    int64_t headDimV = static_cast<int64_t>(baseInfo.GetHeadDimV());
    int64_t s1TypeCost = static_cast<int64_t>(GetDataTypeByteSize(baseInfo.GetQueryDataType()));
    int64_t s2TypeCost = static_cast<int64_t>(GetDataTypeByteSize(baseInfo.GetKvDataType()));
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        int64_t s1Size = static_cast<int64_t>(baseInfo.GetQuerySeqSize(bIdx));
        int64_t s2Size = static_cast<int64_t>(baseInfo.GetKvSeqSize(bIdx));
        int64_t s1Cost = s1Size * headDimQk * s1TypeCost * 2L; // 2: Q和O两份数据
        int64_t s2vCost = s2Size * (headDimQk + headDimV) * s2TypeCost;
        int64_t singleHeadCost = s1Cost + s2vCost;
        maxSingleHeadTokenCost = std::max(maxSingleHeadTokenCost, singleHeadCost);
        maxMSize = std::max(maxMSize, GetMSize(bIdx, baseInfo));
        for (uint32_t n2Idx = 0; n2Idx < baseInfo.GetKvHeadNum(); ++n2Idx) {
            if (!IsWithinTolerance(tokenLimit, int64_t{0}, tokenSize + singleHeadCost) && tokenSize != 0) {
                gridInfo.sectionBnIdx.emplace_back(bn2Idx);
                gridInfo.sectionNum++;
                tokenSize = 0L;
            }
            tokenSize += singleHeadCost;
            bn2Idx++;
        }
    }
    // 最后一个section切分点
    gridInfo.sectionBnIdx.emplace_back(baseInfo.GetBatchSize() * baseInfo.GetKvHeadNum());
    gridInfo.sectionNum++;

    // 如果M轴小于最小基本块，则不进行section切分
    // 如果最大单个head数据量不超过每个核可用的L2容量，则不进行section切分
    if (maxMSize <= m_param.mBaseSize ||
        maxSingleHeadTokenCost <= tokenLimit / computeContext.deviceInfo.aicCoreMaxNum) {
        gridInfo.sectionBnIdx.clear();
        gridInfo.sectionBnIdx.emplace_back(baseInfo.GetBatchSize() * baseInfo.GetKvHeadNum());
        gridInfo.sectionNum = 1;
    }

    for (uint32_t &bnIdx : gridInfo.sectionBnIdx) {
        bnIdx = ToOutputLayoutBnIdx(bnIdx, baseInfo);
    }
}

inline void SectionStreamKImpl::CalcBatchCost(uint32_t bIdx, const ComputeContext &computeContext, CostInfo &costInfo)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    costInfo.bNCostOfEachBatch[bIdx] = 0;
    costInfo.bNBlockOfEachBatch[bIdx] = 0U;
    costInfo.bNLastBlockCostOfEachBatch[bIdx] = 0U;

    if (baseInfo.GetQuerySeqSize(bIdx) == 0U || baseInfo.GetKvSeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    MCache mCache;
    CalcBatchCache(bIdx, computeContext, bCache);
    for (uint32_t mIdx = 0; mIdx < gridInfo.mBaseNum[bIdx]; mIdx++) {
        CalcMCache(mIdx, computeContext, bCache, mCache);
        costInfo.bNCostOfEachBatch[bIdx] += mCache.mCost;
        costInfo.bNBlockOfEachBatch[bIdx] += mCache.mBlock;

        if (mCache.mBlock > 0) {
            costInfo.bNLastBlockCostOfEachBatch[bIdx] = mCache.mLastBlockCost;
        }
    }
}

inline SectionStreamKImpl::CostTable SectionStreamKImpl::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                                       uint32_t mTailSize, uint32_t s2TailSize)
{
    CostTable typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = m_param.costFunc(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (mTailSize == 0U) ? 0U : m_param.costFunc(mTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : m_param.costFunc(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] =
        (mTailSize == 0U || s2TailSize == 0U) ? 0U : m_param.costFunc(mTailSize, s2TailSize);
    return typeCost;
}

inline void SectionStreamKImpl::CalcBatchCache(uint32_t bIdx, const ComputeContext &computeContext,
                                               BatchCache &batchCache)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = baseInfo.GetQuerySeqSize(bIdx);
    batchCache.s2Size = baseInfo.GetKvSeqSize(bIdx);
    batchCache.preTokenLeftUp = baseInfo.GetPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = baseInfo.GetNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost =
        CalcCostTable(m_param.mBaseSize, m_param.s2BaseSize, gridInfo.mTailSize[bIdx], gridInfo.s2TailSize[bIdx]);
}

inline void SectionStreamKImpl::CalcMCache(uint32_t mIdx, const ComputeContext &computeContext,
                                           const BatchCache &batchCache, MCache &mCache)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    mCache.bIdx = batchCache.bIdx;
    mCache.mIdx = mIdx;

    auto s2Range = CalcS2Range(mIdx, baseInfo, batchCache);
    mCache.s2Start = s2Range.first;
    mCache.s2End = s2Range.second;

    if (mCache.s2Start >= mCache.s2End || gridInfo.mBaseNum[batchCache.bIdx] == 0) {
        mCache.mBlock = 0;
        mCache.mCost = 0;
        mCache.mLastBlockCost = 0;
        mCache.mNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    mCache.mBlock = mCache.s2End - mCache.s2Start;
    int64_t curTailS2Num =
        (gridInfo.s2TailSize[batchCache.bIdx] != 0U && mCache.s2End == gridInfo.s2BaseNum[batchCache.bIdx]) ? 1L : 0L;
    int64_t curNormalS2Num = static_cast<int64_t>(mCache.mBlock) - curTailS2Num;
    if (mIdx == (gridInfo.mBaseNum[batchCache.bIdx] - 1U) && gridInfo.mTailSize[batchCache.bIdx] != 0U) {
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

    if (mCache.mBlock == 1) {
        mCache.mLastBlockCost += m_param.v0Cost;
    }
    mCache.mCost += m_param.v0Cost;
}

inline void SectionStreamKImpl::CalcCostInfo(ComputeContext &computeContext)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    CostInfo &costInfo = computeContext.costInfo;
    costInfo.sectionBlockNum.resize(gridInfo.sectionNum);
    costInfo.sectionCost.resize(gridInfo.sectionNum);

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        CalcBatchCost(bIdx, computeContext, costInfo);
    }

    for (uint32_t sectionIdx = 0; sectionIdx < gridInfo.sectionBnIdx.size(); ++sectionIdx) {
        uint32_t bnStartIdx = (sectionIdx == 0U) ? 0U : gridInfo.sectionBnIdx[sectionIdx - 1U];
        uint32_t bnEndIdx = gridInfo.sectionBnIdx[sectionIdx];
        uint32_t headNum = GetHeadNum(baseInfo);

        uint32_t bIdx = 0;
        for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
            bIdx = FloorDiv(bnIdx, headNum);
            costInfo.sectionBlockNum[sectionIdx] += costInfo.bNBlockOfEachBatch[bIdx];
            costInfo.sectionCost[sectionIdx] += costInfo.bNCostOfEachBatch[bIdx];
        }
    }
}

inline int64_t SectionStreamKImpl::CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    int64_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    int64_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

inline uint32_t SectionStreamKImpl::GetHeadNum(const IBaseInfo &baseInfo) const
{
    if (m_param.outputLayout == OutputLayout::BN1_S1) {
        return baseInfo.GetQueryHeadNum();
    }
    return baseInfo.GetKvHeadNum();
}

inline uint32_t SectionStreamKImpl::GetMSize(uint32_t batchIdx, const IBaseInfo &baseInfo) const
{
    uint32_t s1Size = baseInfo.GetQuerySeqSize(batchIdx);
    if (m_param.outputLayout == OutputLayout::BN1_S1) {
        return s1Size;
    }
    return s1Size * baseInfo.GetGroupSize();
}

inline uint32_t SectionStreamKImpl::ToOutputLayoutBnIdx(uint32_t bn2Idx, const IBaseInfo &baseInfo) const
{
    if (m_param.outputLayout == OutputLayout::BN1_S1) {
        return bn2Idx * baseInfo.GetGroupSize();
    }
    return bn2Idx;
}

inline Range<uint32_t> SectionStreamKImpl::CalcCoreRange(uint32_t sectionIdx, const ComputeContext &computeContext)
{
    const DeviceInfo &deviceInfo = computeContext.deviceInfo;

    uint32_t maxCore = std::min(deviceInfo.aicCoreMaxNum, computeContext.costInfo.sectionBlockNum[sectionIdx]);
    uint32_t minCore = static_cast<uint32_t>(
        std::lround(std::sqrt(static_cast<float>(computeContext.costInfo.sectionBlockNum[sectionIdx]) + 0.25f) + 0.5f));
    minCore = std::max(minCore, deviceInfo.aicCoreMinNum);
    minCore = std::min(minCore, maxCore);

    return std::make_pair(minCore, maxCore);
}

inline Range<uint32_t> SectionStreamKImpl::CalcS2Range(uint32_t mIdx, const IBaseInfo &baseInfo,
                                                       const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // 实际序列长度为0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // 无mask
    if (baseInfo.GetSparseMode() == SparseMode::BUTT) {
        s2Start = 0U;
        s2End = CeilDiv(batchCache.s2Size, m_param.s2BaseSize);
        return std::make_pair(s2Start, s2End);
    }

    // 1. 将当前M轴基本块映射回其覆盖的S1 token范围
    int64_t mFirstToken = static_cast<int64_t>(mIdx) * static_cast<int64_t>(m_param.mBaseSize);
    int64_t mLastToken = NumToIndex(std::min(mFirstToken + static_cast<int64_t>(m_param.mBaseSize),
                                             static_cast<int64_t>(GetMSize(batchCache.bIdx, baseInfo))));

    int64_t s1FirstToken;
    int64_t s1LastToken;
    if (m_param.outputLayout == OutputLayout::BN1_S1) {
        // 非合轴模式：M轴即S1轴，无需按G映射
        s1FirstToken = mFirstToken;
        s1LastToken = mLastToken;
    } else {
        // 合轴模式：M轴为S1/G合轴，根据实际排布映射回S1轴
        if (baseInfo.GetIsS1G()) {
            s1FirstToken = mFirstToken / static_cast<int64_t>(baseInfo.GetGroupSize());
            s1LastToken = mLastToken / static_cast<int64_t>(baseInfo.GetGroupSize());
        } else {
            const int64_t firstGroupIdx = mFirstToken / static_cast<int64_t>(batchCache.s1Size);
            const int64_t lastGroupIdx = mLastToken / static_cast<int64_t>(batchCache.s1Size);
            if (firstGroupIdx == lastGroupIdx) {
                s1FirstToken = mFirstToken % static_cast<int64_t>(batchCache.s1Size);
                s1LastToken = mLastToken % static_cast<int64_t>(batchCache.s1Size);
            } else {
                s1FirstToken = 0;
                s1LastToken = batchCache.s1Size;
            }
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. 将token索引转换为S2基本块索引
    // 无有效token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // 获取有效范围
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(NumToIndex(batchCache.s2Size)));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(NumToIndex(batchCache.s2Size)));
    s2Start = static_cast<uint32_t>(s2FirstToken) / m_param.s2BaseSize;
    s2End = ToOpenInterval(static_cast<uint32_t>(s2LastToken) / m_param.s2BaseSize); // 基本块结束索引

    return std::make_pair(s2Start, s2End);
}

inline SectionStreamKImpl::SectionStreamKImplResult SectionStreamKImpl::ScheduleSection(
    uint32_t sectionIdx, const ComputeContext &computeContext)
{
    const DeviceInfo &deviceInfo = computeContext.deviceInfo;

    SectionStreamKImplResult bestResultNoFd(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
    bestResultNoFd.maxCost = INT64_MAX;
    bestResultNoFd.usedCoreNum = deviceInfo.aicCoreMaxNum;
    SectionStreamKImplResult bestResultWithFd(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
    bestResultWithFd.maxCost = INT64_MAX;
    bestResultWithFd.usedCoreNum = deviceInfo.aicCoreMaxNum;
    SectionStreamKImplResult tmpResult(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
    FaConfig faNoFdConfig{};
    FaConfig faWithFdConfig{};
    faNoFdConfig.FdOn = false;
    faNoFdConfig.sectionIdx = sectionIdx;
    faNoFdConfig.coreLimit = bestResultNoFd.usedCoreNum;
    faNoFdConfig.costLimit = bestResultNoFd.maxCost;
    faWithFdConfig.FdOn = true;
    faWithFdConfig.sectionIdx = sectionIdx;
    faWithFdConfig.coreLimit = bestResultWithFd.usedCoreNum;
    faWithFdConfig.costLimit = bestResultWithFd.maxCost;

    auto coreRange = CalcCoreRange(sectionIdx, computeContext);
    for (uint32_t i = coreRange.first; i <= coreRange.second; ++i) {
        faNoFdConfig.coreNum = i;
        faWithFdConfig.coreNum = i;

        ScheduleFa(faNoFdConfig, computeContext, tmpResult);
        if (tmpResult.maxCost < bestResultNoFd.maxCost) {
            bestResultNoFd = tmpResult;
            faNoFdConfig.coreLimit = bestResultNoFd.usedCoreNum;
            faNoFdConfig.costLimit = bestResultNoFd.maxCost;
        }
        tmpResult.Clear();

        if (m_param.fdOn) {
            ScheduleFa(faWithFdConfig, computeContext, tmpResult);
            if (tmpResult.maxCost < bestResultWithFd.maxCost) {
                bestResultWithFd = tmpResult;
                faWithFdConfig.coreLimit = bestResultWithFd.usedCoreNum;
                faWithFdConfig.costLimit = bestResultWithFd.maxCost;
            }
            tmpResult.Clear();
        }
    }
    ScheduleFd(deviceInfo.aivCoreMaxNum, bestResultWithFd);
    return (CheckChooseWithFd(computeContext.gridInfo.sectionNum, bestResultNoFd, bestResultWithFd)) ?
               bestResultWithFd :
               bestResultNoFd;
}

inline void SectionStreamKImpl::ScheduleFa(const FaConfig &faConfig, const ComputeContext &computeContext,
                                           SectionStreamKImplResult &result)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const CostInfo &costInfo = computeContext.costInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    if (faConfig.coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curSectionIdx = faConfig.sectionIdx;
    assignContext.curBIdx = (faConfig.sectionIdx == 0) ?
                                0U :
                                FloorDiv(gridInfo.sectionBnIdx[faConfig.sectionIdx - 1U], GetHeadNum(baseInfo));
    assignContext.curBNIdx = (faConfig.sectionIdx == 0) ? 0U : gridInfo.sectionBnIdx[faConfig.sectionIdx - 1U];
    assignContext.curMIdx = 0U;
    assignContext.unassignedCost = costInfo.sectionCost[faConfig.sectionIdx];
    assignContext.bNCost = costInfo.bNCostOfEachBatch[assignContext.curBIdx];
    assignContext.bNBlock = costInfo.bNBlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, computeContext, assignContext.batchCache);
    CalcMCache(assignContext.curMIdx, computeContext, assignContext.batchCache, assignContext.mCache);
    assignContext.curS2Idx = assignContext.mCache.s2Start;

    for (uint32_t i = 0; i < faConfig.coreNum; ++i) {
        if (result.maxCost > faConfig.costLimit && i > faConfig.coreLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
            assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / static_cast<int64_t>(faConfig.coreNum - i);

        if (!faConfig.FdOn) {
            assignContext.coreCache.costLimit = std::max(assignContext.coreCache.costLimit, assignContext.mCache.mCost);
        } else {
            int64_t curCost = assignContext.mCache.mNormalBlockCost;
            if (assignContext.curS2Idx == (assignContext.mCache.s2End - 1U)) {
                curCost = assignContext.mCache.mLastBlockCost;
            }
            if (assignContext.curS2Idx == assignContext.mCache.s2Start && assignContext.mCache.mBlock != 1) {
                curCost += m_param.v0Cost;
            }
            assignContext.coreCache.costLimit = std::max(assignContext.coreCache.costLimit, curCost);
        }

        AssignByBatch(computeContext, assignContext);
        AssignByRow(computeContext, assignContext);
        if (faConfig.FdOn) {
            AssignByBlock(assignContext);
        }

        result.bNEnd[i] = assignContext.curBNIdx;
        result.mEnd[i] = assignContext.curMIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;
        if (faConfig.FdOn && IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(computeContext, assignContext, result);
            assignContext.preFdDataNum += assignContext.curKvSplitPart;
            assignContext.curKvSplitPart = 1U;
        }

        if (assignContext.curS2Idx > assignContext.mCache.s2Start &&
            assignContext.curS2Idx <= assignContext.mCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

inline void SectionStreamKImpl::ScheduleFd(uint32_t aivNum, SectionStreamKImplResult &result)
{
    if (result.fdTaskNum == 0U) {
        return;
    }
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0U;
    for (uint32_t i = 0; i < result.fdTaskNum; i++) {
        totalFDLoad += result.s2SplitNum[i] * result.mSize[i];
    }
    // 计算每个核处理的load
    uint32_t emptyVectorNum = aivNum - result.fdTaskNum;
    uint64_t averageLoad = CeilDiv(totalFDLoad, static_cast<uint64_t>(aivNum));
    uint32_t curCoreIndex = 0U;

    // 1. 计算全局平均负载，向上取整避免核负载为0
    // 2. 计算当前归约任务所用核数，向下取整且至少为1
    // 3. 计算当前归约任务每个核的平均行数，向上取整，避免行数为0
    // 4. 重新计算正确的核数，避免因为向上平均行数导致有核为空
    for (uint32_t i = 0; i < result.fdTaskNum; ++i) {
        if (emptyVectorNum == 0U) {
            result.taskIdx[curCoreIndex] = i;
            result.mStart[curCoreIndex] = 0U;
            result.mLen[curCoreIndex] = result.mSize[i];
            curCoreIndex++;
            continue;
        }

        uint32_t curVecNum = std::max(
            1U, static_cast<uint32_t>(static_cast<uint64_t>(result.s2SplitNum[i] * result.mSize[i]) / averageLoad));
        uint32_t curAvgMSize = CeilDiv(result.mSize[i], static_cast<uint32_t>(curVecNum));
        curVecNum = CeilDiv(result.mSize[i], curAvgMSize);
        curVecNum = std::min(curVecNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核

        for (uint32_t vid = 0; vid < curVecNum; vid++) {
            result.taskIdx[curCoreIndex] = i;
            result.mStart[curCoreIndex] = vid * curAvgMSize;
            result.mLen[curCoreIndex] = (vid < curVecNum - 1U) ? curAvgMSize : (result.mSize[i] - vid * curAvgMSize);
            curCoreIndex++;
        }
        emptyVectorNum -= (curVecNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    result.usedVecNum = curCoreIndex;
}

inline bool SectionStreamKImpl::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                   const SectionStreamKImplResult &result)
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
    if (assignContext.curBNIdx == result.bNEnd[assignContext.curCoreIdx - 1U] &&
        assignContext.curMIdx == result.mEnd[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

inline void SectionStreamKImpl::RecordFDInfo(const ComputeContext &computeContext, const AssignContext &assignContext,
                                             SectionStreamKImplResult &result)
{
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bNEnd[assignContext.curCoreIdx - 1U] / GetHeadNum(baseInfo);
    uint32_t splitMIdx = result.mEnd[assignContext.curCoreIdx - 1U];
    uint32_t mSize = GetMSize(splitBIdx, baseInfo);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdMSize = (splitMIdx == NumToIndex(gridInfo.mBaseNum[splitBIdx])) ?
                              (mSize - splitMIdx * m_param.mBaseSize) :
                              m_param.mBaseSize;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.bNIdx[result.fdTaskNum] = result.bNEnd[assignContext.curCoreIdx - 1U];
    result.mIdx[result.fdTaskNum] = result.mEnd[assignContext.curCoreIdx - 1U];
    result.workspaceIdx[result.fdTaskNum] = assignContext.preFdDataNum;
    result.s2SplitNum[result.fdTaskNum] = assignContext.curKvSplitPart;
    result.mSize[result.fdTaskNum] = curFdMSize;
    result.fdTaskNum++;
}

inline bool SectionStreamKImpl::CheckChooseWithFd(uint32_t sectionNum, const SectionStreamKImplResult &noFd,
                                                  const SectionStreamKImplResult &withFd)
{
    if (!m_param.fdOn) {
        return false;
    }

    if (sectionNum > 1U) {
        return true;
    }

    const int64_t full_block_cost = m_param.costFunc(m_param.mBaseSize, m_param.s2BaseSize);
    if (noFd.maxCost <= m_param.fdLeastBlock * full_block_cost) {
        return false;
    }
    int64_t fdTolerance = m_param.fdTolerance * full_block_cost;
    return noFd.maxCost - fdTolerance > withFd.maxCost; // using minus in case overflow
}

inline void SectionStreamKImpl::AssignByBatch(const ComputeContext &computeContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;
    const CostInfo &costInfo = computeContext.costInfo;
    uint32_t headNum = GetHeadNum(baseInfo);

    while (assignContext.bNCost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             SafeFloorDiv(costInfo.bNLastBlockCostOfEachBatch[assignContext.curBIdx],
                                          m_param.faToleranceRatio, int64_t{0}),
                             assignContext.coreCache.cost + assignContext.bNCost)) {
        assignContext.coreCache.cost += assignContext.bNCost;
        assignContext.coreCache.block += assignContext.bNBlock;
        assignContext.curBNIdx++;

        // to the end
        if (assignContext.curBNIdx == gridInfo.sectionBnIdx[assignContext.curSectionIdx]) {
            assignContext.curMIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBNIdx / headNum != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBNIdx / headNum;
            CalcBatchCache(assignContext.curBIdx, computeContext, assignContext.batchCache);
        }

        assignContext.bNCost = costInfo.bNCostOfEachBatch[assignContext.curBIdx];
        assignContext.bNBlock = costInfo.bNBlockOfEachBatch[assignContext.curBIdx];
        assignContext.curMIdx = 0U;
        CalcMCache(assignContext.curMIdx, computeContext, assignContext.batchCache, assignContext.mCache);
        assignContext.curS2Idx = assignContext.mCache.s2Start;
    }
}

inline void SectionStreamKImpl::AssignByRow(const ComputeContext &computeContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             SafeFloorDiv(assignContext.mCache.mLastBlockCost, m_param.faToleranceRatio, int64_t{0}),
                             assignContext.coreCache.cost + assignContext.mCache.mCost)) {
        assignContext.coreCache.cost += assignContext.mCache.mCost;
        assignContext.coreCache.block += assignContext.mCache.mBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bNCost =
            assignContext.bNCost > assignContext.mCache.mCost ? assignContext.bNCost - assignContext.mCache.mCost : 0;
        assignContext.bNBlock = assignContext.bNBlock > assignContext.mCache.mBlock ?
                                    assignContext.bNBlock - assignContext.mCache.mBlock :
                                    0U;
        // 计算新一行的信息
        do {
            assignContext.curMIdx++;
            CalcMCache(assignContext.curMIdx, computeContext, assignContext.batchCache, assignContext.mCache);
        } while (assignContext.mCache.mBlock == 0);
        assignContext.curS2Idx = assignContext.mCache.s2Start;
    }
}

inline void SectionStreamKImpl::AssignByBlock(AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.mCache.mNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.mCache.s2End - 1U)) {
        curCost = assignContext.mCache.mLastBlockCost;
    }

    int64_t realCost = curCost;
    if (assignContext.curS2Idx == assignContext.mCache.s2Start && assignContext.mCache.mBlock != 1U) {
        realCost += m_param.v0Cost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             SafeFloorDiv(realCost, m_param.faToleranceRatio, int64_t{0}),
                             assignContext.coreCache.cost + realCost)) {
        assignContext.coreCache.cost += realCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bNCost = assignContext.bNCost - realCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.mCache.mCost = assignContext.mCache.mCost - realCost;
        assignContext.bNBlock--;
        assignContext.mCache.mBlock--;

        realCost = curCost;
        if (assignContext.curS2Idx == 0U && assignContext.mCache.mBlock != 1U) {
            realCost += m_param.v0Cost;
        }
    }
}

} // namespace load_balance
#endif
