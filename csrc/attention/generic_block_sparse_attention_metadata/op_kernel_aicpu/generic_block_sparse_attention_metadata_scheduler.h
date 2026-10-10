/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_SCHEDULER_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_SCHEDULER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../generic_block_sparse_attention_metadata_tiling.h"

namespace aicpu {
namespace generic_block_sparse_attention_metadata {

enum class BSAScheduleStatus : uint32_t {
    BSA_SUCCESS = 0U,
    BSA_INVALID_PARAM = 1U,
    BSA_OVERFLOW = 2U,
    BSA_METADATA_TOO_SMALL = 3U,
    BSA_SCHEDULE_TOO_LARGE = 4U,
};

struct ScheduleInput {
    int64_t batchSize = 0;
    int64_t numQHeads = 0;
    int64_t numKvHeads = 0;
    int64_t maxQSeqLen = 0;
    int64_t headDim = 0;
    int64_t blockShapeX = 1;
    int64_t blockShapeY = 0;
    int64_t blockIndexStride = 0;
    int64_t qBlockStorageNum = 0;
    int64_t isPackedGQA = 1;
    int64_t residualBlockMode = 0;
    int64_t isConsistentTopk = 0;
    int64_t aicCoreNum = 0;
    std::vector<int64_t> qSeqLens; // seqUsed/cuQLens/maxQLen
    // qUnit-major, sparseHead-minor logical order.
    std::vector<int64_t> validBlockNums;
};

struct DecodeSchedule {
    int64_t baseTaskStart = 0;
    int64_t baseTaskEnd = 0;
    int64_t firstBlockStart = 0;
    int64_t lastBlockEnd = 0;
};

struct CombineSchedule {
    int64_t baseTask = 0;
    int64_t firstCore = 0;
    int64_t partialStart = 0;
    int64_t partialCount = 0;
};

struct ScheduleResult {
    int64_t saUsedCoreNum = 0;
    int64_t saTotalTaskNum = 0;
    int64_t totalQTokenNum = 0;
    int64_t sparseHeadNum = 0;
    int64_t groupSize = 0;
    int64_t fdActiveCoreNum = 0;
    int64_t decodePerCoreTaskNum = 0;
    int64_t combineTaskNum = 0;
    int64_t fdScheduleFlags = 0;
    int64_t fdTotalFlatTaskNum = 0;
    int64_t fdPartialTaskNum = 0;
    int64_t configSignature = 0;
    std::vector<int64_t> qLogicalStarts;
    std::vector<int64_t> blockPrefix;
    std::array<DecodeSchedule, optiling::generic_block_sparse_attention_metadata::MAX_DECODE_CORE_NUM>
        decodeSchedules{};
    std::array<CombineSchedule, optiling::generic_block_sparse_attention_metadata::MAX_COMBINE_TASK_NUM>
        combineSchedules{};
};

struct TaskInfo {
    int64_t qUnit = 0;
    int64_t sparseHeadIdx = 0;
    int64_t batchIdx = 0;
    int64_t qTokenInBatch = 0;
    int64_t kvHeadIdx = 0;
    int64_t qHeadStart = 0;
    int64_t qHeadCount = 0;
};

inline BSAScheduleStatus BuildSchedule(const ScheduleInput &input, ScheduleResult &result);
inline BSAScheduleStatus DecodeTask(const ScheduleInput &input, const ScheduleResult &result, int64_t taskIdx,
                                    TaskInfo &taskInfo);
inline BSAScheduleStatus EncodeMetadata(const ScheduleResult &result,
                                        optiling::generic_block_sparse_attention_metadata::MetadataType *metadata,
                                        size_t metadataElementNum);

namespace {

using optiling::generic_block_sparse_attention_metadata::MetadataType;

constexpr uint64_t FD_COST_M16 = 125U;
constexpr uint64_t FD_COST_N = 740U;
constexpr uint64_t FD_COST_M16_N = 35U;
constexpr uint64_t FD_LAUNCH_COST = 278U;

inline uint32_t HashConfigValue(uint32_t hash, uint32_t value)
{
    constexpr uint32_t FNV_PRIME = 16777619U;
    for (uint32_t shift = 0; shift < 32U; shift += 8U) {
        hash ^= (value >> shift) & 0xFFU;
        hash *= FNV_PRIME;
    }
    return hash;
}

inline uint32_t CalculateConfigSignature(const ScheduleInput &input)
{
    uint32_t hash = 2166136261U;
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.batchSize));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.numQHeads));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.numKvHeads));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.headDim));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.blockShapeX));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.blockShapeY));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.blockIndexStride));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.qBlockStorageNum));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.isPackedGQA));
    // residual_block_mode / is_consistent_topk do not change the schedule itself, but they must
    // participate in the signature so the main op rejects metadata generated with other settings.
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.residualBlockMode));
    hash = HashConfigValue(hash, static_cast<uint32_t>(input.isConsistentTopk));
    return hash;
}

inline bool AddOverflow(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return true;
    }
    result = lhs + rhs;
    return false;
}

inline bool MulOverflow(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

inline bool AddOverflow(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return true;
    }
    result = lhs + rhs;
    return false;
}

inline bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

inline bool FitsMetadataType(int64_t value)
{
    return value >= static_cast<int64_t>(std::numeric_limits<MetadataType>::min()) &&
           value <= static_cast<int64_t>(std::numeric_limits<MetadataType>::max());
}

inline int64_t CeilDiv(int64_t lhs, int64_t rhs)
{
    return lhs == 0 ? 0 : (lhs - 1) / rhs + 1;
}

inline uint32_t CalcFdBestCore(uint64_t totalCost, uint64_t launchCost, uint32_t maxCore)
{
    if (totalCost == 0U || launchCost == 0U || maxCore == 0U) {
        return 0U;
    }
    const double ratio = static_cast<double>(totalCost) / static_cast<double>(launchCost);
    const double root = (std::sqrt(1.0 + 4.0 * ratio) - 1.0) / 2.0;
    const uint32_t bestCore = static_cast<uint32_t>(std::ceil(root));
    return std::max(1U, std::min(bestCore, maxCore));
}

inline BSAScheduleStatus AccumulateFdCost(int64_t m16, int64_t validBlockNum, uint64_t &totalCost,
                                          uint64_t &totalValidBlockNum)
{
    uint64_t mCost = 0U;
    uint64_t nCost = 0U;
    uint64_t mn = 0U;
    uint64_t mnCost = 0U;
    uint64_t taskCost = 0U;
    if (MulOverflow(FD_COST_M16, static_cast<uint64_t>(m16), mCost) ||
        MulOverflow(FD_COST_N, static_cast<uint64_t>(validBlockNum), nCost) ||
        MulOverflow(static_cast<uint64_t>(m16), static_cast<uint64_t>(validBlockNum), mn) ||
        MulOverflow(FD_COST_M16_N, mn, mnCost) || AddOverflow(mCost, nCost, taskCost) ||
        AddOverflow(taskCost, mnCost, taskCost) || AddOverflow(totalCost, taskCost, totalCost) ||
        AddOverflow(totalValidBlockNum, static_cast<uint64_t>(validBlockNum), totalValidBlockNum)) {
        return BSAScheduleStatus::BSA_OVERFLOW;
    }
    return BSAScheduleStatus::BSA_SUCCESS;
}

inline bool IsScheduleInputValid(const ScheduleInput &input)
{
    using namespace optiling::generic_block_sparse_attention_metadata;
    return input.batchSize >= 0 && input.numQHeads > 0 && input.numKvHeads > 0 && input.headDim > 0 &&
           input.blockShapeX == 1 && input.blockShapeY > 0 && input.blockIndexStride > 0 &&
           input.qBlockStorageNum > 0 && (input.isPackedGQA == 0 || input.isPackedGQA == 1) &&
           (input.residualBlockMode == 0 || input.residualBlockMode == 1) && input.aicCoreNum > 0 &&
           input.aicCoreNum <= MAX_AIC_CORE_NUM && input.qSeqLens.size() == static_cast<size_t>(input.batchSize);
}

inline BSAScheduleStatus BuildSaTaskInfo(const ScheduleInput &input, ScheduleResult &result)
{
    result.groupSize = input.numQHeads / input.numKvHeads;
    result.sparseHeadNum = input.isPackedGQA == 1 ? input.numKvHeads : input.numQHeads;
    result.qLogicalStarts.resize(static_cast<size_t>(input.batchSize) + 1U, 0);

    int64_t totalQTokenNum = 0;
    for (int64_t batchIdx = 0; batchIdx < input.batchSize; ++batchIdx) {
        const int64_t qSeqLen = input.qSeqLens[static_cast<size_t>(batchIdx)];
        if (qSeqLen < 0 || (input.maxQSeqLen > 0 && qSeqLen > input.maxQSeqLen)) {
            return BSAScheduleStatus::BSA_INVALID_PARAM;
        }
        result.qLogicalStarts[static_cast<size_t>(batchIdx)] = totalQTokenNum; // the Q tokens start position in  batch.
        if (AddOverflow(totalQTokenNum, qSeqLen, totalQTokenNum) || !FitsMetadataType(totalQTokenNum)) {
            return BSAScheduleStatus::BSA_OVERFLOW;
        }
    }
    result.qLogicalStarts[static_cast<size_t>(input.batchSize)] = totalQTokenNum;
    result.totalQTokenNum = totalQTokenNum;
    if (MulOverflow(totalQTokenNum, result.sparseHeadNum, result.saTotalTaskNum) ||
        !FitsMetadataType(result.saTotalTaskNum)) {
        return BSAScheduleStatus::BSA_OVERFLOW;
    }
    if (input.validBlockNums.size() != static_cast<size_t>(result.saTotalTaskNum)) {
        return BSAScheduleStatus::BSA_INVALID_PARAM;
    }
    result.saUsedCoreNum = std::min(result.saTotalTaskNum, input.aicCoreNum);
    return BSAScheduleStatus::BSA_SUCCESS;
}

inline BSAScheduleStatus BuildBlockPrefix(const ScheduleInput &input, ScheduleResult &result, uint64_t &totalCost,
                                          uint64_t &totalValidBlockNum)
{
    result.blockPrefix.resize(static_cast<size_t>(result.saTotalTaskNum) + 1U, 0);
    const int64_t qHeadCount = input.isPackedGQA == 1 ? result.groupSize : 1;
    const int64_t m16 = CeilDiv(qHeadCount, 16);
    for (int64_t task = 0; task < result.saTotalTaskNum; ++task) {
        const int64_t validBlockNum = input.validBlockNums[static_cast<size_t>(task)];
        const int64_t maxValidBlockNum = input.blockIndexStride + (input.residualBlockMode == 1 ? 1 : 0);
        if (validBlockNum < 0 || validBlockNum > maxValidBlockNum) {
            return BSAScheduleStatus::BSA_INVALID_PARAM;
        }
        int64_t prefix = 0;
        if (AddOverflow(result.blockPrefix[static_cast<size_t>(task)], validBlockNum, prefix) ||
            !FitsMetadataType(prefix)) {
            return BSAScheduleStatus::BSA_OVERFLOW;
        }
        result.blockPrefix[static_cast<size_t>(task) + 1U] = prefix;
        if (validBlockNum > 0) {
            const BSAScheduleStatus status = AccumulateFdCost(m16, validBlockNum, totalCost, totalValidBlockNum);
            if (status != BSAScheduleStatus::BSA_SUCCESS) {
                return status;
            }
        }
    }
    return BSAScheduleStatus::BSA_SUCCESS;
}

inline bool ShouldBuildDecodeSchedule(const ScheduleInput &input, const ScheduleResult &result, uint64_t totalCost,
                                      uint64_t totalValidBlockNum)
{
    const int64_t totalSplitTaskNum = result.blockPrefix.back();
    // Decode scheduling partitions sparse-list block ranges. The kernel resolves each range with the runtime
    // blockShapeY, so the schedule is independent of the sparse block's token width.
    const bool fdShapeSupported =
        input.headDim > 0 && input.headDim <= 512 && input.blockIndexStride >= 12 &&
        result.groupSize > 0 && result.groupSize <= 128 &&
        (input.maxQSeqLen <= 0 || input.maxQSeqLen >= result.totalQTokenNum) && result.saTotalTaskNum > 0 &&
        static_cast<uint64_t>(result.saTotalTaskNum) * 10U < static_cast<uint64_t>(input.aicCoreNum) * 3U;
    return fdShapeSupported && totalSplitTaskNum > result.saUsedCoreNum && totalCost > 0U && totalValidBlockNum > 0U;
}

inline BSAScheduleStatus BuildDecodeCoreSchedules(ScheduleResult &result, int64_t totalSplitTaskNum)
{
    for (int64_t coreIdx = 0; coreIdx < result.fdActiveCoreNum; ++coreIdx) {
        DecodeSchedule &schedule = result.decodeSchedules[static_cast<size_t>(coreIdx)];
        const int64_t flatStart = std::min(coreIdx * result.decodePerCoreTaskNum, totalSplitTaskNum);
        const int64_t flatEnd = std::min(flatStart + result.decodePerCoreTaskNum, totalSplitTaskNum);
        if (flatStart == flatEnd) {
            continue;
        }
        const auto firstIt = std::upper_bound(result.blockPrefix.begin(), result.blockPrefix.end(), flatStart);
        const auto lastIt = std::upper_bound(result.blockPrefix.begin(), result.blockPrefix.end(), flatEnd - 1);
        if (firstIt == result.blockPrefix.begin() || lastIt == result.blockPrefix.begin()) {
            return BSAScheduleStatus::BSA_INVALID_PARAM;
        }
        const int64_t firstTask = static_cast<int64_t>(firstIt - result.blockPrefix.begin()) - 1;
        const int64_t lastTask = static_cast<int64_t>(lastIt - result.blockPrefix.begin()) - 1;
        schedule.baseTaskStart = firstTask;
        schedule.baseTaskEnd = lastTask + 1;
        schedule.firstBlockStart = flatStart - result.blockPrefix[static_cast<size_t>(firstTask)];
        schedule.lastBlockEnd = flatEnd - result.blockPrefix[static_cast<size_t>(lastTask)];
    }
    return BSAScheduleStatus::BSA_SUCCESS;
}

inline BSAScheduleStatus BuildCombineSchedules(ScheduleResult &result)
{
    using namespace optiling::generic_block_sparse_attention_metadata;
    int64_t partialTaskNum = 0;
    for (int64_t task = 0; task < result.saTotalTaskNum; ++task) {
        const int64_t taskStart = result.blockPrefix[static_cast<size_t>(task)];
        const int64_t taskEnd = result.blockPrefix[static_cast<size_t>(task) + 1U];
        if (taskStart == taskEnd) {
            continue;
        }
        const int64_t firstCore = taskStart / result.decodePerCoreTaskNum;
        const int64_t lastCore = (taskEnd - 1) / result.decodePerCoreTaskNum;
        const int64_t partialCount = lastCore - firstCore + 1;
        if (partialCount <= 1) {
            continue;
        }
        if (partialCount > result.fdActiveCoreNum ||
            partialCount > static_cast<int64_t>(MAX_FD_PARTIAL_PER_BASE_TASK) || firstCore < 0 ||
            firstCore + partialCount > result.fdActiveCoreNum) {
            return BSAScheduleStatus::BSA_SCHEDULE_TOO_LARGE;
        }
        if (result.combineTaskNum >= MAX_COMBINE_TASK_NUM) {
            return BSAScheduleStatus::BSA_SCHEDULE_TOO_LARGE;
        }
        CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(result.combineTaskNum++)];
        combine.baseTask = task;
        combine.firstCore = firstCore;
        combine.partialStart = partialTaskNum;
        combine.partialCount = partialCount;
        if (AddOverflow(partialTaskNum, partialCount, partialTaskNum) || !FitsMetadataType(partialTaskNum)) {
            return BSAScheduleStatus::BSA_OVERFLOW;
        }
    }
    result.fdPartialTaskNum = partialTaskNum;
    return BSAScheduleStatus::BSA_SUCCESS;
}

} // namespace

inline BSAScheduleStatus BuildSchedule(const ScheduleInput &input, ScheduleResult &result)
{
    using namespace optiling::generic_block_sparse_attention_metadata;
    result = {};
    if (!IsScheduleInputValid(input)) {
        return BSAScheduleStatus::BSA_INVALID_PARAM;
    }
    result.configSignature = static_cast<int64_t>(CalculateConfigSignature(input));
    result.fdScheduleFlags = static_cast<int64_t>(FD_ACTUAL_BLOCK_PREFIX);
    BSAScheduleStatus status = BuildSaTaskInfo(input, result);
    if (status != BSAScheduleStatus::BSA_SUCCESS) {
        return status;
    }
    uint64_t totalCost = 0U;
    uint64_t totalValidBlockNum = 0U;
    status = BuildBlockPrefix(input, result, totalCost, totalValidBlockNum);
    if (status != BSAScheduleStatus::BSA_SUCCESS) {
        return status;
    }
    const int64_t totalSplitTaskNum = result.blockPrefix.back();
    result.fdTotalFlatTaskNum = totalSplitTaskNum;
    if (!ShouldBuildDecodeSchedule(input, result, totalCost, totalValidBlockNum)) {
        return BSAScheduleStatus::BSA_SUCCESS;
    }
    const uint64_t maxCoreLimit = std::min<uint64_t>(
        std::min<uint64_t>(static_cast<uint64_t>(input.aicCoreNum), MAX_FD_ACTIVE_CORE_NUM), totalValidBlockNum);
    const int64_t bestCoreNum = CalcFdBestCore(totalCost, FD_LAUNCH_COST, static_cast<uint32_t>(maxCoreLimit));
    if (bestCoreNum == 0) {
        return BSAScheduleStatus::BSA_SUCCESS;
    }
    result.decodePerCoreTaskNum = CeilDiv(totalSplitTaskNum, bestCoreNum);
    result.fdActiveCoreNum = CeilDiv(totalSplitTaskNum, result.decodePerCoreTaskNum);
    status = BuildDecodeCoreSchedules(result, totalSplitTaskNum);
    if (status != BSAScheduleStatus::BSA_SUCCESS) {
        return status;
    }
    status = BuildCombineSchedules(result);
    if (status == BSAScheduleStatus::BSA_SUCCESS) {
        // for codecheck pass.
        result.fdScheduleFlags = static_cast<int64_t>(static_cast<uint32_t>(result.fdScheduleFlags) |
                                                      static_cast<uint32_t>(FD_SCHEDULE_ENABLED));
    }
    return status;
}

inline BSAScheduleStatus DecodeTask(const ScheduleInput &input, const ScheduleResult &result, int64_t taskIdx,
                                    TaskInfo &taskInfo)
{
    if (taskIdx < 0 || taskIdx >= result.saTotalTaskNum || result.sparseHeadNum <= 0 ||
        result.qLogicalStarts.size() != static_cast<size_t>(input.batchSize) + 1U) {
        return BSAScheduleStatus::BSA_INVALID_PARAM;
    }
    taskInfo = {};
    taskInfo.qUnit = taskIdx / result.sparseHeadNum;
    taskInfo.sparseHeadIdx = taskIdx % result.sparseHeadNum;
    if (input.isPackedGQA == 1) {
        taskInfo.kvHeadIdx = taskInfo.sparseHeadIdx;
        taskInfo.qHeadStart = taskInfo.sparseHeadIdx * result.groupSize;
        taskInfo.qHeadCount = result.groupSize;
    } else {
        taskInfo.qHeadStart = taskInfo.sparseHeadIdx;
        taskInfo.qHeadCount = 1;
        if (result.groupSize <= 0) {
            return BSAScheduleStatus::BSA_INVALID_PARAM;
        }
        taskInfo.kvHeadIdx = taskInfo.sparseHeadIdx / result.groupSize;
    }
    const auto batchIt = std::upper_bound(result.qLogicalStarts.begin(), result.qLogicalStarts.end(), taskInfo.qUnit);
    if (batchIt == result.qLogicalStarts.begin()) {
        return BSAScheduleStatus::BSA_INVALID_PARAM;
    }
    taskInfo.batchIdx = static_cast<int64_t>(batchIt - result.qLogicalStarts.begin()) - 1;
    if (taskInfo.batchIdx >= input.batchSize) {
        return BSAScheduleStatus::BSA_INVALID_PARAM;
    }
    taskInfo.qTokenInBatch = taskInfo.qUnit - result.qLogicalStarts[static_cast<size_t>(taskInfo.batchIdx)];
    return BSAScheduleStatus::BSA_SUCCESS;
}

inline BSAScheduleStatus EncodeMetadata(const ScheduleResult &result, MetadataType *metadata, size_t metadataElementNum)
{
    using namespace optiling::generic_block_sparse_attention_metadata;
    if (metadata == nullptr || metadataElementNum < METADATA_TOTAL_SIZE || result.fdActiveCoreNum < 0 ||
        result.fdActiveCoreNum > MAX_DECODE_CORE_NUM || result.combineTaskNum < 0 ||
        result.combineTaskNum > MAX_COMBINE_TASK_NUM || result.fdPartialTaskNum < 0 || result.fdTotalFlatTaskNum < 0) {
        return BSAScheduleStatus::BSA_METADATA_TOO_SMALL;
    }
    std::fill(metadata, metadata + METADATA_TOTAL_SIZE, 0);
    metadata[MAGIC_INDEX] = METADATA_MAGIC;
    metadata[VERSION_INDEX] = METADATA_VERSION;
    metadata[METADATA_USED_SIZE_INDEX] = static_cast<MetadataType>(METADATA_USED_SIZE);
    metadata[SA_USED_CORE_NUM_INDEX] = static_cast<MetadataType>(result.saUsedCoreNum);
    metadata[SA_TOTAL_TASK_NUM_INDEX] = static_cast<MetadataType>(result.saTotalTaskNum);
    metadata[FD_ACTIVE_CORE_NUM_INDEX] = static_cast<MetadataType>(result.fdActiveCoreNum);
    metadata[DECODE_PER_CORE_TASK_NUM_INDEX] = static_cast<MetadataType>(result.decodePerCoreTaskNum);
    metadata[COMBINE_TASK_NUM_INDEX] = static_cast<MetadataType>(result.combineTaskNum);
    metadata[FD_SCHEDULE_FLAGS_INDEX] = static_cast<MetadataType>(result.fdScheduleFlags);
    metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX] = static_cast<MetadataType>(result.fdTotalFlatTaskNum);
    metadata[FD_PARTIAL_TASK_NUM_INDEX] = static_cast<MetadataType>(result.fdPartialTaskNum);
    metadata[CONFIG_SIGNATURE_INDEX] = static_cast<MetadataType>(result.configSignature);

    for (uint32_t coreIdx = 0; coreIdx < MAX_DECODE_CORE_NUM; ++coreIdx) {
        const DecodeSchedule &schedule = result.decodeSchedules[coreIdx];
        metadata[GetDecodeScheduleIndex(coreIdx, DECODE_BASE_TASK_START_INDEX)] =
            static_cast<MetadataType>(schedule.baseTaskStart);
        metadata[GetDecodeScheduleIndex(coreIdx, DECODE_BASE_TASK_END_INDEX)] =
            static_cast<MetadataType>(schedule.baseTaskEnd);
        metadata[GetDecodeScheduleIndex(coreIdx, DECODE_FIRST_BLOCK_START_INDEX)] =
            static_cast<MetadataType>(schedule.firstBlockStart);
        metadata[GetDecodeScheduleIndex(coreIdx, DECODE_LAST_BLOCK_END_INDEX)] =
            static_cast<MetadataType>(schedule.lastBlockEnd);
    }
    for (uint32_t combineIdx = 0; combineIdx < MAX_COMBINE_TASK_NUM; ++combineIdx) {
        const CombineSchedule &schedule = result.combineSchedules[combineIdx];
        metadata[GetCombineScheduleIndex(combineIdx, COMBINE_BASE_TASK_INDEX)] =
            static_cast<MetadataType>(schedule.baseTask);
        metadata[GetCombineScheduleIndex(combineIdx, COMBINE_FIRST_CORE_INDEX)] =
            static_cast<MetadataType>(schedule.firstCore);
        metadata[GetCombineScheduleIndex(combineIdx, COMBINE_PARTIAL_START_INDEX)] =
            static_cast<MetadataType>(schedule.partialStart);
        metadata[GetCombineScheduleIndex(combineIdx, COMBINE_PARTIAL_COUNT_INDEX)] =
            static_cast<MetadataType>(schedule.partialCount);
    }
    return BSAScheduleStatus::BSA_SUCCESS;
}

} // namespace generic_block_sparse_attention_metadata
} // namespace aicpu

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_SCHEDULER_H
