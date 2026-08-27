/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "../../op_kernel_aicpu/sparse_attention_score_metadata_q_seq_utils.h"
#include "../../op_kernel_aicpu/sparse_attention_score_metadata_scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace aicpu::sparse_attention_score_metadata;
using namespace optiling::generic_block_sparse_attention_metadata;

#define CHECK_TRUE(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " << #expr << std::endl;                       \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_EQ(lhs, rhs)                                                                                             \
    do {                                                                                                               \
        const auto lhsValue = (lhs);                                                                                   \
        const auto rhsValue = (rhs);                                                                                   \
        if (lhsValue != rhsValue) {                                                                                    \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " << #lhs << " != " << #rhs << std::endl;        \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

ScheduleInput MakeInput()
{
    ScheduleInput input;
    input.batchSize = 2;
    input.numQHeads = 8;
    input.numKvHeads = 2;
    input.maxQSeqLen = 3;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 6;
    input.isPackedGQA = 1;
    input.aicCoreNum = 8;
    input.qSeqLens = {2, 1};
    input.validBlockNums = {0, 0, 0, 0, 0, 0};
    return input;
}

uint32_t GetHostFdPartialCapacity(uint32_t aicCoreNum)
{
    const uint32_t maxNonEmptyBaseTaskNum = aicCoreNum == 0U ? 0U :
        std::min<uint32_t>(MAX_COMBINE_TASK_NUM, (aicCoreNum * 3U - 1U) / 10U);
    const uint32_t maxActiveCoreNum = std::min<uint32_t>(aicCoreNum, MAX_FD_ACTIVE_CORE_NUM);
    return maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ?
        0U : maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
}

void TestPrefillSchedule()
{
    ScheduleInput input = MakeInput();
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.totalQTokenNum, 3);
    CHECK_EQ(result.saTotalTaskNum, 6);
    CHECK_EQ(result.saUsedCoreNum, 6);
    CHECK_EQ(result.sparseHeadNum, 2);
    CHECK_EQ(result.groupSize, 4);
    CHECK_EQ(result.fdActiveCoreNum, 0);

    TaskInfo task;
    CHECK_EQ(DecodeTask(input, result, 4, task), ScheduleStatus::SUCCESS);
    CHECK_EQ(task.qUnit, 2);
    CHECK_EQ(task.batchIdx, 1);
    CHECK_EQ(task.qTokenInBatch, 0);
    CHECK_EQ(task.kvHeadIdx, 0);
    CHECK_EQ(task.qHeadStart, 0);
    CHECK_EQ(task.qHeadCount, 4);
}

void TestActualBlockPrefixDecode()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.blockPrefix[0], 0);
    CHECK_EQ(result.blockPrefix[1], 12);
    CHECK_EQ(result.blockPrefix[2], 15);
    CHECK_EQ(result.fdActiveCoreNum, 5);
    CHECK_EQ(result.decodePerCoreTaskNum, 3);

    CHECK_EQ(result.decodeSchedules[0].baseTaskStart, 0);
    CHECK_EQ(result.decodeSchedules[0].baseTaskEnd, 1);
    CHECK_EQ(result.decodeSchedules[0].firstBlockStart, 0);
    CHECK_EQ(result.decodeSchedules[0].lastBlockEnd, 3);
    CHECK_EQ(result.decodeSchedules[3].firstBlockStart, 9);
    CHECK_EQ(result.decodeSchedules[3].lastBlockEnd, 12);
    CHECK_EQ(result.decodeSchedules[4].baseTaskStart, 1);
    CHECK_EQ(result.decodeSchedules[4].baseTaskEnd, 2);
    CHECK_EQ(result.decodeSchedules[4].firstBlockStart, 0);
    CHECK_EQ(result.decodeSchedules[4].lastBlockEnd, 3);
    CHECK_EQ(result.decodeSchedules[5].baseTaskEnd, 0);

    CHECK_EQ(result.combineTaskNum, 1);
    CHECK_EQ(result.combineSchedules[0].baseTask, 0);
    CHECK_EQ(result.combineSchedules[0].firstCore, 0);
    CHECK_EQ(result.combineSchedules[0].partialStart, 0);
    CHECK_EQ(result.combineSchedules[0].partialCount, 4);
}

void TestDecodeShapeConstraints()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};

    ScheduleResult result;
    input.blockShapeY = 64;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.fdActiveCoreNum, 0);

    input.blockShapeY = 128;
    input.headDim = 64;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.fdActiveCoreNum, 0);
}

void TestZeroCountsAndRepeatedPrefix()
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 3;
    input.numQHeads = 1;
    input.numKvHeads = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {3};
    input.validBlockNums = {0, 12, 3};
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.blockPrefix[0], 0);
    CHECK_EQ(result.blockPrefix[1], 0);
    CHECK_EQ(result.blockPrefix[2], 12);
    CHECK_EQ(result.blockPrefix[3], 15);
    CHECK_EQ(result.decodeSchedules[0].baseTaskStart, 1);
}

void TestSparseBlockCapacityGate()
{
    ScheduleInput lhs;
    lhs.batchSize = 1;
    lhs.numQHeads = 4;
    lhs.numKvHeads = 1;
    lhs.maxQSeqLen = 2;
    lhs.headDim = 128;
    lhs.blockShapeX = 1;
    lhs.blockShapeY = 128;
    lhs.blockIndexStride = 12;
    lhs.qBlockStorageNum = 2;
    lhs.isPackedGQA = 1;
    lhs.aicCoreNum = 32;
    lhs.qSeqLens = {2};
    lhs.validBlockNums = {12, 3};
    ScheduleInput rhs = lhs;
    rhs.blockIndexStride = 17;
    ScheduleResult lhsResult;
    ScheduleResult rhsResult;
    CHECK_EQ(BuildSchedule(lhs, lhsResult), ScheduleStatus::SUCCESS);
    CHECK_EQ(BuildSchedule(rhs, rhsResult), ScheduleStatus::SUCCESS);
    CHECK_TRUE(lhsResult.fdActiveCoreNum > 0);
    CHECK_EQ(rhsResult.fdActiveCoreNum, 0);
}

void TestActiveCoreCapacityAcrossBaseTasks()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 8;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 8;
    input.isPackedGQA = 1;
    input.aicCoreNum = 28;
    input.qSeqLens = {8};
    input.validBlockNums.assign(8, 16);

    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_TRUE(result.fdActiveCoreNum > 16);
    CHECK_TRUE(result.fdActiveCoreNum <= input.aicCoreNum);
    CHECK_EQ(result.combineTaskNum, 8);
    int64_t partialTaskNum = 0;
    for (int64_t task = 0; task < result.combineTaskNum; ++task) {
        const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(task)];
        CHECK_TRUE(combine.partialCount > 1);
        CHECK_TRUE(combine.partialCount <= static_cast<int64_t>(MAX_FD_PARTIAL_PER_BASE_TASK));
        CHECK_TRUE(combine.firstCore + combine.partialCount <= result.fdActiveCoreNum);
        partialTaskNum += combine.partialCount;
    }
    CHECK_EQ(result.fdPartialTaskNum, partialTaskNum);
}

void TestPartialTaskNumCanExceedActiveCoreNum()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 7;
    input.qSeqLens = {2};
    input.validBlockNums = {3, 3};

    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.fdActiveCoreNum, 3);
    CHECK_EQ(result.combineTaskNum, 2);
    CHECK_EQ(result.fdPartialTaskNum, 4);
    CHECK_TRUE(result.fdPartialTaskNum > result.fdActiveCoreNum);
    CHECK_EQ(result.fdPartialTaskNum,
             static_cast<int64_t>(input.validBlockNums.size()) + result.fdActiveCoreNum - 1);
    CHECK_TRUE(result.fdPartialTaskNum <= GetHostFdPartialCapacity(static_cast<uint32_t>(input.aicCoreNum)));
}

void TestPartialWorkspaceUpperBound()
{
    CHECK_EQ(GetHostFdPartialCapacity(0U), 0U);
    CHECK_EQ(GetHostFdPartialCapacity(28U), 35U);
    for (uint32_t aicCoreNum = 1U; aicCoreNum <= MAX_FD_ACTIVE_CORE_NUM; ++aicCoreNum) {
        const uint32_t maxNonEmptyBaseTaskNum = (aicCoreNum * 3U - 1U) / 10U;
        const uint32_t hostCapacity = GetHostFdPartialCapacity(aicCoreNum);
        for (uint32_t baseTaskNum = 1U; baseTaskNum <= maxNonEmptyBaseTaskNum; ++baseTaskNum) {
            for (int64_t validBlockNum = 1; validBlockNum <= 16; ++validBlockNum) {
                ScheduleInput input;
                input.batchSize = 1;
                input.numQHeads = 4;
                input.numKvHeads = 1;
                input.maxQSeqLen = baseTaskNum;
                input.headDim = 128;
                input.blockShapeX = 1;
                input.blockShapeY = 128;
                input.blockIndexStride = 16;
                input.qBlockStorageNum = baseTaskNum;
                input.isPackedGQA = 1;
                input.aicCoreNum = aicCoreNum;
                input.qSeqLens = {baseTaskNum};
                input.validBlockNums.assign(baseTaskNum, validBlockNum);

                ScheduleResult result;
                CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
                if ((static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED) == 0U) {
                    continue;
                }
                const int64_t actualIntersectionUpperBound =
                    static_cast<int64_t>(baseTaskNum) + result.fdActiveCoreNum - 1;
                CHECK_TRUE(result.fdPartialTaskNum <= actualIntersectionUpperBound);
                CHECK_TRUE(result.fdPartialTaskNum <= hostCapacity);
                for (int64_t combineIdx = 0; combineIdx < result.combineTaskNum; ++combineIdx) {
                    const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(combineIdx)];
                    CHECK_TRUE(combine.partialStart >= 0);
                    CHECK_TRUE(combine.partialStart + combine.partialCount <= result.fdPartialTaskNum);
                }
            }
        }
    }
}

void TestUnpackedInternalTaskModel()
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 1;
    input.qSeqLens = {1};
    input.isPackedGQA = 0;
    input.validBlockNums.assign(8, 0);
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    CHECK_EQ(result.sparseHeadNum, 8);
    CHECK_EQ(result.saTotalTaskNum, 8);

    TaskInfo task;
    CHECK_EQ(DecodeTask(input, result, 5, task), ScheduleStatus::SUCCESS);
    CHECK_EQ(task.sparseHeadIdx, 5);
    CHECK_EQ(task.qHeadStart, 5);
    CHECK_EQ(task.qHeadCount, 1);
    CHECK_EQ(task.kvHeadIdx, 1);
}

void TestInvalidBlockCount()
{
    ScheduleInput input = MakeInput();
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::INVALID_PARAM);
}

void TestTndSeqUsedUsesStorageOffsets()
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    CHECK_TRUE(BuildTndQSeqLayout({0, 4, 8}, {2, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_EQ(actualQSeqLens[0], 2);
    CHECK_EQ(actualQSeqLens[1], 3);
    CHECK_EQ(qStorageBlockStarts[0], 0);
    CHECK_EQ(qStorageBlockStarts[1], 4);

    const int32_t counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> validBlockNums;
    CHECK_TRUE(GatherTndValidBlockNums(counts, 1, 8, 1, 8, actualQSeqLens, qStorageBlockStarts, validBlockNums));
    CHECK_EQ(validBlockNums.size(), 5U);
    CHECK_EQ(validBlockNums[0], 1);
    CHECK_EQ(validBlockNums[1], 2);
    CHECK_EQ(validBlockNums[2], 5);
    CHECK_EQ(validBlockNums[3], 6);
    CHECK_EQ(validBlockNums[4], 7);
}

void TestTndQSeqValidation()
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {5, 3}, true, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 3}, {}, false, 4, 1, 4, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({1, 4, 8}, {}, false, 4, 1, 8, actualQSeqLens, qStorageBlockStarts));
    CHECK_TRUE(!BuildTndQSeqLayout({0, 4, 8}, {}, false, 4, 1, 7, actualQSeqLens, qStorageBlockStarts));
}

void TestMetadataEncoding()
{
    ScheduleInput input;
    input.batchSize = 1;
    input.numQHeads = 4;
    input.numKvHeads = 1;
    input.maxQSeqLen = 2;
    input.headDim = 128;
    input.blockShapeX = 1;
    input.blockShapeY = 128;
    input.blockIndexStride = 16;
    input.qBlockStorageNum = 2;
    input.isPackedGQA = 1;
    input.aicCoreNum = 32;
    input.qSeqLens = {2};
    input.validBlockNums = {12, 3};
    ScheduleResult result;
    CHECK_EQ(BuildSchedule(input, result), ScheduleStatus::SUCCESS);
    std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
    CHECK_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), ScheduleStatus::SUCCESS);
    CHECK_EQ(metadata[MAGIC_INDEX], METADATA_MAGIC);
    CHECK_EQ(metadata[VERSION_INDEX], METADATA_VERSION);
    CHECK_EQ(metadata[METADATA_USED_SIZE_INDEX], static_cast<int32_t>(METADATA_USED_SIZE));
    CHECK_EQ(metadata[SA_USED_CORE_NUM_INDEX], 2);
    CHECK_EQ(metadata[SA_TOTAL_TASK_NUM_INDEX], 2);
    CHECK_EQ(metadata[FD_ACTIVE_CORE_NUM_INDEX], 5);
    CHECK_EQ(metadata[DECODE_PER_CORE_TASK_NUM_INDEX], 3);
    CHECK_EQ(metadata[COMBINE_TASK_NUM_INDEX], 1);
    CHECK_EQ(metadata[FD_SCHEDULE_FLAGS_INDEX],
             static_cast<int32_t>(FD_SCHEDULE_ENABLED | FD_ACTUAL_BLOCK_PREFIX));
    CHECK_EQ(metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX], 15);
    CHECK_EQ(metadata[FD_PARTIAL_TASK_NUM_INDEX], 4);
    CHECK_EQ(metadata[CONFIG_SIGNATURE_INDEX], static_cast<int32_t>(result.configSignature));
    CHECK_EQ(metadata[GetDecodeScheduleIndex(4, DECODE_BASE_TASK_START_INDEX)], 1);
    CHECK_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_BASE_TASK_INDEX)], 0);
    CHECK_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_PARTIAL_COUNT_INDEX)], 4);
    CHECK_EQ(metadata[METADATA_USED_SIZE], 0);
}

int main()
{
    TestPrefillSchedule();
    TestActualBlockPrefixDecode();
    TestDecodeShapeConstraints();
    TestZeroCountsAndRepeatedPrefix();
    TestSparseBlockCapacityGate();
    TestActiveCoreCapacityAcrossBaseTasks();
    TestPartialTaskNumCanExceedActiveCoreNum();
    TestPartialWorkspaceUpperBound();
    TestUnpackedInternalTaskModel();
    TestInvalidBlockCount();
    TestTndSeqUsedUsesStorageOffsets();
    TestTndQSeqValidation();
    TestMetadataEncoding();
    std::cout << "All sparse attention score metadata scheduler tests passed." << std::endl;
    return 0;
}
