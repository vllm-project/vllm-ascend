/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_q_seq_utils.h"
#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>

using namespace aicpu::generic_block_sparse_attention_metadata;
using namespace optiling::generic_block_sparse_attention_metadata;

namespace {

constexpr int64_t SUPPORTED_HEAD_DIM = 128;
constexpr int64_t SUPPORTED_BLOCK_SHAPE_X = 1;
constexpr int64_t SUPPORTED_BLOCK_SHAPE_Y = 128;
constexpr int64_t DEFAULT_BLOCK_INDEX_STRIDE = 16;
constexpr int64_t PACKED_GQA = 1;
constexpr int64_t UNPACKED_GQA = 0;

constexpr int64_t DEFAULT_BATCH_SIZE = 2;
constexpr int64_t DEFAULT_NUM_Q_HEADS = 8;
constexpr int64_t DEFAULT_NUM_KV_HEADS = 2;
constexpr int64_t DEFAULT_MAX_Q_SEQ_LEN = 3;
constexpr int64_t UNSPECIFIED_MAX_Q_SEQ_LEN = -1;
constexpr int64_t DEFAULT_Q_BLOCK_STORAGE_NUM = 6;
constexpr int64_t DEFAULT_AIC_CORE_NUM = 8;
constexpr int64_t DEFAULT_FIRST_Q_SEQ_LEN = 2;
constexpr int64_t DEFAULT_SECOND_Q_SEQ_LEN = 1;
constexpr int64_t DEFAULT_TOTAL_Q_TOKEN_NUM = 3;
constexpr int64_t DEFAULT_SA_TASK_NUM = 6;
constexpr int64_t DEFAULT_GROUP_SIZE = 4;
constexpr int64_t DEFAULT_DECODE_TASK_INDEX = 4;

constexpr int64_t DECODE_BATCH_SIZE = 1;
constexpr int64_t DECODE_NUM_Q_HEADS = 4;
constexpr int64_t DECODE_NUM_KV_HEADS = 1;
constexpr int64_t DECODE_Q_SEQ_LEN = 2;
constexpr int64_t DECODE_Q_BLOCK_STORAGE_NUM = 2;
constexpr int64_t DECODE_AIC_CORE_NUM = 32;
constexpr int64_t DECODE_FIRST_BLOCK_NUM = 12;
constexpr int64_t DECODE_SECOND_BLOCK_NUM = 3;
constexpr int64_t DECODE_TOTAL_VALID_BLOCK_NUM = DECODE_FIRST_BLOCK_NUM + DECODE_SECOND_BLOCK_NUM;
constexpr int64_t DECODE_ACTIVE_CORE_NUM = 5;
constexpr int64_t DECODE_PER_CORE_TASK_NUM = 3;
constexpr int64_t DECODE_FOURTH_CORE_INDEX = 3;
constexpr int64_t DECODE_FIFTH_CORE_INDEX = 4;
constexpr int64_t DECODE_FIRST_UNUSED_CORE_INDEX = 5;
constexpr int64_t DECODE_FOURTH_CORE_FIRST_BLOCK = 9;
constexpr int64_t DECODE_COMBINE_PARTIAL_COUNT = 4;

constexpr int64_t UNSUPPORTED_DIM = 513;
constexpr int64_t MAX_BLOCK_CAPACITY = 256;
constexpr int64_t ACTIVE_BASE_TASK_NUM = 8;
constexpr int64_t ACTIVE_AIC_CORE_NUM = 28;
constexpr int64_t ACTIVE_VALID_BLOCK_NUM = 16;
constexpr int64_t PARTIAL_AIC_CORE_NUM = 7;
constexpr int64_t PARTIAL_VALID_BLOCK_NUM = 3;
constexpr int64_t PARTIAL_ACTIVE_CORE_NUM = 3;
constexpr int64_t PARTIAL_COMBINE_TASK_NUM = 2;
constexpr int64_t PARTIAL_TASK_NUM = 4;
constexpr uint32_t HOST_CAPACITY_AIC_CORE_NUM = 28U;
constexpr uint32_t HOST_PARTIAL_CAPACITY = 35U;
constexpr uint32_t FD_BASE_TASK_RATIO_NUMERATOR = 3U;
constexpr uint32_t FD_BASE_TASK_RATIO_DENOMINATOR = 10U;

constexpr int64_t UNPACKED_TASK_INDEX = 5;
constexpr int64_t TND_Q_STORAGE_LEN_PER_BATCH = 4;
constexpr int64_t TND_TOTAL_Q_STORAGE_LEN = 8;
constexpr int64_t TND_FIRST_BATCH_USED_Q = 2;
constexpr int64_t TND_SECOND_BATCH_USED_Q = 3;
constexpr int64_t TND_INVALID_FIRST_BATCH_USED_Q = 5;
constexpr int64_t TND_INVALID_TOTAL_Q_STORAGE_LEN = 7;
constexpr int32_t TND_FIRST_BLOCK_COUNT = 1;

class GenericBlockSparseAttentionMetadataSchedulerTest : public testing::Test {};

ScheduleInput MakeInput()
{
    ScheduleInput input;
    input.batchSize = DEFAULT_BATCH_SIZE;
    input.numQHeads = DEFAULT_NUM_Q_HEADS;
    input.numKvHeads = DEFAULT_NUM_KV_HEADS;
    input.maxQSeqLen = DEFAULT_MAX_Q_SEQ_LEN;
    input.headDim = SUPPORTED_HEAD_DIM;
    input.blockShapeX = SUPPORTED_BLOCK_SHAPE_X;
    input.blockShapeY = SUPPORTED_BLOCK_SHAPE_Y;
    input.blockIndexStride = DEFAULT_BLOCK_INDEX_STRIDE;
    input.qBlockStorageNum = DEFAULT_Q_BLOCK_STORAGE_NUM;
    input.isPackedGQA = PACKED_GQA;
    input.aicCoreNum = DEFAULT_AIC_CORE_NUM;
    input.qSeqLens = {DEFAULT_FIRST_Q_SEQ_LEN, DEFAULT_SECOND_Q_SEQ_LEN};
    input.validBlockNums.assign(DEFAULT_Q_BLOCK_STORAGE_NUM, 0);
    return input;
}

ScheduleInput MakeDecodeInput()
{
    ScheduleInput input;
    input.batchSize = DECODE_BATCH_SIZE;
    input.numQHeads = DECODE_NUM_Q_HEADS;
    input.numKvHeads = DECODE_NUM_KV_HEADS;
    input.maxQSeqLen = DECODE_Q_SEQ_LEN;
    input.headDim = SUPPORTED_HEAD_DIM;
    input.blockShapeX = SUPPORTED_BLOCK_SHAPE_X;
    input.blockShapeY = SUPPORTED_BLOCK_SHAPE_Y;
    input.blockIndexStride = DEFAULT_BLOCK_INDEX_STRIDE;
    input.qBlockStorageNum = DECODE_Q_BLOCK_STORAGE_NUM;
    input.isPackedGQA = PACKED_GQA;
    input.aicCoreNum = DECODE_AIC_CORE_NUM;
    input.qSeqLens = {DECODE_Q_SEQ_LEN};
    input.validBlockNums = {DECODE_FIRST_BLOCK_NUM, DECODE_SECOND_BLOCK_NUM};
    return input;
}

uint32_t GetHostFdPartialCapacity(uint32_t aicCoreNum)
{
    const uint32_t maxNonEmptyBaseTaskNum =
        aicCoreNum == 0U ? 0U :
                           std::min<uint32_t>(MAX_COMBINE_TASK_NUM, (aicCoreNum * FD_BASE_TASK_RATIO_NUMERATOR - 1U) /
                                                                        FD_BASE_TASK_RATIO_DENOMINATOR);
    const uint32_t maxActiveCoreNum = std::min<uint32_t>(aicCoreNum, MAX_FD_ACTIVE_CORE_NUM);
    return maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ? 0U : maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PrefillSchedule)
{
    ScheduleInput input = MakeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.totalQTokenNum, DEFAULT_TOTAL_Q_TOKEN_NUM);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_SA_TASK_NUM);
    ASSERT_EQ(result.saUsedCoreNum, DEFAULT_SA_TASK_NUM);
    ASSERT_EQ(result.sparseHeadNum, DEFAULT_NUM_KV_HEADS);
    ASSERT_EQ(result.groupSize, DEFAULT_GROUP_SIZE);
    ASSERT_EQ(result.fdActiveCoreNum, 0);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, DEFAULT_DECODE_TASK_INDEX, task), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(task.qUnit, DEFAULT_FIRST_Q_SEQ_LEN);
    ASSERT_EQ(task.batchIdx, 1);
    ASSERT_EQ(task.qTokenInBatch, 0);
    ASSERT_EQ(task.kvHeadIdx, 0);
    ASSERT_EQ(task.qHeadStart, 0);
    ASSERT_EQ(task.qHeadCount, DEFAULT_GROUP_SIZE);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActualBlockPrefixDecode)
{
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], DECODE_FIRST_BLOCK_NUM);
    ASSERT_EQ(result.blockPrefix[DECODE_Q_BLOCK_STORAGE_NUM], DECODE_TOTAL_VALID_BLOCK_NUM);
    ASSERT_EQ(result.fdActiveCoreNum, DECODE_ACTIVE_CORE_NUM);
    ASSERT_EQ(result.decodePerCoreTaskNum, DECODE_PER_CORE_TASK_NUM);

    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskEnd, 1);
    ASSERT_EQ(result.decodeSchedules[0].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].lastBlockEnd, DECODE_PER_CORE_TASK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FOURTH_CORE_INDEX].firstBlockStart, DECODE_FOURTH_CORE_FIRST_BLOCK);
    ASSERT_EQ(result.decodeSchedules[DECODE_FOURTH_CORE_INDEX].lastBlockEnd, DECODE_FIRST_BLOCK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].baseTaskStart, 1);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].baseTaskEnd, DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].lastBlockEnd, DECODE_SECOND_BLOCK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIRST_UNUSED_CORE_INDEX].baseTaskEnd, 0);

    ASSERT_EQ(result.combineTaskNum, 1);
    ASSERT_EQ(result.combineSchedules[0].baseTask, 0);
    ASSERT_EQ(result.combineSchedules[0].firstCore, 0);
    ASSERT_EQ(result.combineSchedules[0].partialStart, 0);
    ASSERT_EQ(result.combineSchedules[0].partialCount, DECODE_COMBINE_PARTIAL_COUNT);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, DecodeShapeConstraints)
{
    for (int64_t headDim : {1, 64, 127, 128, 129, 256, 257, 384, 511, 512}) {
        ScheduleInput input = MakeDecodeInput();
        ScheduleResult result;
        input.headDim = headDim;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_GT(result.fdActiveCoreNum, 0);
    }
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    input.headDim = UNSUPPORTED_DIM;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, GeneralizedYUsesFdSchedule)
{
    for (int64_t y = 1; y <= 128; ++y) {
        ScheduleInput input = MakeDecodeInput();
        input.blockShapeY = y;
        ScheduleResult result;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_GT(result.fdActiveCoreNum, 0);
        ASSERT_NE(static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED, 0U);
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ZeroCountsAndRepeatedPrefix)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = DEFAULT_MAX_Q_SEQ_LEN;
    input.numQHeads = 1;
    input.numKvHeads = 1;
    input.aicCoreNum = DECODE_AIC_CORE_NUM;
    input.qSeqLens = {DEFAULT_MAX_Q_SEQ_LEN};
    input.validBlockNums = {0, DECODE_FIRST_BLOCK_NUM, DECODE_SECOND_BLOCK_NUM};
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], 0);
    ASSERT_EQ(result.blockPrefix[DECODE_Q_BLOCK_STORAGE_NUM], DECODE_FIRST_BLOCK_NUM);
    ASSERT_EQ(result.blockPrefix[DEFAULT_MAX_Q_SEQ_LEN], DECODE_TOTAL_VALID_BLOCK_NUM);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, SparseBlockCapacityUpTo256)
{
    ScheduleInput lhs = MakeDecodeInput();
    lhs.blockIndexStride = DECODE_FIRST_BLOCK_NUM;
    ScheduleInput rhs = lhs;
    rhs.blockIndexStride = MAX_BLOCK_CAPACITY;
    ScheduleResult lhsResult;
    ScheduleResult rhsResult;
    ASSERT_EQ(BuildSchedule(lhs, lhsResult), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(BuildSchedule(rhs, rhsResult), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_TRUE(lhsResult.fdActiveCoreNum > 0);
    ASSERT_EQ(rhsResult.fdActiveCoreNum, lhsResult.fdActiveCoreNum);
    ASSERT_EQ(rhsResult.decodePerCoreTaskNum, lhsResult.decodePerCoreTaskNum);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, DecodeSupportsSparseBlockCapacityAbove256)
{
    ScheduleInput input = MakeDecodeInput();
    input.blockIndexStride = 820;
    input.validBlockNums.assign(static_cast<size_t>(input.qSeqLens[0] * input.numKvHeads), 820);
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    EXPECT_GT(result.fdActiveCoreNum, 0);
    EXPECT_EQ(result.fdTotalFlatTaskNum, 820 * input.qSeqLens[0] * input.numKvHeads);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActiveCoreCapacityAcrossBaseTasks)
{
    ScheduleInput input = MakeDecodeInput();
    input.maxQSeqLen = ACTIVE_BASE_TASK_NUM;
    input.qBlockStorageNum = ACTIVE_BASE_TASK_NUM;
    input.aicCoreNum = ACTIVE_AIC_CORE_NUM;
    input.qSeqLens = {ACTIVE_BASE_TASK_NUM};
    input.validBlockNums.assign(ACTIVE_BASE_TASK_NUM, ACTIVE_VALID_BLOCK_NUM);

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_TRUE(result.fdActiveCoreNum > ACTIVE_VALID_BLOCK_NUM);
    ASSERT_TRUE(result.fdActiveCoreNum <= input.aicCoreNum);
    ASSERT_EQ(result.combineTaskNum, ACTIVE_BASE_TASK_NUM);
    int64_t partialTaskNum = 0;
    for (int64_t task = 0; task < result.combineTaskNum; ++task) {
        const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(task)];
        ASSERT_TRUE(combine.partialCount > 1);
        ASSERT_TRUE(combine.partialCount <= static_cast<int64_t>(MAX_FD_PARTIAL_PER_BASE_TASK));
        ASSERT_TRUE(combine.firstCore + combine.partialCount <= result.fdActiveCoreNum);
        partialTaskNum += combine.partialCount;
    }
    ASSERT_EQ(result.fdPartialTaskNum, partialTaskNum);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialTaskNumCanExceedActiveCoreNum)
{
    ScheduleInput input = MakeDecodeInput();
    input.aicCoreNum = PARTIAL_AIC_CORE_NUM;
    input.validBlockNums.assign(DECODE_Q_BLOCK_STORAGE_NUM, PARTIAL_VALID_BLOCK_NUM);

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, PARTIAL_ACTIVE_CORE_NUM);
    ASSERT_EQ(result.combineTaskNum, PARTIAL_COMBINE_TASK_NUM);
    ASSERT_EQ(result.fdPartialTaskNum, PARTIAL_TASK_NUM);
    ASSERT_TRUE(result.fdPartialTaskNum > result.fdActiveCoreNum);
    ASSERT_EQ(result.fdPartialTaskNum, static_cast<int64_t>(input.validBlockNums.size()) + result.fdActiveCoreNum - 1);
    ASSERT_TRUE(result.fdPartialTaskNum <= GetHostFdPartialCapacity(static_cast<uint32_t>(input.aicCoreNum)));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialWorkspaceUpperBound)
{
    ASSERT_EQ(GetHostFdPartialCapacity(0U), 0U);
    ASSERT_EQ(GetHostFdPartialCapacity(HOST_CAPACITY_AIC_CORE_NUM), HOST_PARTIAL_CAPACITY);
    for (uint32_t aicCoreNum = 1U; aicCoreNum <= MAX_FD_ACTIVE_CORE_NUM; ++aicCoreNum) {
        const uint32_t maxNonEmptyBaseTaskNum =
            (aicCoreNum * FD_BASE_TASK_RATIO_NUMERATOR - 1U) / FD_BASE_TASK_RATIO_DENOMINATOR;
        const uint32_t hostCapacity = GetHostFdPartialCapacity(aicCoreNum);
        for (uint32_t baseTaskNum = 1U; baseTaskNum <= maxNonEmptyBaseTaskNum; ++baseTaskNum) {
            for (int64_t validBlockNum = 1; validBlockNum <= ACTIVE_VALID_BLOCK_NUM; ++validBlockNum) {
                ScheduleInput input = MakeDecodeInput();
                input.maxQSeqLen = baseTaskNum;
                input.qBlockStorageNum = baseTaskNum;
                input.aicCoreNum = aicCoreNum;
                input.qSeqLens = {baseTaskNum};
                input.validBlockNums.assign(baseTaskNum, validBlockNum);

                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
                if ((static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED) == 0U) {
                    continue;
                }
                const int64_t actualIntersectionUpperBound =
                    static_cast<int64_t>(baseTaskNum) + result.fdActiveCoreNum - 1;
                ASSERT_TRUE(result.fdPartialTaskNum <= actualIntersectionUpperBound);
                ASSERT_TRUE(result.fdPartialTaskNum <= hostCapacity);
                for (int64_t combineIdx = 0; combineIdx < result.combineTaskNum; ++combineIdx) {
                    const CombineSchedule &combine = result.combineSchedules[static_cast<size_t>(combineIdx)];
                    ASSERT_TRUE(combine.partialStart >= 0);
                    ASSERT_TRUE(combine.partialStart + combine.partialCount <= result.fdPartialTaskNum);
                }
            }
        }
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, UnpackedInternalTaskModel)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 1;
    input.qSeqLens = {1};
    input.isPackedGQA = UNPACKED_GQA;
    input.validBlockNums.assign(DEFAULT_NUM_Q_HEADS, 0);
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.sparseHeadNum, DEFAULT_NUM_Q_HEADS);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_NUM_Q_HEADS);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, UNPACKED_TASK_INDEX, task), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(task.sparseHeadIdx, UNPACKED_TASK_INDEX);
    ASSERT_EQ(task.qHeadStart, UNPACKED_TASK_INDEX);
    ASSERT_EQ(task.qHeadCount, 1);
    ASSERT_EQ(task.kvHeadIdx, 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, InvalidBlockCount)
{
    ScheduleInput input = MakeInput();
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_INVALID_PARAM);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ResidualBlockExtendsVirtualCapacity)
{
    ScheduleInput input = MakeInput();
    input.residualBlockMode = 1;
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[1], input.blockIndexStride + 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndSeqUsedUsesStorageOffsets)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN},
                                   {TND_FIRST_BATCH_USED_Q, TND_SECOND_BATCH_USED_Q}, true, SUPPORTED_BLOCK_SHAPE_X,
                                   TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens, qStorageBlockStarts));
    ASSERT_EQ(actualQSeqLens[0], TND_FIRST_BATCH_USED_Q);
    ASSERT_EQ(actualQSeqLens[1], TND_SECOND_BATCH_USED_Q);
    ASSERT_EQ(qStorageBlockStarts[0], 0);
    ASSERT_EQ(qStorageBlockStarts[1], TND_Q_STORAGE_LEN_PER_BATCH);

    std::array<int32_t, TND_TOTAL_Q_STORAGE_LEN> counts{};
    std::iota(counts.begin(), counts.end(), TND_FIRST_BLOCK_COUNT);
    std::vector<int64_t> validBlockNums;
    ASSERT_TRUE(GatherTndValidBlockNums(counts.data(), 1, TND_TOTAL_Q_STORAGE_LEN, SUPPORTED_BLOCK_SHAPE_X,
                                        TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens, qStorageBlockStarts, validBlockNums));
    ASSERT_EQ(validBlockNums.size(), static_cast<size_t>(TND_FIRST_BATCH_USED_Q + TND_SECOND_BATCH_USED_Q));
    ASSERT_EQ(validBlockNums[0], counts[0]);
    ASSERT_EQ(validBlockNums[1], counts[1]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q], counts[TND_Q_STORAGE_LEN_PER_BATCH]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q + 1], counts[TND_Q_STORAGE_LEN_PER_BATCH + 1]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q + TND_SECOND_BATCH_USED_Q - 1],
              counts[TND_Q_STORAGE_LEN_PER_BATCH + TND_SECOND_BATCH_USED_Q - 1]);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, SchedulerAllowsUnspecifiedMaxQSeqLen)
{
    ScheduleInput input = MakeInput();
    input.maxQSeqLen = UNSPECIFIED_MAX_Q_SEQ_LEN;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.totalQTokenNum, DEFAULT_TOTAL_Q_TOKEN_NUM);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_SA_TASK_NUM);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndQSeqWithoutSeqUsedUsesStorageLengths)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                   SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                   qStorageBlockStarts));
    ASSERT_EQ(actualQSeqLens[0], TND_Q_STORAGE_LEN_PER_BATCH);
    ASSERT_EQ(actualQSeqLens[1], TND_Q_STORAGE_LEN_PER_BATCH);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndQSeqValidation)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN},
                                    {TND_INVALID_FIRST_BATCH_USED_Q, TND_SECOND_BATCH_USED_Q}, true,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_SECOND_BATCH_USED_Q}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_Q_STORAGE_LEN_PER_BATCH, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({1, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_INVALID_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, MetadataEncoding)
{
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
    ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(metadata[MAGIC_INDEX], METADATA_MAGIC);
    ASSERT_EQ(metadata[VERSION_INDEX], METADATA_VERSION);
    ASSERT_EQ(metadata[METADATA_USED_SIZE_INDEX], static_cast<int32_t>(METADATA_USED_SIZE));
    ASSERT_EQ(metadata[SA_USED_CORE_NUM_INDEX], DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(metadata[SA_TOTAL_TASK_NUM_INDEX], DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(metadata[FD_ACTIVE_CORE_NUM_INDEX], DECODE_ACTIVE_CORE_NUM);
    ASSERT_EQ(metadata[DECODE_PER_CORE_TASK_NUM_INDEX], DECODE_PER_CORE_TASK_NUM);
    ASSERT_EQ(metadata[COMBINE_TASK_NUM_INDEX], 1);
    ASSERT_EQ(metadata[FD_SCHEDULE_FLAGS_INDEX], static_cast<int32_t>(FD_SCHEDULE_ENABLED | FD_ACTUAL_BLOCK_PREFIX));
    ASSERT_EQ(metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX], DECODE_TOTAL_VALID_BLOCK_NUM);
    ASSERT_EQ(metadata[FD_PARTIAL_TASK_NUM_INDEX], DECODE_COMBINE_PARTIAL_COUNT);
    ASSERT_EQ(metadata[CONFIG_SIGNATURE_INDEX], static_cast<int32_t>(result.configSignature));
    ASSERT_EQ(metadata[GetDecodeScheduleIndex(DECODE_FIFTH_CORE_INDEX, DECODE_BASE_TASK_START_INDEX)], 1);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_BASE_TASK_INDEX)], 0);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_PARTIAL_COUNT_INDEX)], DECODE_COMBINE_PARTIAL_COUNT);
    ASSERT_EQ(metadata[METADATA_USED_SIZE], 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ScheduleInputDefaultsMatchInterface)
{
    ScheduleInput input;
    EXPECT_EQ(input.residualBlockMode, 0);
    EXPECT_EQ(input.isConsistentTopk, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ConfigSignatureTracksResidualAndConsistentTopk)
{
    ScheduleInput base = MakeDecodeInput();
    ScheduleResult baseResult;
    ASSERT_EQ(BuildSchedule(base, baseResult), BSAScheduleStatus::BSA_SUCCESS);
    const uint32_t baseSignature = baseResult.configSignature;
    EXPECT_EQ(baseSignature, CalculateConfigSignature(base));

    ScheduleInput residualChanged = MakeDecodeInput();
    residualChanged.residualBlockMode = 1;
    EXPECT_NE(CalculateConfigSignature(residualChanged), baseSignature);

    ScheduleInput topkChanged = MakeDecodeInput();
    topkChanged.isConsistentTopk = 1;
    EXPECT_NE(CalculateConfigSignature(topkChanged), baseSignature);

    // The reserved topk flag affects the signature, but not scheduling.
    ScheduleResult topkResult;
    ASSERT_EQ(BuildSchedule(topkChanged, topkResult), BSAScheduleStatus::BSA_SUCCESS);
    EXPECT_EQ(topkResult.blockPrefix, baseResult.blockPrefix);
    EXPECT_EQ(topkResult.saUsedCoreNum, baseResult.saUsedCoreNum);
    EXPECT_EQ(topkResult.saTotalTaskNum, baseResult.saTotalTaskNum);
    EXPECT_EQ(topkResult.fdPartialTaskNum, baseResult.fdPartialTaskNum);
    EXPECT_EQ(topkResult.combineTaskNum, baseResult.combineTaskNum);
    EXPECT_NE(topkResult.configSignature, baseSignature);
}

} // namespace
