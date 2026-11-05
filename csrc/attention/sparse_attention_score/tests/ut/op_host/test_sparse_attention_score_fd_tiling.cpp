/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_host/sparse_attention_score_tiling.h"

namespace {

constexpr uint64_t NORMAL_FP16_KEY = 10001;
constexpr uint64_t NORMAL_BF16_KEY = 10002;
constexpr uint64_t FD_FP16_KEY = 10005;
constexpr uint64_t FD_BF16_KEY = 10006;
constexpr uint64_t ARCH22_NORMAL_FP16_KEY = 20001;
constexpr uint64_t ARCH22_NORMAL_BF16_KEY = 20002;
constexpr uint64_t ARCH22_FD_FP16_KEY = 20003;
constexpr uint64_t ARCH22_FD_BF16_KEY = 20004;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t INNER_PRECISE = 4;
constexpr uint64_t ARCH35_AIC_NUM = 28;

void RunTilingCase(ge::DataType dtype, int64_t qTokens, int64_t qHeads, int64_t kvHeads,
                   int64_t topK, ge::graphStatus expectedStatus, uint64_t expectedKey,
                   const char *socVersion = "Ascend950",
                   uint64_t aicNum = ARCH35_AIC_NUM, bool validateFdRanges = false)
{
    int32_t actualQSeq[] = {static_cast<int32_t>(qTokens)};
    int32_t actualKvSeq[] = {static_cast<int32_t>(topK * BLOCK_SIZE)};
    optiling::SparseAttentionScoreCompileInfo compileInfo{};

    gert::TilingContextPara context(
        "SparseAttentionScore",
        {
            {{{qTokens, qHeads, HEAD_DIM}, {qTokens, qHeads, HEAD_DIM}}, dtype, ge::FORMAT_ND},
            {{{topK, BLOCK_SIZE, kvHeads, HEAD_DIM}, {topK, BLOCK_SIZE, kvHeads, HEAD_DIM}},
             dtype, ge::FORMAT_ND},
            {{{topK, BLOCK_SIZE, kvHeads, HEAD_DIM}, {topK, BLOCK_SIZE, kvHeads, HEAD_DIM}},
             dtype, ge::FORMAT_ND},
            {{{kvHeads, qTokens, topK}, {kvHeads, qTokens, topK}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, topK}, {1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{kvHeads, qTokens}, {kvHeads, qTokens}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualQSeq},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualKvSeq},
        },
        {
            {{{qTokens, qHeads, HEAD_DIM}, {qTokens, qHeads, HEAD_DIM}}, dtype, ge::FORMAT_ND},
        },
        {
            {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvHeads)},
            {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831845F)},
            {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(BLOCK_SIZE)},
            {"topK", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topK)},
            {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INNER_PRECISE)},
        },
        &compileInfo, socVersion, aicNum, 262144, 16384);

    if (!validateFdRanges) {
        ExecuteTestCase(context, expectedStatus, expectedKey);
        return;
    }

    ASSERT_EQ(expectedStatus, ge::GRAPH_SUCCESS);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectedKey);

    const uint32_t totalBaseTasks = static_cast<uint32_t>(qTokens * kvHeads);
    const uint32_t totalSplitTasks = totalBaseTasks * static_cast<uint32_t>(topK);
    const uint32_t usedCoreNum = tilingInfo.blockNum;
    const uint32_t perCoreTaskNum = (totalSplitTasks + usedCoreNum - 1) / usedCoreNum;

    optiling::SparseAttentionScoreTilingData tilingData(tilingInfo.tilingData.get());
    const uint32_t *coreTaskStart = tilingData.get_fdCoreTaskStart();
    const uint32_t *coreTaskEnd = tilingData.get_fdCoreTaskEnd();
    for (uint32_t core = 0; core < usedCoreNum; ++core) {
        EXPECT_EQ(coreTaskStart[core], std::min(core * perCoreTaskNum, totalSplitTasks));
        EXPECT_EQ(coreTaskEnd[core], std::min((core + 1) * perCoreTaskNum, totalSplitTasks));
    }

    const uint32_t *combineBaseTask = tilingData.get_fdCombineBaseTask();
    const uint32_t *partialStartByBase = tilingData.get_fdPartialStartByBase();
    const uint32_t *partialCountByBase = tilingData.get_fdPartialCountByBase();
    uint32_t combineTask = 0;
    uint32_t partialTask = 0;
    for (uint32_t task = 0; task < totalBaseTasks; ++task) {
        const uint32_t taskStart = task * static_cast<uint32_t>(topK);
        const uint32_t firstCore = taskStart / perCoreTaskNum;
        const uint32_t lastCore = (taskStart + static_cast<uint32_t>(topK) - 1) / perCoreTaskNum;
        const uint32_t splitCount = lastCore - firstCore + 1;
        if (splitCount > 1) {
            EXPECT_EQ(combineBaseTask[combineTask++], task);
            EXPECT_EQ(partialStartByBase[task], partialTask);
            EXPECT_EQ(partialCountByBase[task], splitCount);
            partialTask += splitCount;
        } else {
            EXPECT_EQ(partialCountByBase[task], 0U);
        }
    }
}

class SparseAttentionScoreFdTilingTest : public testing::Test {};

TEST_F(SparseAttentionScoreFdTilingTest, Bf16EligibleShapeAutomaticallyUsesFdKey)
{
    RunTilingCase(ge::DT_BF16, 1, 16, 1, 16, ge::GRAPH_SUCCESS, FD_BF16_KEY,
        "Ascend950", ARCH35_AIC_NUM, true);
}

TEST_F(SparseAttentionScoreFdTilingTest, Bf16EightBaseTasksUseCostModelRanges)
{
    RunTilingCase(ge::DT_BF16, 8, 16, 1, 16, ge::GRAPH_SUCCESS, FD_BF16_KEY,
        "Ascend950", ARCH35_AIC_NUM, true);
}

TEST_F(SparseAttentionScoreFdTilingTest, Fp16EligibleShapeUsesFdKey)
{
    RunTilingCase(ge::DT_FLOAT16, 1, 16, 1, 16, ge::GRAPH_SUCCESS, FD_FP16_KEY,
        "Ascend950", ARCH35_AIC_NUM, true);
}

TEST_F(SparseAttentionScoreFdTilingTest, TopKOneCannotCreateExtraShard)
{
    RunTilingCase(ge::DT_BF16, 1, 16, 1, 1, ge::GRAPH_SUCCESS, NORMAL_BF16_KEY);
}

TEST_F(SparseAttentionScoreFdTilingTest, TopKSeventeenIsOutsideFdGate)
{
    RunTilingCase(ge::DT_BF16, 1, 16, 1, 17, ge::GRAPH_SUCCESS, NORMAL_BF16_KEY);
}

TEST_F(SparseAttentionScoreFdTilingTest, TopKBelowFdGateFallsBack)
{
    RunTilingCase(ge::DT_BF16, 12, 16, 2, 2, ge::GRAPH_SUCCESS, NORMAL_BF16_KEY);
}

TEST_F(SparseAttentionScoreFdTilingTest, BaseTasksEqualAicFallsBack)
{
    RunTilingCase(ge::DT_BF16, 28, 1, 1, 16, ge::GRAPH_SUCCESS, NORMAL_BF16_KEY);
}

TEST_F(SparseAttentionScoreFdTilingTest, Arch22Bf16EligibleShapeAutomaticallyUsesFdKey)
{
    RunTilingCase(ge::DT_BF16, 1, 16, 1, 16, ge::GRAPH_SUCCESS,
        ARCH22_FD_BF16_KEY, "Ascend910B", 20, true);
}

TEST_F(SparseAttentionScoreFdTilingTest, Arch22Fp16EligibleShapeUsesFdKey)
{
    RunTilingCase(ge::DT_FLOAT16, 1, 16, 1, 16, ge::GRAPH_SUCCESS,
        ARCH22_FD_FP16_KEY, "Ascend910B", 20, true);
}

TEST_F(SparseAttentionScoreFdTilingTest, Arch22BaseTasksEqualAicFallsBack)
{
    RunTilingCase(ge::DT_BF16, 20, 1, 1, 16, ge::GRAPH_SUCCESS,
        ARCH22_NORMAL_BF16_KEY, "Ascend910B", 20);
}

TEST_F(SparseAttentionScoreFdTilingTest, Arch22TopKOneCannotCreateExtraShard)
{
    RunTilingCase(ge::DT_FLOAT16, 1, 16, 1, 1, ge::GRAPH_SUCCESS,
        ARCH22_NORMAL_FP16_KEY, "Ascend910B", 20);
}

}  // namespace
