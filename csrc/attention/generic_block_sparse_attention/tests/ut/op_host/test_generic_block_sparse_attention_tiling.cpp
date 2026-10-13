/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

class generic_block_sparse_attention_tiling_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "generic_block_sparse_attention_tiling_ut TearDown" << endl;
    }
};

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr float kScale = 1.0f / sqrt(static_cast<float>(kD));

// Expected host tiling keys (BSA-style bitfields; must match GenerateTilingKey).
constexpr uint64_t kFp16Tiling910B = 9200000071001002ULL;
constexpr uint64_t kFp16TndTndTiling910B = 9200000030001002ULL;
constexpr uint64_t kBf16Tiling910B = 9200000071023222ULL;
constexpr uint64_t kFp16HalfSmTiling910B = 9200000071101002ULL;
constexpr uint64_t kFp16Tiling950 = 9250000071101002ULL;
constexpr uint64_t kBf16Tiling950 = 9250000071123222ULL;
constexpr uint64_t kFp8Fp16Tiling950 = 9250000071101012ULL;
constexpr uint64_t kFp8Bf16Tiling950 = 9250000071101022ULL;

std::vector<int64_t> kBlockShape = {kBlockShapeX, kBlockSize};

std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(ge::DataType qkvDtype, int64_t headDim = kD,
                                                                   int64_t numHeads = kN1,
                                                                   int64_t kvHeads = kN2)
{
    return {
        {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, qkvDtype, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kvHeads, headDim}, {kMaxBlocks, kBlockSize, kvHeads, headDim}}, qkvDtype,
         ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kvHeads, headDim}, {kMaxBlocks, kBlockSize, kvHeads, headDim}}, qkvDtype,
         ge::FORMAT_ND},
        {{{kvHeads, kTotalQBlocks, kTopK}, {kvHeads, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kvHeads, kTotalQBlocks}, {kvHeads, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch}, {kBatch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kBatch, kMaxBlocks}, {kBatch, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::TensorDescription> MakeDenseInputs(ge::DataType qkvDtype, int64_t headDim = kD,
                                                                        int64_t topK = kTopK)
{
    auto inputs = MakeInputs(qkvDtype, headDim);
    inputs[1] = {{{kS2, kN2, headDim}, {kS2, kN2, headDim}}, qkvDtype, ge::FORMAT_ND};
    inputs[2] = inputs[1];
    inputs[3] = {{{kN2, kTotalQBlocks, topK}, {kN2, kTotalQBlocks, topK}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    return inputs;
}

uint32_t ReadTilingU32(const TilingInfo &tilingInfo, size_t byteOffset)
{
    uint32_t value = 0;
    std::memcpy(&value, tilingInfo.tilingData.get() + byteOffset, sizeof(uint32_t));
    return value;
}

std::vector<gert::TilingContextPara::TensorDescription> MakeOutputs(ge::DataType outDtype, bool returnLse,
                                                                    int64_t headDim = kD,
                                                                    int64_t numHeads = kN1)
{
    if (returnLse) {
        return {
            {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, outDtype, ge::FORMAT_ND},
            {{{kT, numHeads, 1}, {kT, numHeads, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
    }
    return {
        {{{kT, numHeads, headDim}, {kT, numHeads, headDim}}, outDtype, ge::FORMAT_ND},
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(
    int64_t maskType, int64_t quantType, int64_t softmaxPrec, int64_t returnLse, float dstTypeMax = 0.0f,
    int64_t winLeft = -1, int64_t winRight = -1, int64_t layoutSparsePattern = 4, const std::string &layoutQ = "TND",
    const std::string &layoutKv = "PA_BBND", int64_t residualBlockMode = 0, bool isConsistentTopk = false,
    const std::vector<int64_t> &blockShape = kBlockShape, float scaleValue = kScale)
{
    return {
        {"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(blockShape)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutKv)},
        {"layout_sparse_pattern", Ops::Transformer::AnyValue::CreateFrom<int64_t>(layoutSparsePattern)},
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
        {"mask_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskType)},
        {"quant_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantType)},
        {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(dstTypeMax)},
        {"softmax_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(softmaxPrec)},
        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winLeft)},
        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(winRight)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(returnLse)},
        {"residual_block_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(residualBlockMode)},
        {"is_consistent_topk", Ops::Transformer::AnyValue::CreateFrom<bool>(isConsistentTopk)},
    };
}
} // namespace

TEST_F(generic_block_sparse_attention_tiling_ut, generalized_y_arch35)
{
    for (int64_t y : {1, 3, 16, 17, 63, 64, 65, 95, 127, 128}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (bool paged : {false, true}) {
                auto inputs = MakeInputs(dtype);
                if (!paged) {
                    inputs[1] = {{{kS2, kN2, kD}, {kS2, kN2, kD}}, dtype, ge::FORMAT_ND};
                    inputs[2] = inputs[1];
                    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
                    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
                }
                GenericBlockSparseAttentionCompileInfo compileInfo;
                const uint64_t key = paged ? (dtype == ge::DT_FLOAT16 ? kFp16Tiling950 : kBf16Tiling950) :
                    (dtype == ge::DT_FLOAT16 ? 9250000030101002ULL : 9250000030123222ULL);
                gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs, MakeOutputs(dtype, false),
                    MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 1, "TND", paged ? "PA_BBND" : "TND", 0, false,
                              {1, y}),
                    &compileInfo, "Ascend950", 32, 262144);
                ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, key);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, generalized_y_rejects_out_of_range)
{
    for (int64_t y : {int64_t(0), int64_t(-1), int64_t(129), int64_t(256), int64_t(257), int64_t(4294967297LL)}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
            MakeOutputs(ge::DT_FLOAT16, false),
            MakeAttrs(1, 0, 1, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", 0, false, {1, y}),
            &compileInfo, "Ascend950", 32, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, generalized_y_rejects_arch22_and_fp8)
{
    for (bool fp8 : {false, true}) {
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara ctx("GenericBlockSparseAttention",
            MakeInputs(fp8 ? ge::DT_FLOAT8_E4M3FN : ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
            MakeAttrs(1, fp8 ? 5 : 0, fp8 ? 1 : 0, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", 0, false,
                      {1, 64}),
            &compileInfo, fp8 ? "Ascend950" : "Ascend910B", 32, 262144);
        ExecuteTestCase(ctx, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16HalfSmTiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_return_softmax_lse)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 0, 0, 1), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_emits_refactored_tile_policy)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);

    // Nq/Nkv=4 selects qBaseTile=16. Low precision currently caps the
    // policy's kvBaseTile=512 to 256.
    const std::array<uint32_t, 14> expectedTail = {
        16U, 256U,
        16U, 256U, 128U, 128U,
        16U, 128U, 256U, 256U,
        1U, 2U, 2U, 3U,
    };
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(expectedTail));
    std::array<uint32_t, 14> actualTail = {};
    std::memcpy(actualTail.data(), tilingInfo.tilingData.get() + tilingInfo.tilingDataSize - sizeof(actualTail),
                sizeof(actualTail));
    EXPECT_EQ(actualTail, expectedTail);
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_emits_head_dim_256_tile_policy)
{
    constexpr int64_t headDim = 256;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16, headDim),
        MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend950", 56, 262144);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);

    const std::array<uint32_t, 14> expectedTail = {
        16U, 512U,
        16U, 128U, 256U, 256U,
        16U, 256U, 512U, 512U,
        1U, 2U, 1U, 3U,
    };
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(expectedTail));
    std::array<uint32_t, 14> actualTail = {};
    std::memcpy(actualTail.data(), tilingInfo.tilingData.get() + tilingInfo.tilingDataSize - sizeof(actualTail),
                sizeof(actualTail));
    EXPECT_EQ(actualTail, expectedTail);
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_high_precision_base_tile_policy)
{
    constexpr std::array<int64_t, 8> groupSizes = {1, 16, 17, 32, 33, 64, 65, 128};
    constexpr std::array<int64_t, 4> headDims = {1, 128, 129, 256};
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t headDim : headDims) {
            for (int64_t groupSize : groupSizes) {
                SCOPED_TRACE(testing::Message() << "dtype=" << static_cast<int>(dtype) << ", D=" << headDim
                                                << ", groupSize=" << groupSize);
                GenericBlockSparseAttentionCompileInfo compileInfo;
                gert::TilingContextPara tilingContextPara(
                    "GenericBlockSparseAttention", MakeInputs(dtype, headDim, groupSize, 1),
                    MakeOutputs(dtype, false, headDim, groupSize),
                    MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", 0, false, kBlockShape,
                              1.0f / sqrt(static_cast<float>(headDim))),
                    &compileInfo, "Ascend950", 56, 262144);
                TilingInfo tilingInfo;
                ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
                ASSERT_NE(tilingInfo.tilingData, nullptr);

                const uint32_t qBaseTile = groupSize <= 16 ? 16U :
                                               groupSize <= 32 ? 32U : groupSize <= 64 ? 64U : 128U;
                const uint32_t kvBaseTile = qBaseTile <= 32U ? 512U : qBaseTile <= 64U ? 256U : 128U;
                const bool smallHeadDim = headDim <= 128;
                const uint32_t embeddingSizeAligned = smallHeadDim ? 128U : 256U;
                const uint32_t mm1L1TileN = smallHeadDim && qBaseTile <= 64U ? 256U : 128U;
                const uint32_t vL1BufNum = smallHeadDim || qBaseTile > 32U ? 2U : 1U;
                const std::array<uint32_t, 14> expectedTail = {
                    qBaseTile, kvBaseTile,
                    qBaseTile, mm1L1TileN, embeddingSizeAligned, embeddingSizeAligned,
                    qBaseTile, embeddingSizeAligned, kvBaseTile, kvBaseTile,
                    1U, 2U, vL1BufNum, 3U,
                };
                ASSERT_GE(tilingInfo.tilingDataSize, sizeof(expectedTail));
                std::array<uint32_t, 14> actualTail = {};
                std::memcpy(actualTail.data(), tilingInfo.tilingData.get() + tilingInfo.tilingDataSize -
                                                    sizeof(actualTail),
                            sizeof(actualTail));
                EXPECT_EQ(actualTail, expectedTail);
            }
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_emits_head_dim_512_tile_policy)
{
    for (int64_t headDim : {257, 384, 511, 512}) {
        for (int64_t precision : {0, 1}) {
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara tilingContextPara(
                "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16, headDim),
                MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, precision, 0),
                &compileInfo, "Ascend950", 56, 262144);
            TilingInfo tilingInfo;
            ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
            ASSERT_NE(tilingInfo.tilingData, nullptr);
            const uint32_t kvTile = precision == 0 ? 384U : 256U;
            const std::array<uint32_t, 14> expectedTail = {
                16U, kvTile, 16U, 128U, 512U, 128U,
                16U, 512U, kvTile, kvTile, 1U, 2U, 1U, 3U,
            };
            std::array<uint32_t, 14> actualTail = {};
            ASSERT_GE(tilingInfo.tilingDataSize, sizeof(actualTail));
            std::memcpy(actualTail.data(), tilingInfo.tilingData.get() + tilingInfo.tilingDataSize - sizeof(actualTail),
                        sizeof(actualTail));
            EXPECT_EQ(actualTail, expectedTail);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, nonquant_950_rejects_head_dim_over_512)
{
    constexpr int64_t headDim = 513;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16, headDim),
        MakeOutputs(ge::DT_FLOAT16, false, headDim), MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kBf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_fp16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Fp16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, tnd_paged_fp8_out_bf16_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 5, 1, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp8Bf16Tiling950);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_layout)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_layout_sparse_pattern)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 1), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_residual_block_mode)
{
    for (int64_t residualMode : {-1, 1, 2}) {
        SCOPED_TRACE(residualMode);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
            MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", residualMode), &compileInfo, "Ascend910B", 40,
            196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_block_mode_arch35_non_quant)
{
    for (bool paged : {false, true}) {
        for (bool returnLse : {false, true}) {
            auto inputs = MakeInputs(ge::DT_FLOAT16);
            if (!paged) {
                inputs[1] = {{{kS2, kN2, kD}, {kS2, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
                inputs[2] = inputs[1];
                inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
                inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
            }
            const uint64_t baseKey = paged ? kFp16Tiling950 : 9250000030101002ULL;
            const uint64_t lseOffset = returnLse ? 100000000ULL : 0ULL;
            constexpr uint64_t residualOffset = 1000000000ULL;
            GenericBlockSparseAttentionCompileInfo compileInfo;
            gert::TilingContextPara ctx("GenericBlockSparseAttention", inputs,
                MakeOutputs(ge::DT_FLOAT16, returnLse),
                MakeAttrs(1, 0, 1, returnLse, 0.0f, -1, -1, 1, "TND", paged ? "PA_BBND" : "TND", 1),
                &compileInfo, "Ascend950", 56, 262144);
            ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, baseKey + lseOffset + residualOffset);
        }
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_block_mode_rejects_arch35_quant)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara ctx("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
        MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 5, 1, 0, 0.0f, -1, -1, 1, "TND", "PA_BBND", 1),
        &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, residual_none_and_consistent_topk_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, true), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kFp16Tiling910B);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_mask_type)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(0, 0, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_block_shape_x)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    std::vector<int64_t> badBlockShape = {128, 128};
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, badBlockShape), &compileInfo, "Ascend910B",
        40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_win_left)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0, 0.0f, 127, -1),
                                              &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_bf16_halfsm_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_BF16),
                                              MakeOutputs(ge::DT_BF16, false), MakeAttrs(1, 0, 1, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_on_910b)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, quant_type_dtype_mismatch)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 5, 0, 0), &compileInfo,
                                              "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_metadata)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_cu_seq_lengths_q)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[11] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_seqused_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[14] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, pa_rejects_cu_seq_lengths_kv)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[12] = {{{kBatch + 1}, {kBatch + 1}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, scale_value_zero_defaults_to_rsqrt_d)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16, false),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "PA_BBND", 0, false, kBlockShape, 0.0f), &compileInfo,
        "Ascend910B", 40, 196608);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16Tiling910B));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    // scaleValue follows the 10 uint32 fields in the POD tiling layout.
    constexpr size_t kScaleValueOffset = 10U * sizeof(uint32_t);
    ASSERT_GT(tilingInfo.tilingDataSize, kScaleValueOffset + sizeof(float));
    float scaleValue = 0.0f;
    std::memcpy(&scaleValue, tilingInfo.tilingData.get() + kScaleValueOffset, sizeof(float));
    EXPECT_FLOAT_EQ(scaleValue, kScale);
}

TEST_F(generic_block_sparse_attention_tiling_ut, high_precision_softmax_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT16),
                                              MakeOutputs(ge::DT_FLOAT16, false), MakeAttrs(1, 0, 0, 0), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 9250000071001002ULL);
}

TEST_F(generic_block_sparse_attention_tiling_ut, missing_block_table)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[15] = {{}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, invalid_metadata_size)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    inputs[5] = {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", inputs, MakeOutputs(ge::DT_FLOAT16, false),
                                              MakeAttrs(1, 0, 0, 0), &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_fp8_with_lse_950)
{
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GenericBlockSparseAttention", MakeInputs(ge::DT_FLOAT8_E4M3FN),
                                              MakeOutputs(ge::DT_FLOAT16, true), MakeAttrs(1, 5, 1, 1), &compileInfo,
                                              "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_tiling_ut, supported_dense_head_dim_910b)
{
    constexpr size_t kEmbeddingSizeOffset = 3U * sizeof(uint32_t);
    constexpr size_t kFdStaticEnabledOffset = 120U;
    for (int64_t headDim : {1, 17, 64, 128, 240, 256}) {
        SCOPED_TRACE(headDim);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim),
            MakeOutputs(ge::DT_FLOAT16, false, headDim),
            MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16TndTndTiling910B));
        ASSERT_NE(tilingInfo.tilingData, nullptr);
        ASSERT_GT(tilingInfo.tilingDataSize, kFdStaticEnabledOffset + sizeof(uint32_t));
        EXPECT_EQ(ReadTilingU32(tilingInfo, kEmbeddingSizeOffset), static_cast<uint32_t>(headDim));
        EXPECT_EQ(ReadTilingU32(tilingInfo, kFdStaticEnabledOffset), 0U);
    }
}

TEST_F(generic_block_sparse_attention_tiling_ut, fd_enabled_dense_head_dim_17_910b)
{
    constexpr int64_t headDim = 17;
    constexpr int64_t topK = 12;
    constexpr size_t kEmbeddingSizeOffset = 3U * sizeof(uint32_t);
    constexpr size_t kFdStaticEnabledOffset = 120U;
    GenericBlockSparseAttentionCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim, topK),
        MakeOutputs(ge::DT_FLOAT16, false, headDim),
        MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(kFp16TndTndTiling910B));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_GT(tilingInfo.tilingDataSize, kFdStaticEnabledOffset + sizeof(uint32_t));
    EXPECT_EQ(ReadTilingU32(tilingInfo, kEmbeddingSizeOffset), static_cast<uint32_t>(headDim));
    EXPECT_EQ(ReadTilingU32(tilingInfo, kFdStaticEnabledOffset), 1U);
}

TEST_F(generic_block_sparse_attention_tiling_ut, unsupported_dense_head_dim_910b)
{
    for (int64_t headDim : {int64_t(0), int64_t(257)}) {
        SCOPED_TRACE(headDim);
        GenericBlockSparseAttentionCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "GenericBlockSparseAttention", MakeDenseInputs(ge::DT_FLOAT16, headDim),
            MakeOutputs(ge::DT_FLOAT16, false, headDim),
            MakeAttrs(1, 0, 0, 0, 0.0f, -1, -1, 4, "TND", "TND"), &compileInfo, "Ascend910B", 40, 196608);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    }
}
