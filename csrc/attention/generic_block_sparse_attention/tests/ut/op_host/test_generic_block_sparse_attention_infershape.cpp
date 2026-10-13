/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class generic_block_sparse_attention_infershape_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "generic_block_sparse_attention_infershape_ut SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "generic_block_sparse_attention_infershape_ut TearDown" << endl;
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
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr float kScale = 1.0f / sqrt(static_cast<float>(kD));

std::vector<int64_t> kBlockShape = {1, kBlockSize};

std::vector<gert::InfershapeContextPara::TensorDescription> MakeInputs()
{
    return {
        {{{kT, kN1, kD}, {kT, kN1, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kN2, kD}, {kMaxBlocks, kBlockSize, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{kMaxBlocks, kBlockSize, kN2, kD}, {kMaxBlocks, kBlockSize, kN2, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{kN2, kTotalQBlocks, kTopK}, {kN2, kTotalQBlocks, kTopK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{kN2, kTotalQBlocks}, {kN2, kTotalQBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
        {{{kBatch, kMaxBlocks}, {kBatch, kMaxBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(int64_t returnLse, const std::string &layoutQ = "TND",
                                                           const std::string &layoutKv = "PA_BBND")
{
    return {
        {"block_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(kBlockShape)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutKv)},
        {"layout_sparse_pattern", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(kScale)},
        {"mask_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"quant_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        {"softmax_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(returnLse)},
        {"residual_block_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"is_consistent_topk", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}
} // namespace

TEST_F(generic_block_sparse_attention_infershape_ut, tnd_paged_no_lse)
{
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", MakeInputs(),
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {kT, kN1, kD},
        {0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(generic_block_sparse_attention_infershape_ut, tnd_paged_with_lse)
{
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", MakeInputs(),
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {kT, kN1, kD},
        {kT, kN1, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(generic_block_sparse_attention_infershape_ut, unsupported_layout)
{
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", MakeInputs(),
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(0, "TND", "TND"));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_infershape_ut, key_value_d_mismatch)
{
    auto inputs = MakeInputs();
    inputs[1] = {{{kMaxBlocks, kBlockSize, kN2, 64}, {kMaxBlocks, kBlockSize, kN2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", inputs,
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(0));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_infershape_ut, query_not_3d)
{
    auto inputs = MakeInputs();
    inputs[0] = {{{kBatch, kS1, kN1, kD}, {kBatch, kS1, kN1, kD}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", inputs,
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(0));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(generic_block_sparse_attention_infershape_ut, unknown_rank_query)
{
    auto inputs = MakeInputs();
    inputs[0] = {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::InfershapeContextPara infershapeContextPara("GenericBlockSparseAttention", inputs,
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      MakeAttrs(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
