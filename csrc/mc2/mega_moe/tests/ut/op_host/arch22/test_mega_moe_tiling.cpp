/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {
class MegaMoeArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(MegaMoeArch22TilingTest, Test0)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kNHalf = 14336;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kNHalf, kHiddenSize}, {kNHalf, kHiddenSize}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"dispatch_quant_mode",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_mode=2
         {"dispatch_quant_out_type",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_out_type=INT8
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_token_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank},
        {1, 1}, &compileInfo, "Ascend910B", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
}

TEST_F(MegaMoeArch22TilingTest, ActivationParamValidationWithLargeIntermediateHidden)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 20;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kBs = 32;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kIntermediateHidden = kN / 2;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back(
            {{{kIntermediateHidden, kHiddenSize}, {kIntermediateHidden, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    }

    struct ActivationCase {
        std::string activation;
        std::vector<float> params;
        ge::graphStatus expectedStatus;
    };
    const std::vector<ActivationCase> activationCases = {
        {"swiglu", {}, ge::GRAPH_SUCCESS},
        {"swiglustep", {}, ge::GRAPH_SUCCESS},
        {"swigluoai", {}, ge::GRAPH_SUCCESS},
        {"situglu", {}, ge::GRAPH_SUCCESS},
        {"swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS},
        {"swiglu", {0.0f}, ge::GRAPH_SUCCESS},
        {"swiglustep", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS},
        {"swigluoai", {std::numeric_limits<float>::max(), 0.0f, 0.0f}, ge::GRAPH_SUCCESS},
        {"swigluoai", {std::numeric_limits<float>::max(), -1.702f, -1.0f}, ge::GRAPH_SUCCESS},
        {"situglu", {-2.0f}, ge::GRAPH_SUCCESS},
        {"situglu", {-2.0f, -2.0f}, ge::GRAPH_SUCCESS},
        {"swiglu", {-1.0f}, ge::GRAPH_FAILED},
        {"swigluoai",
         {std::numeric_limits<float>::max(), std::numeric_limits<float>::quiet_NaN(), 1.0f},
         ge::GRAPH_FAILED},
        {"swigluoai",
         {std::numeric_limits<float>::max(), 1.702f, std::numeric_limits<float>::infinity()},
         ge::GRAPH_FAILED},
        {"situglu", {0.0f}, ge::GRAPH_FAILED},
        {"situglu", {1.0f, 0.0f}, ge::GRAPH_FAILED},
        {"situglu", {std::numeric_limits<float>::quiet_NaN()}, ge::GRAPH_FAILED},
        {"situglu", {1.0f, std::numeric_limits<float>::infinity()}, ge::GRAPH_FAILED},
    };
    for (const auto &activationCase : activationCases) {
        gert::TilingContextPara tilingContextPara(
            "MegaMoe", inputs,
            {
                {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
             {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2LL * 1024 * 1024 * 1024)},
             {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
             {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"dispatch_quant_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>(activationCase.activation)},
             {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(activationCase.params)},
             {"activation_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, 0, 0, 0, 0, 0, 0}, {1, 1}, &compileInfo, "Ascend910B",
            coreNum, ubSize);

        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, activationCase.expectedStatus, UINT64_MAX);
    }
}

} // namespace MegaMoeUT
