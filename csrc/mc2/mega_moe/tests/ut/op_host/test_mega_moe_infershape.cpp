/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MegaMoeUT {

class MegaMoeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(MegaMoeInferShapeTest, InferShape_H4096_BS128)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 8}, {}};
    gert::StorageShape topkWeightsShape = {{128, 8}, {}};
    gert::StorageShape weight1Shape = {{4, 2048, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 1024}, {}};
    gert::StorageShape weightScales1Shape = {{4, 2048, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 16, 2}, {}};
    gert::StorageShape xActiveMaskShape = {{}, {}};
    gert::StorageShape scalesShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MegaMoe",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_BF16, ge::FORMAT_ND},
         {topkIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {weight1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weight2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weightScales1Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {weightScales2Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {xActiveMaskShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{128, 4096}, {4}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

TEST_F(MegaMoeInferShapeTest, InferShape_H5120_BS256)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{256, 5120}, {}};
    gert::StorageShape topkIdsShape = {{256, 6}, {}};
    gert::StorageShape topkWeightsShape = {{256, 6}, {}};
    gert::StorageShape weight1Shape = {{8, 3072, 5120}, {}};
    gert::StorageShape weight2Shape = {{8, 5120, 1536}, {}};
    gert::StorageShape weightScales1Shape = {{8, 3072, 80, 2}, {}};
    gert::StorageShape weightScales2Shape = {{8, 5120, 24, 2}, {}};
    gert::StorageShape xActiveMaskShape = {{}, {}};
    gert::StorageShape scalesShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MegaMoe",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_BF16, ge::FORMAT_ND},
         {topkIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {topkWeightsShape, ge::DT_BF16, ge::FORMAT_ND},
         {weight1Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {weight2Shape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
         {weightScales1Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {weightScales2Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {xActiveMaskShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(23)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{256, 5120}, {8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

TEST_F(MegaMoeInferShapeTest, InferShape_H7168_BS512)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{512, 7168}, {}};
    gert::StorageShape topkIdsShape = {{512, 8}, {}};
    gert::StorageShape topkWeightsShape = {{512, 8}, {}};
    gert::StorageShape weight1Shape = {{2, 4096, 7168}, {}};
    gert::StorageShape weight2Shape = {{2, 7168, 2048}, {}};
    gert::StorageShape weightScales1Shape = {{2, 4096, 112, 2}, {}};
    gert::StorageShape weightScales2Shape = {{2, 7168, 32, 2}, {}};
    gert::StorageShape xActiveMaskShape = {{}, {}};
    gert::StorageShape scalesShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MegaMoe",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_BF16, ge::FORMAT_ND},
         {topkIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {weight1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weight2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weightScales1Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {weightScales2Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {xActiveMaskShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{512, 7168}, {2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

TEST_F(MegaMoeInferShapeTest, InferShape_InvalidEpWorldSize)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 8}, {}};
    gert::StorageShape topkWeightsShape = {{128, 8}, {}};
    gert::StorageShape weight1Shape = {{4, 2048, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 1024}, {}};
    gert::StorageShape weightScales1Shape = {{4, 2048, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 16, 2}, {}};
    gert::StorageShape xActiveMaskShape = {{}, {}};
    gert::StorageShape scalesShape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MegaMoe",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_BF16, ge::FORMAT_ND},
         {topkIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {weight1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weight2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
         {weightScales1Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {weightScales2Shape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
         {xActiveMaskShape, ge::DT_INT8, ge::FORMAT_ND},
         {scalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_BF16, ge::FORMAT_ND}, {{}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeInferShapeTest, InferDatatype_BF16_FP8E4M3FN)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;
    ge::DataType topkIdsType = ge::DT_INT32;
    ge::DataType topkWeightsType = ge::DT_FLOAT;
    ge::DataType weight1Type = ge::DT_FLOAT8_E4M3FN;
    ge::DataType weight2Type = ge::DT_FLOAT8_E4M3FN;
    ge::DataType weightScales1Type = ge::DT_FLOAT8_E8M0;
    ge::DataType weightScales2Type = ge::DT_FLOAT8_E8M0;
    ge::DataType xActiveMaskType = ge::DT_INT8;
    ge::DataType scalesType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_UNDEFINED;
    ge::DataType expertTokenNumsType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(10, 2)
            .NodeAttrs({{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
                        {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                        {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
                        {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
                        {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(
                                                  {std::numeric_limits<float>::max()})},
                        {"activation_out_dtype",
                         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
                        {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, topkIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, topkWeightsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, weight1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, weight2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, weightScales1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, weightScales2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, xActiveMaskType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, scalesType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &topkIdsType, &topkWeightsType, &weight1Type, &weight2Type,
                             &weightScales1Type, &weightScales2Type, &xActiveMaskType, &scalesType})
            .OutputDataTypes({&outputType, &expertTokenNumsType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(MegaMoeInferShapeTest, InferDatatype_BF16_FP8E5M2)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;
    ge::DataType topkIdsType = ge::DT_INT32;
    ge::DataType topkWeightsType = ge::DT_BF16;
    ge::DataType weight1Type = ge::DT_FLOAT8_E5M2;
    ge::DataType weight2Type = ge::DT_FLOAT8_E5M2;
    ge::DataType weightScales1Type = ge::DT_FLOAT8_E8M0;
    ge::DataType weightScales2Type = ge::DT_FLOAT8_E8M0;
    ge::DataType xActiveMaskType = ge::DT_INT8;
    ge::DataType scalesType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_UNDEFINED;
    ge::DataType expertTokenNumsType = ge::DT_UNDEFINED;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(10, 2)
            .NodeAttrs({{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2097152)},
                        {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                        {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(23)},
                        {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
                        {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(
                                                  {std::numeric_limits<float>::max()})},
                        {"activation_out_dtype",
                         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
                        {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                        {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, topkIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, topkWeightsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, weight1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, weight2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, weightScales1Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, weightScales2Type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, xActiveMaskType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, scalesType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &topkIdsType, &topkWeightsType, &weight1Type, &weight2Type,
                             &weightScales1Type, &weightScales2Type, &xActiveMaskType, &scalesType})
            .OutputDataTypes({&outputType, &expertTokenNumsType})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MegaMoe")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_INT32);
}

} // namespace MegaMoeUT
