/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class QuantLightningIndexerV2Proto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2Proto TearDown" << std::endl;
    }
};

// BSND/BSND, return_value=1, topk=128
TEST_F(QuantLightningIndexerV2Proto, QuantLightningIndexerV2_infershape_bsnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexerV2",
        // 输入Tensor (13个)
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_k    input8 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                   // metadata     input12
        },
        // 输出Tensor
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        // 属性
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 1, 128}, {1, 8, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND/TND, return_value=0, topk=2048
TEST_F(QuantLightningIndexerV2Proto, QuantLightningIndexerV2_infershape_tnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexerV2",
        {
            {{{64, 32, 128}, {64, 32, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // q            input0
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},   // k            input1
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // w            input2
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // q_descale    input3
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // k_descale    input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cu_seqlens_q input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cu_seqlens_k input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_q    input7 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_k    input8 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cmp_residual_k input9 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // block_table  input10 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}, // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}  // metadata     input12
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 1, 2048}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND/PA_BBND, return_value=1, topk=2048
TEST_F(QuantLightningIndexerV2Proto, QuantLightningIndexerV2_infershape_tnd_pa)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexerV2",
        {
            {{{64, 32, 128}, {64, 32, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // q            input0
            {{{1, 16, 1, 128}, {1, 16, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // k (PA)  input1
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // w            input2
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // q_descale    input3
            {{{1, 16, 1}, {1, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // k_descale    input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                           // cu_seqlens_q input5
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                           // cu_seqlens_k input6 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                           // seqused_q    input7 (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true},                           // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}, // cmp_residual_k input9 (optional)
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}, // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}  // metadata     input12
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 1, 2048}, {64, 1, 2048}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// invalid layout_q should fail
TEST_F(QuantLightningIndexerV2Proto, QuantLightningIndexerV2_infershape_layout_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantLightningIndexerV2",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_k    input8 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // block_table  input10 (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                   // metadata     input12
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// infer dataType
TEST_F(QuantLightningIndexerV2Proto, QuantLightningIndexerV2_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QuantLightningIndexerV2")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType inputQ = ge::DT_INT8;
        ge::DataType inputK = ge::DT_INT8;
        ge::DataType inputW = ge::DT_FLOAT16;
        ge::DataType inputScale = ge::DT_FLOAT16;
        ge::DataType inputI32 = ge::DT_INT32;
        ge::DataType outputRef0 = ge::DT_INT32;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(13, 2)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&inputQ, &inputK, &inputW, &inputScale, &inputScale, &inputI32, &inputI32, &inputI32,
                                 &inputI32, &inputI32, &inputI32, &inputI32, &inputI32})
                .NodeAttrs({{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                            {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef0);
    }
}
