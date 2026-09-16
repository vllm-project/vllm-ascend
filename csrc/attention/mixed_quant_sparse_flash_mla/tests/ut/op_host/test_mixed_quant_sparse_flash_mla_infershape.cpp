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

class MixedQuantSparseFlashMlaProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MixedQuantSparseFlashMlaProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MixedQuantSparseFlashMlaProto TearDown" << std::endl;
    }
};

// TND q + PA_BBND ori_kv, return_softmax_lse=true
TEST_F(MixedQuantSparseFlashMlaProto, MixedQuantSparseFlashMla_infershape_tnd_pa_lse)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MixedQuantSparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},                  // q            input0
            {{{128, 128, 1, 608}, {128, 128, 1, 608}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // ori_kv input1
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_kv       input2 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_sparse_indices input3 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_sparse_indices input4 (optional)
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},   // ori_block_table input5
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_block_table input6 (optional)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true},     // cu_seqlens_q input7
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_ori_kv input8 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_cmp_kv input9 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_q    input10 (optional)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true},     // seqused_ori_kv input11
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_cmp_kv input12 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_residual_kv input13 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_topk_length input14 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_topk_length input15 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // sinks        input16 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}  // metadata     input17 (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // attn_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // softmax_lse
        },
        {{"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 64, 512}, {1, 512, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BSND q + TND ori_kv, return_softmax_lse=false
TEST_F(MixedQuantSparseFlashMlaProto, MixedQuantSparseFlashMla_infershape_bsnd_tnd_no_lse)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MixedQuantSparseFlashMla",
        {
            {{{2, 128, 64, 512}, {2, 128, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},  // q            input0
            {{{128, 1, 608}, {128, 1, 608}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // ori_kv       input1 (TND)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true},                   // cmp_kv       input2 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_sparse_indices input3 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_sparse_indices input4 (optional)
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND},   // ori_block_table input5
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_block_table input6 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_q input7 (optional)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},     // cu_seqlens_ori_kv input8
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_cmp_kv input9 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_q    input10 (optional)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},     // seqused_ori_kv input11
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_cmp_kv input12 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_residual_kv input13 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_topk_length input14 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_topk_length input15 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // sinks        input16 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}  // metadata     input17 (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // attn_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // softmax_lse
        },
        {{"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 128, 64, 512}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// kv head num must be greater than 0: ori_kv with dim2=0 should fail
TEST_F(MixedQuantSparseFlashMlaProto, MixedQuantSparseFlashMla_infershape_kv_headnum_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MixedQuantSparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},                  // q            input0
            {{{128, 128, 0, 608}, {128, 128, 0, 608}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // ori_kv (head num = 0)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_kv       input2 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_sparse_indices input3 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_sparse_indices input4 (optional)
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},   // ori_block_table input5
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_block_table input6 (optional)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true},     // cu_seqlens_q input7
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_ori_kv input8 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cu_seqlens_cmp_kv input9 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_q    input10 (optional)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true},     // seqused_ori_kv input11
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // seqused_cmp_kv input12 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_residual_kv input13 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // ori_topk_length input14 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // cmp_topk_length input15 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}, // sinks        input16 (optional)
            {{{0}, {0}}, ge::DT_UNDEFINED, ge::FORMAT_ND, true}  // metadata     input17 (optional)
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, // attn_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // softmax_lse
        },
        {{"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// infer dataType
TEST_F(MixedQuantSparseFlashMlaProto, MixedQuantSparseFlashMla_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MixedQuantSparseFlashMla")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType inputQ = ge::DT_BF16;
        ge::DataType inputKv = ge::DT_FLOAT8_E4M3FN;
        ge::DataType inputFp32 = ge::DT_FLOAT;
        ge::DataType inputI32 = ge::DT_INT32;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(18, 2)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&inputQ, &inputKv, &inputKv, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32,
                                 &inputI32, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32,
                                 &inputFp32, &inputI32})
                .NodeAttrs({{"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
                            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
                            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
    }
}
