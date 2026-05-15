/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_finalize_routing.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RecurrentGatedDeltaRuleTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTest Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTest Proto TearDown" << std::endl;
    }
};

TEST_F(RecurrentGatedDeltaRuleTest, Test0)
{
    int t = 128;
    int nk = 4;
    int dk = 8;
    int nv = 128;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t,nv}, {t,nv}};
    gert::StorageShape stateShape = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{b}, {b}};
    gert::StorageShape ssmStateIndicesShape = {{t}, {t}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};
    gert::StorageShape gkShape = {{}, {}};
    gert::StorageShape accTokensShape = {{b}, {b}};

    gert::InfershapeContextPara infershapeContextPara("RecurrentGatedDeltaRule", 
        {
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {keyShape, ge::DT_BF16, ge::FORMAT_ND},
            {valueShape, ge::DT_BF16, ge::FORMAT_ND},
            {betaShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_BF16, ge::FORMAT_ND},
            {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
            {ssmStateIndicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Math::AnyValue::CreateFrom<float>(1.0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{t, nv, dv}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
