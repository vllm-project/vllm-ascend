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
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/recurrent_gated_delta_rule_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;
using namespace RecurrentGatedDeltaRule;

class RecurrentGatedDeltaRuleTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RecurrentGatedDeltaRuleTilingTest TearDown" << std::endl;
    }
};

TEST_F(RecurrentGatedDeltaRuleTilingTest, Test0)
{
    optiling::RecurrentGatedDeltaRuleCompileInfo compileinfo = {48, 196608}; // aivNum銆乽bSize

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

    gert::TilingContextPara tilingContextPara("RecurrentGatedDeltaRule", 
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

        },
        &compileinfo
    );

    int64_t expectTilingKey = 0UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(RecurrentGatedDeltaRuleTilingTest, SelectsDoubleOutputBufferProfile)
{
    optiling::RecurrentGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    int t = 128;
    int nk = 4;
    int dk = 128;
    int nv = 8;
    int dv = 128;
    int sBlockNum = 128;
    int b = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{sBlockNum, nv, dv, dk}, {sBlockNum, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{b}, {b}};
    gert::StorageShape ssmStateIndicesShape = {{t}, {t}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::TilingContextPara tilingContextPara("RecurrentGatedDeltaRule",
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
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(RecurrentGatedDeltaRuleTilingData));

    auto *tilingData = reinterpret_cast<const RecurrentGatedDeltaRuleTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->stateOutBufferNum, 2U);
    EXPECT_EQ(tilingData->attnOutBufferNum, 2U);
    EXPECT_EQ(tilingData->vStep, 64U);
}
