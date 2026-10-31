/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../test_quant_lightning_indexer_v2_utils.h"

// DAV_2201 (Ascend910B) tiling cases for QuantLightningIndexerV2
class QuantLightningIndexerV2TilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch22 TearDown" << std::endl;
    }
};

// BSND/PA_BBND int8 success on Ascend910B: quant_mode=2, topk=2048, mask_mode=0
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_int8_pa_success)
{
    qliv2_ut::CaseParam p;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND/PA_BBND int8 success with cmp_residual_k and output_idx_offset on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_cmp_residual_success)
{
    qliv2_ut::CaseParam p;
    p.cmpRatio = 4;
    p.maskMode = 3;
    p.cmpResidual = {2};
    p.idxOffset = {2, 39, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// layout_k only supports PA_BBND on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_layout_k_failed)
{
    qliv2_ut::CaseParam p;
    p.layoutK = "BSND";
    p.blockTable = {};
    p.sequsedK = {};
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode only supports 2 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_quant_mode_failed)
{
    qliv2_ut::CaseParam p;
    p.quantMode = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// return_value only supports false on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_return_value_failed)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 2048 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_topk_over_limit_failed)
{
    qliv2_ut::CaseParam p;
    p.topk = 4096;
    p.outShape = {2, 39, 1, 4096};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio must be a power of 2 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_cmp_ratio_not_pow2_failed)
{
    qliv2_ut::CaseParam p;
    p.cmpRatio = 3;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be int8 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_q_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.qType = ge::DT_FLOAT16;
    p.kType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale and k_descale must be float16 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_scale_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of w must be float16 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_w_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.wType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// gSize (N1/N2) must equal 64 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_gsize_failed)
{
    qliv2_ut::CaseParam p;
    p.qShape = {2, 39, 128, 128};
    p.wShape = {2, 39, 128};
    p.qScaleShape = {2, 39, 128};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// mask_mode only supports 0 or 3
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_mask_mode_failed)
{
    qliv2_ut::CaseParam p;
    p.maskMode = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// metadata is required
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_metadata_missing_failed)
{
    qliv2_ut::CaseParam p;
    p.metadata = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be same: q=int8, k=fp8 should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_qk_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_values must be bfloat16
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_values_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.valuesType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head dim of q only supports 128
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_head_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.qShape = {2, 39, 64, 127};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// block_size of k must be a multiple of 16
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_block_size_failed)
{
    qliv2_ut::CaseParam p;
    p.kShape = {2, 17, 1, 128};
    p.kScaleShape = {2, 17, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head num of k only supports 1
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_k_headnum_failed)
{
    qliv2_ut::CaseParam p;
    p.kShape = {2, 16, 2, 128};
    p.kScaleShape = {2, 16, 2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// unsupported npu arch (Ascend310P) should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_tiling_unsupported_arch_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend310P";
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}
