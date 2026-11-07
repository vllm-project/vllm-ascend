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

// return_value=1 is supported on Ascend910B (DCP 跨 rank 全局 TopK 需要 score)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_return_value_success)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 1;
    p.valuesShape = {2, 39, 1, 2048};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// The private local-candidate mode exports the native FP32 comparator and
// consumes per-query visible lengths through the existing offset input slot.
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_local_candidates_success)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 2;
    p.idxOffset = {2, 39, 1};
    p.valuesShape = p.outShape;
    p.valuesType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_local_candidates_missing_lengths)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 2;
    p.valuesShape = p.outShape;
    p.valuesType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_local_candidates_reject_bf16_scores)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 2;
    p.idxOffset = {2, 39, 1};
    p.valuesShape = p.outShape;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_local_candidates_reject_causal_mask)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 2;
    p.maskMode = 3;
    p.idxOffset = {2, 39, 1};
    p.valuesShape = p.outShape;
    p.valuesType = ge::DT_FLOAT;
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

// ======================= DCP (Decode Context Parallel, 解码上下文并行) =======================
static qliv2_ut::CaseParam MakeDcpCase()
{
    qliv2_ut::CaseParam p;
    p.layoutQ = "TND";
    p.layoutK = "PA_BBND";
    p.qShape = {4, 64, 128};       // TND: (T1, N1, D)
    p.wShape = {4, 64};
    p.qScaleShape = {4, 64};
    p.kShape = {1, 16, 1, 128};    // PA: (block_num, block_size, N2, D)
    p.kScaleShape = {1, 16, 1};
    p.blockTable = {1, 1};
    p.cuSeqQ = {0, 4};
    p.sequsedK = {1};
    p.outShape = {4, 1, 2048};
    p.valuesShape = {4, 1, 2048};
    p.maskMode = 3;
    p.returnValue = 1;
    p.causalLens = {1};
    p.enableDcp = true;
    p.cpWorld = 2;
    p.cpRank = 1;
    p.cpInterleave = 16;
    return p;
}

// DCP 合法组合: TND + PA_BBND, return_value=1, mask_mode=3
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_success)
{
    auto p = MakeDcpCase();
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// DCP 无 causal (mask_mode=0) 也合法
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_mask_mode0_success)
{
    auto p = MakeDcpCase();
    p.maskMode = 0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// DCP 要求 layout_q=TND
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_layout_failed)
{
    auto p = MakeDcpCase();
    p.layoutQ = "BSND";
    p.qShape = {2, 2, 64, 128};
    p.wShape = {2, 2, 64};
    p.qScaleShape = {2, 2, 64};
    p.outShape = {2, 2, 1, 2048};
    p.valuesShape = {2, 2, 1, 2048};
    p.cuSeqQ = {};
    p.causalLens = {1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// DCP 要求 return_value=1
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_return_value_failed)
{
    auto p = MakeDcpCase();
    p.returnValue = 0;
    p.valuesShape = {0};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// DCP 要求 causal_seqlens_kv_global 必传
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_causal_missing_failed)
{
    auto p = MakeDcpCase();
    p.causalLens = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// DCP 要求 cp_rank 在 [0, cp_world)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_rank_failed)
{
    auto p = MakeDcpCase();
    p.cpRank = 2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// DCP 与 cmp_ratio>1 互斥
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_cmp_ratio_failed)
{
    auto p = MakeDcpCase();
    p.cmpRatio = 4;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// DCP 禁止 output_idx_offset (避免与框架 local->global 映射重复偏移)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_idx_offset_failed)
{
    auto p = MakeDcpCase();
    p.idxOffset = {4, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// enable_dcp=false 时 cp_* 必须为默认值
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_dcp_off_nondefault_failed)
{
    auto p = MakeDcpCase();
    p.enableDcp = false;
    p.causalLens = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}
