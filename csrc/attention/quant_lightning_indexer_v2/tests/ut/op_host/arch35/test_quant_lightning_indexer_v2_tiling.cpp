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

// DAV_3510 (Ascend950) tiling cases for QuantLightningIndexerV2
class QuantLightningIndexerV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch35 TearDown" << std::endl;
    }
};

namespace {
// Base of a valid Ascend950 TND/PA_BBND fp8 case
qliv2_ut::CaseParam Make950TndPaFp8()
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {2, 16, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.kScaleShape = {2, 16, 1};
    p.outShape = {78, 1, 2048};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.layoutK = "PA_BBND";
    p.quantMode = 1;
    p.maxSeqlenQ = 64;
    p.maskMode = 3;
    return p;
}
} // namespace

// TND/PA_BBND fp8 success on Ascend950: quant_mode=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_tnd_pa_success)
{
    qliv2_ut::RunTilingCase(Make950TndPaFp8(), ge::GRAPH_SUCCESS);
}

// TND/PA_BBND fp8 success with cmp_residual_k and output_idx_offset on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_cmp_residual_success)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {2};
    p.idxOffset = {78, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND/BSND mxfp8 success on Ascend950: quant_mode=3, e8m0 scale
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp8_bsnd_bsnd_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1, 2, 2};
    p.qScaleShape = {2, 39, 64, 2, 2};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.layoutK = "BSND";
    p.quantMode = 3;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND/TND hifloat8 success on Ascend950: quant_mode=4, return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_hif8_tnd_tnd_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {128, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {1};
    p.kScaleShape = {1};
    p.outShape = {78, 1, 2048};
    p.valuesShape = {78, 1, 2048};
    p.qType = ge::DT_HIFLOAT8;
    p.kType = ge::DT_HIFLOAT8;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.cuSeqK = {3};
    p.layoutQ = "TND";
    p.layoutK = "TND";
    p.quantMode = 4;
    p.maxSeqlenQ = 64;
    p.returnValue = 1;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND/PA_BBND int8 success on Ascend950: quant_mode=2, return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_int8_pa_rv_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 2048};
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND/PA_BBND mxfp4 success on Ascend950: quant_mode=5
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp4_tnd_pa_success)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qType = ge::DT_FLOAT4_E2M1;
    p.kType = ge::DT_FLOAT4_E2M1;
    p.qScaleShape = {78, 64, 2, 2};
    p.kScaleShape = {2, 16, 1, 2, 2};
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.quantMode = 5;
    p.maskMode = 0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// layout_k only supports PA_BBND, BSND or TND on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_layout_k_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "XXX";
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// outside of PA, layout_q and layout_k must be the same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_layout_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "BSND";
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 8192 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_topk_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.topk = 10000;
    p.outShape = {78, 1, 10000};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio must > 0 and <= 128 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_ratio_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 200;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode only supports 1-5 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_quant_mode_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.quantMode = 6;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// return_value only supports 0 or 1 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_return_value_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.returnValue = 2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// max_seqlen_q must >= -1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_max_seqlen_q_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.maxSeqlenQ = -2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q must match quant_mode: int8 with quant_mode=1 should fail
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_q_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qType = ge::DT_INT8;
    p.kType = ge::DT_INT8;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale must match quant_mode: e8m0 with quant_mode=1 should fail
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_scale_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// when q is int8 (quant_mode=2), dtype of w must be float16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_int8_w_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.quantMode = 2;
    p.wType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// when q is not int8 (quant_mode=1), dtype of w must be float
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_w_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.wType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_qk_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kType = ge::DT_INT8;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale and k_descale must be same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_scale_dtype_not_equal_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kScaleType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_indices must be int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_out_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.outType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_values must be bfloat16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_values_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.valuesType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND requires block_table
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_pa_block_table_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND must not provide cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_pa_cu_seqlens_k_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cuSeqK = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND k requires cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_tnd_k_cu_seqlens_k_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.blockTable = {};
    p.sequsedK = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// BSND k must not provide cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_k_cu_seqlens_k_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "BSND";
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.blockTable = {};
    p.sequsedK = {};
    p.cuSeqK = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// non-PA layout must not provide block_table
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_block_table_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.sequsedK = {};
    p.cuSeqK = {3};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio != 1 and mask_mode != 0 require cmp_residual_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_residual_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND q requires cu_seqlens_q
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_tnd_cu_seqlens_q_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cuSeqQ = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// metadata shape size must be 1024
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_metadata_size_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.metadata = {512};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// mxfp8 scale dim num must be q dim num + 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp8_scale_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.qScaleShape = {2, 39, 64};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.layoutK = "BSND";
    p.quantMode = 3;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// hifloat8 scale dim num must be 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_hif8_scale_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {128, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.kScaleShape = {1};
    p.outShape = {78, 1, 2048};
    p.qType = ge::DT_HIFLOAT8;
    p.kType = ge::DT_HIFLOAT8;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.cuSeqK = {3};
    p.layoutQ = "TND";
    p.layoutK = "TND";
    p.quantMode = 4;
    p.maxSeqlenQ = 64;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// fp8 scale dim num must be q dim num - 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_scale_dim_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qScaleShape = {78};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head num of k only supports 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_k_headnum_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kShape = {2, 16, 2, 128};
    p.kScaleShape = {2, 16, 2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// gSize must <= 64 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_gsize_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qShape = {2, 39, 128, 128};
    p.wShape = {2, 39, 128};
    p.qScaleShape = {2, 39, 128};
    p.outShape = {2, 39, 1, 2048};
    p.layoutQ = "BSND";
    p.cuSeqQ = {};
    p.maskMode = 0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// block_size of k must be a multiple of 16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_block_size_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kShape = {2, 17, 1, 128};
    p.kScaleShape = {2, 17, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of cu_seqlens_q only supports int32 (TND/TND so that cu_seqlens_k desc is valid for error logging)
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cu_seqlens_q_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.blockTable = {};
    p.sequsedK = {};
    p.cuSeqK = {3};
    p.cuSeqQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of seqused_q only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_seqused_q_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 2048};
    p.sequsedQ = {2};
    p.sequsedQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of cmp_residual_k only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_residual_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {2};
    p.cmpResidualType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of output_idx_offset only supports int32 (seqused_q provided so its desc is valid for error logging)
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_idx_offset_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.idxOffset = {2, 39, 64};
    p.idxOffsetType = ge::DT_INT64;
    p.sequsedQ = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of block_table only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_block_table_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.blockTableType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of seqused_k only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_seqused_k_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.sequsedKType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// last dim of sparse_values must be same as topk when return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_values_topk_mismatch_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 1024};
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head dim of q only supports 128
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_head_dim_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qShape = {78, 64, 127};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}
