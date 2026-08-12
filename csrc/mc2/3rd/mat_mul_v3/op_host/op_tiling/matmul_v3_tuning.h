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
 * \file matmul_v3_tuning.h
 * \brief
 */
#ifndef MATMUL_V3_TUNING_H__
#define MATMUL_V3_TUNING_H__

#include "register/tuning_tiling_registry.h"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
#pragma pack(push)
#pragma pack(1)
struct Mc2MatMulV3InputArgs { // GemmInputArgs
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t batch_a1;
    int64_t batch_a2;
    int64_t batch_a3;
    int64_t batch_a4;
    int64_t batch_b1;
    int64_t batch_b2;
    int64_t batch_b3;
    int64_t batch_b4;
    float l1_fused_num;
    float aub_double_num;
    float bub_double_num;
    float fused_double_operand_num;
    ge::DataType a_dtype;
    ge::DataType b_dtype;
    ge::DataType out_dtype;
    ge::Format a_format;
    ge::Format b_format;
    ge::Format out_format;
    bool trans_a_flag;
    bool trans_b_flag;
    bool bias_flag;
    bool reserved_bool;
    bool m_align_flag;
    bool k_align_flag;
    bool n_align_flag;
    uint64_t reserved_params1; // 保留字段，变量名的修改不影响hash结果
    uint64_t reserved_params2;
    uint64_t reserved_params3;
    uint64_t reserved_params4;
    uint64_t reserved_params5;
    uint64_t reserved_params6;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Mc2MatMulV3TunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseK);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, depthA1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, depthB1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKa);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKb);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbL0A);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbL0B);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbL0C);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, l2MTileCnt);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, l2NTileCnt);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, l2MTileBlock);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, l2NTileBlock);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, l2IterateOrder);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, tilingEnable);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Mc2MatMulV3TunnerTiling, FIELD(Mc2MatMulV3TunnerTiling, usedCoreNum),
               FIELD(Mc2MatMulV3TunnerTiling, singleCoreM), FIELD(Mc2MatMulV3TunnerTiling, singleCoreN),
               FIELD(Mc2MatMulV3TunnerTiling, singleCoreK), FIELD(Mc2MatMulV3TunnerTiling, baseM),
               FIELD(Mc2MatMulV3TunnerTiling, baseN), FIELD(Mc2MatMulV3TunnerTiling, baseK),
               FIELD(Mc2MatMulV3TunnerTiling, depthA1), FIELD(Mc2MatMulV3TunnerTiling, depthB1),
               FIELD(Mc2MatMulV3TunnerTiling, stepM), FIELD(Mc2MatMulV3TunnerTiling, stepN),
               FIELD(Mc2MatMulV3TunnerTiling, iterateOrder), FIELD(Mc2MatMulV3TunnerTiling, stepKa),
               FIELD(Mc2MatMulV3TunnerTiling, stepKb), FIELD(Mc2MatMulV3TunnerTiling, dbL0A),
               FIELD(Mc2MatMulV3TunnerTiling, dbL0B), FIELD(Mc2MatMulV3TunnerTiling, dbL0C),
               FIELD(Mc2MatMulV3TunnerTiling, l2MTileCnt), FIELD(Mc2MatMulV3TunnerTiling, l2NTileCnt),
               FIELD(Mc2MatMulV3TunnerTiling, l2MTileBlock), FIELD(Mc2MatMulV3TunnerTiling, l2NTileBlock),
               FIELD(Mc2MatMulV3TunnerTiling, l2IterateOrder), FIELD(Mc2MatMulV3TunnerTiling, tilingEnable));
} // namespace tuningtiling

#endif
