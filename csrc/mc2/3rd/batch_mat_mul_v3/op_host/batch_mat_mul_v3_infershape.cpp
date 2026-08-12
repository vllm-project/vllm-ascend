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
 * \file batch_mat_mul_v3_infershape.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"

namespace {
constexpr size_t BATCH_MATMUL_BIAS_IDX = 2; // input_bias index is different with op_type
static ge::graphStatus Mc2InferShapeForBatchMatMulV3(gert::InferShapeContext *context)
{
    return Ops::Transformer::InferShapeForBatchMatMul(context, 0, BATCH_MATMUL_BIAS_IDX);
}

// 由于编译期InferShapeRange在InferShape之后，因此该函数无需重复InferShape的dtype校验
static ge::graphStatus Mc2InferShapeRangeForBatchMatMulV3(gert::InferShapeRangeContext *context)
{
    return Ops::Transformer::InferShapeRangeForBatchMatMul(context, 0, BATCH_MATMUL_BIAS_IDX);
}

IMPL_OP_INFERSHAPE(Mc2BatchMatMulV3)
    .InferShape(Mc2InferShapeForBatchMatMulV3)
    .InferShapeRange(Mc2InferShapeRangeForBatchMatMulV3);

} // namespace
