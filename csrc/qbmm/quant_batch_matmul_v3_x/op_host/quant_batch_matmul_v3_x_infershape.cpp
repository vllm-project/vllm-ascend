/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_x_infershape.cpp
 * \brief Inline shape inference for QuantBatchMatmulV3X (no common/ deps); handles ND and FRACTAL_NZ x2.
 * \author Feodor Pisnitchenko
 */
#include "register/op_impl_registry.h"
#include "graph/operator_reg.h"
#include "quant_batch_matmul_v3_x_tiling.h"

constexpr size_t QBM_MAX_SHAPE_SIZE = 6;
constexpr size_t QBM_MIN_SHAPE_SIZE = 2;

ge::graphStatus InferShapeForQuantBatchMatmulV3X(gert::InferShapeContext* context)
{
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    if (shape_x1 == nullptr || shape_x2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    if (dim_a < QBM_MIN_SHAPE_SIZE || dim_a > QBM_MAX_SHAPE_SIZE ||
        dim_b < QBM_MIN_SHAPE_SIZE || dim_b > QBM_MAX_SHAPE_SIZE) {
        return ge::GRAPH_FAILED;
    }

    // Get transpose attributes
    auto attrs = context->GetAttrs();
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    if (attrs != nullptr) {
        auto tx1 = attrs->GetAttrPointer<bool>(1);  // transpose_x1 at index 1
        auto tx2 = attrs->GetAttrPointer<bool>(2);  // transpose_x2 at index 2
        if (tx1 != nullptr) transpose_x1 = *tx1;
        if (tx2 != nullptr) transpose_x2 = *tx2;
    }

    // x1: [..., M, K] or [..., K, M] if transposed
    int64_t M = transpose_x1 ? shape_x1->GetDim(dim_a - 1) : shape_x1->GetDim(dim_a - 2);

    // x2: FRACTAL_NZ K-major layout is [..., K1, N1, N0=16, K0=32] so
    //     N = N1 * N0; ND layout is [..., K, N] (or [..., N, K] if transposed).
    int64_t N = optiling::DeriveN(*shape_x2, transpose_x2);

    // Output shape: batch dims from x1 + [M, N]
    auto output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    output_shape->SetDimNum(dim_a);
    for (size_t i = 0; i < dim_a - 2; ++i) {
        output_shape->SetDim(i, shape_x1->GetDim(i));
    }
    output_shape->SetDim(dim_a - 2, M);
    output_shape->SetDim(dim_a - 1, N);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatmulV3X)
    .InferShape(InferShapeForQuantBatchMatmulV3X);

// Old-style operator_factory registrations. atc uses this registry for InferShape
// and InferFormat. The built-in libopsproto.so overwrites OpImplRegisterV2 entries
// but not operator_factory entries, so these actually take effect in atc.
namespace ge {

graphStatus InferShapeOld(Operator &op)
{
    auto x1_shape = op.GetInputDescByName("x1").GetShape();
    auto x2_shape = op.GetInputDescByName("x2").GetShape();

    size_t dim_a = x1_shape.GetDimNum();
    size_t dim_b = x2_shape.GetDimNum();
    if (dim_a < QBM_MIN_SHAPE_SIZE || dim_a > QBM_MAX_SHAPE_SIZE ||
        dim_b < QBM_MIN_SHAPE_SIZE || dim_b > QBM_MAX_SHAPE_SIZE) {
        return GRAPH_FAILED;
    }

    bool transpose_x1 = false;
    bool transpose_x2 = false;
    op.GetAttr("transpose_x1", transpose_x1);
    op.GetAttr("transpose_x2", transpose_x2);

    int64_t M = transpose_x1 ? x1_shape.GetDim(dim_a - 1) : x1_shape.GetDim(dim_a - 2);

    int64_t N = optiling::DeriveN(x2_shape, transpose_x2);

    std::vector<int64_t> dims;
    for (size_t i = 0; i < dim_a - 2; ++i)
        dims.push_back(x1_shape.GetDim(i));
    dims.push_back(M);
    dims.push_back(N);

    TensorDesc output_desc = op.GetOutputDescByName("y");
    output_desc.SetShape(Shape(dims));
    op.UpdateOutputDesc("y", output_desc);

    return GRAPH_SUCCESS;
}

graphStatus InferFormatOld(Operator &op)
{
    // x1: always ND
    TensorDesc x1_desc = op.GetInputDescByName("x1");
    x1_desc.SetFormat(FORMAT_ND);
    x1_desc.SetOriginFormat(FORMAT_ND);
    op.UpdateInputDesc("x1", x1_desc);

    // x2: fix metadata for FRACTAL_NZ to prevent Transpose insertion
    TensorDesc x2_desc = op.GetInputDescByName("x2");
    Format x2_fmt = x2_desc.GetFormat();

    // Our operator always expects x2 in blocked ZN format.
    // torchair exports with origin_format=ND, logical shape [K,N], transpose_x2=False.
    // Our InferShapeOld handles [K,N]+transpose_x2=False correctly.
    // Just set format to FRACTAL_NZ so atc:
    //   1) doesn't insert Transpose (our InferShape accepts this convention)
    //   2) computes correct ZN storage shape from origin_shape [K,N]
    //      -> [K/32, N/16, 16, 32] for buffer allocation
    x2_desc.SetOriginFormat(FORMAT_FRACTAL_NZ);
    x2_desc.SetFormat(FORMAT_FRACTAL_NZ);
    op.UpdateInputDesc("x2", x2_desc);

    // MatMulA8W8TransposeTransDataFusionPass requires transpose_x2=True for NZ format
    op.SetAttr("transpose_x2", true);

    // scale: always ND
    TensorDesc scale_desc = op.GetInputDescByName("scale");
    scale_desc.SetFormat(FORMAT_ND);
    scale_desc.SetOriginFormat(FORMAT_ND);
    op.UpdateInputDesc("scale", scale_desc);

    // y: always ND
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetFormat(FORMAT_ND);
    y_desc.SetOriginFormat(FORMAT_ND);
    op.UpdateOutputDesc("y", y_desc);

    return GRAPH_SUCCESS;
}

static const InferShapeFuncRegister  if_reg_qbm("QuantBatchMatmulV3X",
    [](Operator &v) -> graphStatus { return InferShapeOld(v); });
static const InferFormatFuncRegister ff_reg_qbm("QuantBatchMatmulV3X",
    [](Operator &v) -> graphStatus { return InferFormatOld(v); });

}  // namespace ge
