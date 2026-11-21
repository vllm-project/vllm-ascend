// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "register/op_impl_registry.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto* x = context->GetInputShape(0);
  const auto* weight = context->GetDynamicInputShape(3, 0);
  auto* output = context->GetOutputShape(0);
  auto* scale = context->GetOutputShape(1);
  const auto* xDesc = context->GetInputDesc(0);
  if (x == nullptr || weight == nullptr || output == nullptr || scale == nullptr || xDesc == nullptr ||
      x->GetDimNum() != 2 || weight->GetDimNum() != 3) {
    return ge::GRAPH_FAILED;
  }

  const auto m = x->GetDim(0);
  const auto n = weight->GetDim(xDesc->GetDataType() == ge::DT_FLOAT4_E2M1 ? 1 : 2);
  const auto outputN = n < 0 ? n : n / 2;
  output->SetDimNum(2);
  output->SetDim(0, m);
  output->SetDim(1, outputN);
  scale->SetDimNum(3);
  scale->SetDim(0, m);
  scale->SetDim(1, outputN < 0 ? outputN : (outputN + 63) / 64);
  scale->SetDim(2, 2);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(GmmSwigluOaiMxQuant).InferShape(InferShape).InferDataType(InferDtype);
