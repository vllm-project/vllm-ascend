/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include "../op_kernel/hc_post_pre_norm_config.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
namespace {

constexpr std::size_t kResidualInput = 0;
constexpr std::size_t kOutputCount = 6;

ge::graphStatus InferShapeForHcPostPreNorm(gert::InferShapeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape* residual = context->GetInputShape(kResidualInput);
  OP_CHECK_NULL_WITH_CONTEXT(context, residual);
  OP_CHECK_IF(residual->GetDimNum() != 3, OP_LOGE(context, "residual must have shape [T, hc_mult, H]"),
              return ge::GRAPH_FAILED);

  const int64_t tokens = residual->GetDim(0);
  const int64_t hcMult = residual->GetDim(1);
  const int64_t hidden = residual->GetDim(2);
  const auto* attrs = context->GetAttrs();
  const auto* returnFp32 =
      attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(::HcPostPreNorm::kReturnFp32AttrIndex);

  const std::array<gert::Shape, kOutputCount> shapes = {
      gert::Shape({tokens, hidden}),
      gert::Shape({tokens, hcMult}),
      gert::Shape({tokens, hcMult, hcMult}),
      gert::Shape({tokens, hcMult}),
      *residual,
      returnFp32 == nullptr || *returnFp32 ? gert::Shape({tokens, hidden}) : gert::Shape({0}),
  };

  for (std::size_t index = 0; index < shapes.size(); ++index) {
    gert::Shape* output = context->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    *output = shapes[index];
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForHcPostPreNorm(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  constexpr std::array<ge::DataType, kOutputCount> types = {
      ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
  };

  for (std::size_t index = 0; index < types.size(); ++index) {
    context->SetOutputDataType(index, types[index]);
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(HcPostPreNorm).InferShape(InferShapeForHcPostPreNorm).InferDataType(InferDataTypeForHcPostPreNorm);

}  // namespace ops
