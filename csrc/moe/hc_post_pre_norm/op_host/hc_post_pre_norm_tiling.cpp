/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "../../hc_pre/op_host/hc_pre_tiling.h"
#include "../op_kernel/hc_post_pre_norm_config.h"

namespace optiling {

REGISTER_TILING_DATA_CLASS(HcPostPreNorm, HcPreTilingData)

namespace {

using HcPostPreNorm::kHcMult;
using HcPostPreNorm::kHiddenSize;

constexpr int64_t kMixCount = kHcMult * (kHcMult + 2);
constexpr int64_t kScaleCount = 3;

// RMSNormCast uses uint32_t for the flattened activation size and offsets.
constexpr int64_t kMaxTokens = std::numeric_limits<uint32_t>::max() / kHiddenSize;

enum InputIndex : std::size_t {
  kResidual = 0,
  kFfnHcFn,
  kFfnHcScale,
  kFfnHcBase,
  kAttnPre,
  kAttnOut,
  kAttnPost,
  kAttnComb,
  kFfnNormWeight,
};

bool HasShape(const gert::StorageShape* storage, std::initializer_list<int64_t> expected) {
  if (storage == nullptr) {
    return false;
  }

  const auto& shape = storage->GetStorageShape();
  if (shape.GetDimNum() != expected.size()) {
    return false;
  }

  std::size_t axis = 0;
  for (const int64_t dim : expected) {
    if (shape.GetDim(axis++) != dim) {
      return false;
    }
  }
  return true;
}

bool CheckInputShapes(gert::TilingContext* context) {
  const auto* residual = context->GetInputShape(kResidual);
  if (residual == nullptr || residual->GetStorageShape().GetDimNum() != 3) {
    return false;
  }

  const int64_t tokens = residual->GetStorageShape().GetDim(0);
  if (tokens <= 0 || tokens > kMaxTokens) {
    return false;
  }

  return HasShape(residual, {tokens, kHcMult, kHiddenSize}) &&
         HasShape(context->GetInputShape(kFfnHcFn), {kMixCount, kHcMult * kHiddenSize}) &&
         HasShape(context->GetInputShape(kFfnHcScale), {kScaleCount}) &&
         HasShape(context->GetInputShape(kFfnHcBase), {kMixCount}) &&
         HasShape(context->GetInputShape(kAttnPre), {tokens, kHcMult}) &&
         HasShape(context->GetInputShape(kAttnOut), {tokens, kHiddenSize}) &&
         HasShape(context->GetInputShape(kAttnPost), {tokens, kHcMult}) &&
         HasShape(context->GetInputShape(kAttnComb), {tokens, kHcMult, kHcMult}) &&
         HasShape(context->GetInputShape(kFfnNormWeight), {kHiddenSize});
}

ge::graphStatus TilingForHcPostPreNorm(gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  OPS_ERR_IF(context->GetPlatformInfo() == nullptr,
             OPS_LOG_E(context->GetNodeName(), "HcPostPreNorm requires platform information"), return ge::GRAPH_FAILED);

  OPS_ERR_IF(!CheckInputShapes(context),
             OPS_LOG_E(context->GetNodeName(),
                       "Invalid HcPostPreNorm input shapes; expected hc_mult=%ld, H=%ld, and matching token counts",
                       kHcMult, kHiddenSize),
             return ge::GRAPH_FAILED);

  const auto* attrs = context->GetAttrs();
  const auto* returnFp32 = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(HcPostPreNorm::kReturnFp32AttrIndex);
  const bool writeFp32 = returnFp32 == nullptr || *returnFp32;
  const auto* residual = context->GetInputShape(kResidual);
  const auto* fp32Output = context->GetOutputShape(HcPostPreNorm::kFp32OutputIndex);
  const bool validOutput = writeFp32 ? HasShape(fp32Output, {residual->GetStorageShape().GetDim(0), kHiddenSize})
                                    : HasShape(fp32Output, {0});
  OPS_ERR_IF(!validOutput, OPS_LOG_E(context->GetNodeName(), "Invalid y_fp32 shape for return_fp32"),
             return ge::GRAPH_FAILED);

  HcPreTiling tiling(context, /*postPreNorm=*/true);
  const auto status = tiling.DoOpTiling();
  if (status == ge::GRAPH_SUCCESS) {
    context->SetTilingKey(writeFp32 ? HcPostPreNorm::kWithFp32TilingKey : HcPostPreNorm::kWithoutFp32TilingKey);
  }
  return status;
}

}  // namespace

IMPL_OP_OPTILING(HcPostPreNorm).Tiling(TilingForHcPostPreNorm);

}  // namespace optiling
