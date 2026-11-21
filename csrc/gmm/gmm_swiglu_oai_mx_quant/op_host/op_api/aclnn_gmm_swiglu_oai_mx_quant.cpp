// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "aclnn_gmm_swiglu_oai_mx_quant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

#include <initializer_list>
#include <limits>

using namespace op;

namespace {
bool MatchesDense(const aclTensor& tensor, DataType dtype, std::initializer_list<int64_t> dimensions) {
  const auto& view = tensor.GetViewShape();
  const auto& storage = tensor.GetStorageShape();
  const auto& strides = tensor.GetViewStrides();
  if (tensor.GetDataType() != dtype || tensor.GetStorageFormat() != ge::FORMAT_ND || tensor.GetViewOffset() != 0 ||
      view.GetDimNum() != dimensions.size() || storage.GetDimNum() != dimensions.size() ||
      strides.size() != dimensions.size()) {
    return false;
  }
  int64_t stride = 1;
  for (size_t dim = dimensions.size(); dim > 0; --dim) {
    const auto extent = dimensions.begin()[dim - 1];
    if (view.GetDim(dim - 1) != extent || storage.GetDim(dim - 1) != extent || strides[dim - 1] != stride) {
      return false;
    }
    if (dim > 1) {
      if (stride > std::numeric_limits<int64_t>::max() / extent) {
        return false;
      }
      stride *= extent;
    }
  }
  return true;
}

bool CheckBuffers(const aclTensor& x, const aclTensor& xScale, const aclTensor& groups, const aclTensor& weight,
                  const aclTensor& weightScale, const aclTensor& y, const aclTensor& yScale) {
  const auto& xShape = x.GetViewShape();
  const auto& wShape = weight.GetViewShape();
  const auto& groupShape = groups.GetViewShape();
  const auto dtype = x.GetDataType();
  const bool fp4 = dtype == ge::DT_FLOAT4_E2M1;
  if ((!fp4 && dtype != ge::DT_FLOAT8_E4M3FN) || xShape.GetDimNum() != 2 || wShape.GetDimNum() != 3 ||
      groupShape.GetDimNum() != 1 || weight.GetDataType() != dtype ||
      weight.GetStorageFormat() != ge::FORMAT_FRACTAL_NZ || weight.GetViewOffset() != 0) {
    return false;
  }
  const auto m = xShape.GetDim(0);
  const auto k = xShape.GetDim(1);
  const auto n = wShape.GetDim(fp4 ? 1 : 2);
  const auto e = groupShape.GetDim(0);
  if (m <= 0 || k <= 0 || n <= 0 || e <= 0 || n % 128 != 0 || k % 128 != 0) {
    return false;
  }
  constexpr auto scale = ge::DT_FLOAT8_E8M0;
  return MatchesDense(x, dtype, {m, k}) && MatchesDense(xScale, scale, {m, k / 64, 2}) &&
         MatchesDense(groups, ge::DT_INT64, {e}) &&
         (fp4 ? MatchesDense(weightScale, scale, {e, n, k / 64, 2})
              : MatchesDense(weightScale, scale, {e, k / 64, n, 2})) &&
         MatchesDense(y, dtype, {m, n / 2}) && MatchesDense(yScale, scale, {m, n / 128, 2});
}
}  // namespace

namespace l0op {
OP_TYPE_REGISTER(GmmSwigluOaiMxQuant);

aclnnStatus LaunchGmmSwigluOaiMxQuant(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                                         const aclTensorList* weight, const aclTensorList* weightScale,
                                         int64_t groupListType, float clampLimit, float gluAlpha, float gluBias,
                                         float dstTypeMax, aclTensor* y, aclTensor* yScale, aclOpExecutor* executor) {
  return ADD_TO_LAUNCHER_LIST_AICORE(GmmSwigluOaiMxQuant, OP_INPUT(x, xScale, groupList, weight, weightScale),
                                     OP_OUTPUT(y, yScale),
                                     OP_ATTR(groupListType, clampLimit, gluAlpha, gluBias, dstTypeMax));
}
}  // namespace l0op

extern "C" {

aclnnStatus aclnnGmmSwigluOaiMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* xScale,
                                                        const aclTensor* groupList, const aclTensorList* weight,
                                                        const aclTensorList* weightScale, int64_t groupListType,
                                                        double clampLimit, double gluAlpha, double gluBias,
                                                        double dstTypeMax, aclTensor* y, aclTensor* yScale,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
  if (!x || !xScale || !groupList || !weight || !weightScale || !y || !yScale || !workspaceSize || !executor) {
    return ACLNN_ERR_PARAM_NULLPTR;
  }
  if (weight->Size() != 1 || weightScale->Size() != 1 || !(*weight)[0] || !(*weightScale)[0] ||
      !CheckBuffers(*x, *xScale, *groupList, *(*weight)[0], *(*weightScale)[0], *y, *yScale)) {
    return ACLNN_ERR_PARAM_INVALID;
  }
  // The launcher uses origin dimensions and NZ storage dimensions separately.
  const std::initializer_list<const aclTensor*> tensors{x, xScale, groupList, (*weight)[0], (*weightScale)[0],
                                                        y, yScale};
  for (const auto* tensor : tensors) {
    tensor->SetOriginalShape(tensor->GetViewShape());
  }
  L2_DFX_PHASE_1(aclnnGmmSwigluOaiMxQuant,
                 DFX_IN(x, xScale, groupList, weight, weightScale, groupListType, clampLimit, gluAlpha, gluBias,
                        dstTypeMax),
                 DFX_OUT(y, yScale));
  auto handle = CREATE_EXECUTOR();
  if (!handle.get()) {
    return ACLNN_ERR_INNER_CREATE_EXECUTOR;
  }
  const auto status = l0op::LaunchGmmSwigluOaiMxQuant(x, xScale, groupList, weight, weightScale, groupListType,
                                                         static_cast<float>(clampLimit), static_cast<float>(gluAlpha),
                                                         static_cast<float>(gluBias), static_cast<float>(dstTypeMax), y,
                                                         yScale, handle.get());
  if (status != ACLNN_SUCCESS) {
    return status;
  }
  *workspaceSize = handle->GetWorkspaceSize();
  handle.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGmmSwigluOaiMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGmmSwigluOaiMxQuant);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // extern "C"
