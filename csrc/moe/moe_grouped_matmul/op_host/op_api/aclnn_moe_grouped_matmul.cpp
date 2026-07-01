/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_grouped_matmul.h"
#include "aclnn_moe_grouped_matmul_weight_nz.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "moe_grouped_matmul_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
  static constexpr size_t ALIGN_NZ_4BIT_N = 64UL;
  static constexpr size_t ALIGN_NZ_4BIT_K = 64UL;
  static constexpr size_t ALIGN_NZ_INT8_N = 32UL;
  static constexpr size_t ALIGN_NZ_K = 16UL;
  static constexpr size_t DIMS_THREE_FOR_GMM = 3UL;
  static constexpr size_t LAST_FIRST_DIM_INDEX = 1;
  static constexpr size_t LAST_SECOND_DIM_INDEX = 2;
  static constexpr size_t LAST_THIRD_DIM_INDEX = 3;

  enum class GMMWeightVersion : uint32_t {
      WeightNd = 1U,
      WeightNz = 2U
  };

  struct MoeGroupedMatmulParams {
      const aclTensorList *x = nullptr;
      const aclTensorList *weight = nullptr;
      const aclTensor *groupTensor = nullptr;
      bool transposeWeight = false;
      bool isSingleWeight = false;
      GMMWeightVersion weightVersion = GMMWeightVersion::WeightNd;
      const aclTensorList *y = nullptr;
      DataType xDtype = DataType::DT_BF16;
  };
}


namespace {

static aclnnStatus CheckNotNull(const aclTensorList *x, const aclTensorList *weight, const aclTensorList *y) {
  CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
  CHECK_COND(x->Size() != 0, ACLNN_ERR_PARAM_INVALID, "x must not be empty tensorlist.");
  CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight must not be nullptr.");
  CHECK_COND(weight->Size() != 0, ACLNN_ERR_PARAM_INVALID, "weight must not be empty tensorlist.");
  CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
  CHECK_COND(y->Size() != 0, ACLNN_ERR_PARAM_INVALID, "y must not be empty tensorlist.");
  return ACLNN_SUCCESS;
}

static aclnnStatus TransWeightToNzCheckAlign(MoeGroupedMatmulParams &gmmParams, const aclTensor *weight)
{
  size_t viewDimNum = weight->GetViewShape().GetDimNum();
  uint64_t k = gmmParams.transposeWeight ? weight->GetViewShape().GetDim(viewDimNum - 1) :
                                           weight->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
  uint64_t n = gmmParams.transposeWeight ? weight->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX) :
                                           weight->GetViewShape().GetDim(viewDimNum - 1);
  bool k_align = false;
  bool n_align = false;
  if (weight->GetDataType() == DataType::DT_BF16 || weight->GetDataType() == DataType::DT_FLOAT16) {
    k_align = k % ALIGN_NZ_K == 0;
    n_align = n % ALIGN_NZ_K == 0;
  }
  CHECK_COND(k_align == true && n_align == true, ACLNN_ERR_PARAM_INVALID,
             "When weight format is FRACTAL_NZ, weight'shape(k[%lu], n[%lu]) should be divisible by the "
             "following shape: BF16/FP16[16, 16]). If the weight is transposed,"
             "the k/n need to be reversed.",
             k, n);
  return ACLNN_SUCCESS;
}

static aclnnStatus TransWeightToNz(MoeGroupedMatmulParams &gmmParams, aclOpExecutor *executor) {
  const aclTensorList *&weights = gmmParams.weight;
  const aclTensorList *&x = gmmParams.x;
  CHECK_COND((*x)[0] != nullptr, ACLNN_ERR_PARAM_INVALID, "The first tensor of x is nullptr!");
  size_t wLength = weights->Size();
  for (size_t i(0); i < wLength; ++i) {
    const aclTensor* weight = (*weights)[i];
    if (weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
        weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ_C0_16 &&
        weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ_C0_32) {
      break;
    }
    TransWeightToNzCheckAlign(gmmParams, weight);
    continue;
  }
  return ACLNN_SUCCESS;
}

static const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor) {
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(input->GetViewFormat());
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

static aclnnStatus DataContiguous(const aclTensorList *&tensors, aclOpExecutor *executor) {
    std::vector<const aclTensor *> tensorsVec;
    const aclTensor *contiguousTensor = nullptr;
    for (size_t i = 0; i < tensors->Size(); ++i) {
        const aclTensor *tensor = (*tensors)[i];
        contiguousTensor = l0op::Contiguous(tensor, executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec.push_back(contiguousTensor);
    }
    tensors = executor->AllocTensorList(tensorsVec.data(), tensorsVec.size());
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(MoeGroupedMatmulParams &params, aclOpExecutor *executorPtr) {
  CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "Contiguous x failed.");  // make x contiguous
  DataType xDtype = (*params.x)[0]->GetDataType();
  DataType weightDtype = (*params.weight)[0]->GetDataType();
  CHECK_COND(DataContiguous(params.weight, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
              "Contiguous weight failed."); // make w contiguous
  params.groupTensor = l0op::Contiguous(params.groupTensor, executorPtr);
  CHECK_COND(params.groupTensor != nullptr, ACLNN_ERR_PARAM_INVALID,
              "Contiguous groupTensor failed.");
  return ACLNN_SUCCESS;
}

static aclnnStatus GetGMMResultByL0Api(MoeGroupedMatmulParams &params, uint64_t *workspaceSize, aclOpExecutor **executor) {
  auto uniqueExecutor = CREATE_EXECUTOR();  // fixed written style, create OpExecutor
  aclOpExecutor *executorPtr = uniqueExecutor.get();
  CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // op::Shape wqbmmNzShape = (*params.weight)[0]->GetStorageShape();

  CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "ParamsDataContiguous failed.");
  if (params.weightVersion == GMMWeightVersion::WeightNz) {
      std::vector<const aclTensor *> tensorsVec;
      for (size_t i = 0; i < params.weight->Size(); ++i) {
          const aclTensor *tensor = (*params.weight)[i];
          op::Shape weightNzShape = tensor->GetViewShape();
          tensor = SetTensorToNZFormat(tensor, weightNzShape, executorPtr);
          tensorsVec.push_back(tensor);
      }
      params.weight = executorPtr->AllocTensorList(tensorsVec.data(), tensorsVec.size());
  }
  CHECK_COND(TransWeightToNz(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
             "TransWeightToNz failed.");
  // Invoke l0 operator MoeGroupedMatmul for calculation.
  auto result = l0op::MoeGroupedMatmul(params.x, params.weight,
                  params.groupTensor, params.transposeWeight,
                  (*params.y)[0]->GetViewShape(), params.y->Size(),
                  (*params.y)[0]->GetDataType(), executorPtr);
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  for (size_t i(0); i < params.y->Size(); ++i) {
    auto viewCopyResult = l0op::ViewCopy((*result)[i], (*params.y)[i], executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  // Standard syntax, get the size of workspace needed during computation.
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}


static aclnnStatus aclnnMoeGroupedMatmulGetWorkspaceSizeCommon(const aclTensorList *x, const aclTensorList *weight,
  const aclTensor *groupList, bool transposeWeight, GMMWeightVersion weightVersion,
  const aclTensorList *y, uint64_t *workspaceSize, aclOpExecutor **executor) {
  DataType xDtype = DataType::DT_UNDEFINED;
  for (size_t i = 0; i < x->Size(); ++i) {
    if ((*x)[i] != nullptr) {
      xDtype = (*x)[i]->GetDataType();
      break;
    }
  }
  MoeGroupedMatmulParams moeGmmParams{x, weight, groupList, transposeWeight, true, weightVersion, y, xDtype};
  aclnnStatus ret = GetGMMResultByL0Api(moeGmmParams, workspaceSize, executor);
  return ret;
}
}

aclnnStatus aclnnMoeGroupedMatmulWeightNzGetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensor *groupList, bool transposeWeight, aclTensorList *out,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnMoeGroupedMatmulWeightNz,
                 DFX_IN(x, weight, groupList),
                 DFX_OUT(out));
  return aclnnMoeGroupedMatmulGetWorkspaceSizeCommon(x, weight, groupList, transposeWeight, GMMWeightVersion::WeightNz, out, workspaceSize, executor);
}

aclnnStatus aclnnMoeGroupedMatmulGetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
  const aclTensor *groupList, bool transposeWeight, aclTensorList *out,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  CHECK_COND(CheckNotNull(x, weight, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
             "one of required inputs is nullptr.");
  // Standard syntax, Check parameters.
  L2_DFX_PHASE_1(aclnnMoeGroupedMatmul,
                 DFX_IN(x, weight, groupList),
                 DFX_OUT(out));
  CHECK_COND(weight->Size() != 0, ACLNN_ERR_PARAM_INVALID, "weight should not be null tensorlist ");
  return aclnnMoeGroupedMatmulGetWorkspaceSizeCommon(x, weight, groupList, transposeWeight, GMMWeightVersion::WeightNd, out, workspaceSize, executor);
}

aclnnStatus aclnnMoeGroupedMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                               aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMoeGroupedMatmul);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeGroupedMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMoeGroupedMatmulWeightNz);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GMM launch aicore");
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
