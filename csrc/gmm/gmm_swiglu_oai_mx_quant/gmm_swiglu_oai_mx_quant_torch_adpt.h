// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TORCH_ADPT_H
#define VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TORCH_ADPT_H

#include "gmm_swiglu_oai_mx_quant_common.h"
#include <algorithm>
#include <array>

namespace vllm_ascend::gmm_swiglu_oai {

struct TensorDesc {
  const at::Tensor& tensor;
  aclDataType dtype;
  bool nz = false;
};

inline void CheckWeightStorage(const at::Tensor& weight) {
  const auto& desc = NPUBridge::GetNpuStorageImplDesc(weight);
  const auto format = desc.npu_format_;
  const std::array<int64_t, 5> packed{weight.size(0), weight.size(2) / 32, weight.size(1) / 16, 16, 32};
  TORCH_CHECK((format == ACL_FORMAT_FRACTAL_NZ || format == ACL_FORMAT_FRACTAL_NZ_C0_32) &&
                  weight.storage_offset() == 0 && desc.storage_sizes_.size() == packed.size() &&
                  std::equal(desc.storage_sizes_.begin(), desc.storage_sizes_.end(), packed.begin()),
              "weight must be batched FRACTAL_NZ with 32-byte C0");
}

inline aclTensor* ConvertType(const TensorDesc& desc) {
  static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
  TORCH_CHECK(createTensor, "aclCreateTensor is unavailable");
  const auto& tensor = desc.tensor;
  auto shape = tensor.sizes().vec();
  const bool fp4 = desc.dtype == ACL_FLOAT4_E2M1;
  if (fp4) {
    shape.back() *= 2;
  }
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t dim = shape.size() - 1; dim > 0; --dim) {
    strides[dim - 1] = strides[dim] * shape[dim];
  }

  // ACL dimensions and strides count logical FP4 values. PyTorch counts packed bytes.
  auto storage = shape;
  if (desc.nz) {
    const int64_t c0 = fp4 ? 64 : 32;
    storage = {shape[0], shape[2] / c0, shape[1] / 16, 16, c0};
  }
  auto* acl = createTensor(shape.data(), shape.size(), desc.dtype, strides.data(), 0,
                           desc.nz ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND, storage.data(), storage.size(),
                           const_cast<void*>(tensor.const_data_ptr()));
  TORCH_CHECK(acl, "could not create MX tensor descriptor");
  return acl;
}

struct SingleTensorList {
  TensorDesc desc;
};

inline aclTensorList* ConvertType(const SingleTensorList& list) {
  static const auto createList = GET_OP_API_FUNC(aclCreateTensorList);
  TORCH_CHECK(createList, "aclCreateTensorList is unavailable");
  const aclTensor* tensor = ConvertType(list.desc);
  auto* result = createList(&tensor, 1);
  if (result == nullptr) {
    Release(const_cast<aclTensor*>(tensor));
  }
  TORCH_CHECK(result, "could not create the grouped matmul tensor list");
  return result;
}

}  // namespace vllm_ascend::gmm_swiglu_oai

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> gmm_swiglu_oai_mx_quant(
    const at::Tensor& x, at::TensorList weight, at::TensorList weightScale, const at::Tensor& xScale,
    const at::Tensor& groups, c10::optional<int64_t> groupListType, c10::optional<double> clampLimit,
    c10::optional<double> gluAlpha, c10::optional<double> gluBias, c10::optional<double> dstTypeMax) {
  const bool fp4 = gmm_swiglu_oai::ValidateInputs(x, weight, weightScale, xScale, groups);
  auto args = gmm_swiglu_oai::ValidateAttrs(fp4, groupListType, clampLimit, gluAlpha, gluBias, dstTypeMax);
  for (const auto& tensor : {x, xScale, groups, weight[0], weightScale[0]}) {
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                "all inputs must be contiguous and on the same NPU");
  }
  TORCH_CHECK(x.is_privateuseone(), "x must be on an NPU");
  gmm_swiglu_oai::CheckWeightStorage(weight[0]);
  for (const auto& tensor : {x, xScale, groups, weightScale[0]}) {
    TORCH_CHECK(IsOpInputBaseFormat(tensor), "x, scales and group_list must have dense storage");
  }

  const c10_npu::OptionalNPUGuard deviceGuard(x.device());
  auto [y, yScale] = gmm_swiglu_oai::MakeOutput(x, weight[0], fp4);
  if (x.size(0) == 0) {
    return {y, yScale};
  }
  const auto dtype = fp4 ? ACL_FLOAT4_E2M1 : ACL_FLOAT8_E4M3FN;
  const gmm_swiglu_oai::TensorDesc activation{x, dtype};
  const gmm_swiglu_oai::TensorDesc activationScale{xScale, ACL_FLOAT8_E8M0};
  const gmm_swiglu_oai::TensorDesc groupList{groups, ACL_INT64};
  const gmm_swiglu_oai::SingleTensorList weights{{weight[0], dtype, true}};
  const gmm_swiglu_oai::SingleTensorList scales{{weightScale[0], ACL_FLOAT8_E8M0}};
  const gmm_swiglu_oai::TensorDesc output{y, dtype};
  const gmm_swiglu_oai::TensorDesc outputScale{yScale, ACL_FLOAT8_E8M0};
  EXEC_NPU_CMD(aclnnGmmSwigluOaiMxQuant, activation, activationScale, groupList, weights, scales, args.groupListType,
               args.clampLimit, args.gluAlpha, args.gluBias, args.dstTypeMax, output, outputScale);
  return {y, yScale};
}

}  // namespace vllm_ascend
#endif
