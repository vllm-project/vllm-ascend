// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TORCH_ADPT_H
#define VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <cmath>

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_META_ONLY
#include <algorithm>
#include <array>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../../aclnn_torch_adapter/op_api_common.h"
#endif

namespace vllm_ascend::gmm_swiglu_oai {

constexpr auto kFP4 = at::ScalarType::Float4_e2m1fn_x2;
constexpr auto kFP8 = at::ScalarType::Float8_e4m3fn;
constexpr auto kScale = at::ScalarType::Float8_e8m0fnu;
constexpr int64_t kInputAlignment = 128;
constexpr int64_t kScalePairGroup = 64;
constexpr int64_t kFp8TokensPerExpert = 513;

inline bool IsScale(at::ScalarType dtype) { return dtype == kScale || dtype == at::kByte || dtype == at::kChar; }

struct Args {
  int64_t groupListType;
  double clampLimit;
  double gluAlpha;
  double gluBias;
  double dstTypeMax;
};

inline Args ValidateAttrs(bool fp4, c10::optional<int64_t> groupListType, c10::optional<double> clampLimit,
                          c10::optional<double> gluAlpha, c10::optional<double> gluBias,
                          c10::optional<double> dstTypeMax) {
  const Args args{groupListType.value_or(0), clampLimit.value_or(7.0), gluAlpha.value_or(1.702), gluBias.value_or(1.0),
                  dstTypeMax.value_or(0.0)};
  TORCH_CHECK(args.groupListType == 0 || args.groupListType == 1, "group_list_type must be 0 or 1");
  TORCH_CHECK(std::isfinite(static_cast<float>(args.clampLimit)) && static_cast<float>(args.clampLimit) > 0 &&
                  std::isfinite(static_cast<float>(args.gluAlpha)) && std::isfinite(static_cast<float>(args.gluBias)),
              "clamp_limit must be finite and positive, and glu_alpha and glu_bias must be finite");
  TORCH_CHECK(args.dstTypeMax == 0 || (fp4 && args.dstTypeMax >= 6 && args.dstTypeMax <= 12),
              "dst_type_max must be 0 for MXFP8 and either 0 or in [6, 12] for MXFP4");
  return args;
}

inline bool ValidateInputs(const at::Tensor& x, at::TensorList weights, at::TensorList scales, const at::Tensor& xScale,
                           const at::Tensor& groups) {
  const auto dtype = x.scalar_type();
  const bool fp4 = dtype == kFP4 || dtype == at::kByte;
  TORCH_CHECK(fp4 || dtype == kFP8, "x must be packed MXFP4 or MXFP8 E4M3FN");
  TORCH_CHECK(weights.size() == 1 && scales.size() == 1, "weight and weight_scale must each contain one tensor");
  const auto& w = weights[0];
  const auto& ws = scales[0];
  TORCH_CHECK(fp4 ? (w.scalar_type() == kFP4 || w.scalar_type() == at::kByte || w.scalar_type() == at::kChar)
                  : w.scalar_type() == kFP8,
              "weight format must match x");
  TORCH_CHECK(IsScale(xScale.scalar_type()) && IsScale(ws.scalar_type()), "MX scales must use E8M0 storage");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 3 && ws.dim() == 4 && xScale.dim() == 3 && groups.dim() == 1 &&
                  groups.scalar_type() == at::kLong,
              "expected rank-2 x, rank-3 weight, rank-4 weight_scale, rank-3 x_scale and a rank-1 int64 group_list");
  const auto m = x.sym_size(0);
  const auto k = x.sym_size(1) * (fp4 ? 2 : 1);
  const auto e = groups.sym_size(0);
  const auto n = w.sym_size(fp4 ? 1 : 2);
  const auto scaleK = k / kScalePairGroup;
  TORCH_CHECK(e > 0 && e <= 1024 && n > 0 && k > 0 && n % kInputAlignment == 0 && k % kInputAlignment == 0 &&
                  w.sym_size(0) == e && w.sym_size(fp4 ? 2 : 1) == x.sym_size(1),
              "unsupported MX grouped matmul dimensions");
  TORCH_CHECK(fp4 || m / e >= kFp8TokensPerExpert, "MXFP8 fusion requires at least 513 tokens per expert on average");
  TORCH_CHECK(xScale.sym_size(0) == m && xScale.sym_size(1) == scaleK && xScale.sym_size(2) == 2 &&
                  ws.sym_size(0) == e && ws.sym_size(fp4 ? 1 : 2) == n && ws.sym_size(fp4 ? 2 : 1) == scaleK &&
                  ws.sym_size(3) == 2,
              "invalid paired E8M0 scale dimensions");
  return fp4;
}

inline std::tuple<at::Tensor, at::Tensor> MakeOutput(const at::Tensor& x, const at::Tensor& weight, bool fp4) {
  const auto n = weight.sym_size(fp4 ? 1 : 2);
  auto y = at::empty_symint({x.sym_size(0), n / (fp4 ? 4 : 2)}, x.options().dtype(fp4 ? at::kByte : kFP8));
  if (fp4) {
    y = y.view(kFP4);
  }
  auto scale = at::empty_symint({x.sym_size(0), n / (2 * kScalePairGroup), 2}, x.options().dtype(at::kByte));
  return {y, scale.view(kScale)};
}

}  // namespace vllm_ascend::gmm_swiglu_oai

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_META_ONLY
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

// Keep descriptor conversion local to this ACLNN op. It expects logical FP4
// shape and strides, raw scale bytes interpreted as E8M0, and explicit ND or
// batched NZ storage dimensions instead of the default Torch metadata.
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
#endif  // VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_META_ONLY
#endif  // VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TORCH_ADPT_H
