// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_COMMON_H
#define VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_COMMON_H

#include <ATen/ATen.h>
#include <cmath>

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
#endif
