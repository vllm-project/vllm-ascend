// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#pragma once

#include <algorithm>
#include <cmath>

#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend::official_shared_swiglu {

inline bool CheckNoLegacyOverride() {
  auto paths = get_custom_lib_path();
  auto defaults = get_default_custom_lib_path();
  paths.insert(paths.end(), defaults.begin(), defaults.end());
  for (const auto& path : paths) {
    const auto library = real_path(path + "/" + GetCustOpApiLibName());
    if (library.empty()) {
      continue;
    }
    auto* handle = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    const bool conflict = dlsym(handle, "aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr;
    dlclose(handle);
    TORCH_CHECK(!conflict, "Official shared SwiGLU cannot coexist with a custom DequantSwigluQuant override: ", library,
                ". A direct libopapi lookup does not isolate the CANN kernel registry. "
                "Use a clean validation process; production coexistence requires namespacing the legacy operator. "
                "Do not unset the complete custom OPP path in a running model or silently fall back.");
  }
  return true;
}

// Do not select the legacy ACLNN implementation from custom OPP libraries.
inline void* GetOpApiFuncAddr(const char* name) {
  static auto handle = dlopen("libopapi.so", RTLD_LAZY | RTLD_LOCAL);
  TORCH_CHECK(handle != nullptr, "Cannot load official libopapi.so");
  return dlsym(handle, name);
}

inline std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& x, const at::Tensor& weight_scale,
                                              const at::Tensor& activation_scale, double clamp_limit,
                                              const c10::optional<at::Tensor>& group_counts,
                                              bool weight_is_interleaved) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kInt, "shared SwiGLU expects a 2D INT32 projection");
  const auto tokens = x.size(0), width = x.size(1), half = width / 2;
  TORCH_CHECK(width > 0 && width % 2 == 0, "shared SwiGLU width must be positive and even");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat && weight_scale.numel() == width,
              "shared SwiGLU expects FP32 per-channel weight scales");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat && activation_scale.numel() == tokens,
              "shared SwiGLU expects one FP32 activation scale per token");
  TORCH_CHECK(weight_scale.device() == x.device() && activation_scale.device() == x.device(),
              "shared SwiGLU inputs must be on the same device");
  TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0, "clamp_limit must be positive and finite");
  if (tokens == 0) {
    return {at::empty({0, half}, x.options().dtype(at::kChar)), at::empty({0}, x.options().dtype(at::kFloat))};
  }
  static const bool no_legacy_override = CheckNoLegacyOverride();
  (void)no_legacy_override;

  constexpr int64_t kMinGroupTokens = 8;
  const auto padded_tokens = std::max(tokens, kMinGroupTokens);
  auto halves = x.chunk(2, -1);
  at::Tensor packed;
  auto scales = activation_scale.reshape({tokens});
  // Repeated rows do not affect the original rows' independent dynamic quantization.
  // Stack the expanded halves directly: no generic ConstantPadNd or full transpose.
  if (tokens == 1) {
    packed = at::stack({halves[0].expand({padded_tokens, half}), halves[1].expand({padded_tokens, half})}, -1);
    auto repeated_scales = at::empty({padded_tokens}, scales.options());
    c10::SmallVector<int64_t, 1> shape{padded_tokens};
    at::IntArrayRef sizes(shape);
    EXEC_NPU_CMD(aclnnExpand, scales, sizes, repeated_scales);
    scales = repeated_scales;
  } else {
    packed = at::stack({halves[0], halves[1]}, -1);
    if (tokens < kMinGroupTokens) {
      const auto repeats = (kMinGroupTokens + tokens - 1) / tokens;
      packed = packed.repeat({repeats, 1, 1}).narrow(0, 0, kMinGroupTokens);
      scales = scales.repeat({repeats}).narrow(0, 0, kMinGroupTokens);
    }
  }
  auto input = packed.view({padded_tokens, width});
  auto weights = weight_scale;
  if (!weight_is_interleaved) {
    auto scale_halves = weight_scale.reshape({width}).chunk(2, 0);
    weights = at::stack({scale_halves[0], scale_halves[1]}, -1).reshape({1, width});
  } else {
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == 1, "prepared weight scales must have shape [1, 2H]");
  }
  at::Tensor group;
  if (group_counts.has_value()) {
    const auto& counts = group_counts.value();
    TORCH_CHECK(counts.scalar_type() == at::kLong && counts.dim() == 1 && counts.numel() > padded_tokens,
                "group_counts must contain the prepared token-count range");
    TORCH_CHECK(counts.device() == x.device(), "group_counts must be on the input device");
    group = counts.narrow(0, padded_tokens, 1);
  } else {
    group = at::full({1}, padded_tokens, x.options().dtype(at::kLong));
  }
  auto y = at::empty({padded_tokens, half}, x.options().dtype(at::kChar));
  auto output_scale = at::empty({padded_tokens}, x.options().dtype(at::kFloat));
  at::Tensor none;
  auto* mode = const_cast<char*>("dynamic");
  auto* rounding = const_cast<char*>("rint");
  bool activate_left = true;
  int64_t dst_type = 2, activate_dim = -1, swiglu_mode = 1;
  double alpha = 1.0, bias = 0.0;
  EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, input, weights, scales, none, none, none, group, activate_left, mode,
               dst_type, rounding, activate_dim, swiglu_mode, clamp_limit, alpha, bias, y, output_scale);
  return {y.narrow(0, 0, tokens), output_scale.narrow(0, 0, tokens)};
}

}  // namespace vllm_ascend::official_shared_swiglu
