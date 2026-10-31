// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#pragma once

#include <ATen/ATen.h>

namespace vllm_ascend::official_shared_swiglu {

inline std::tuple<at::Tensor, at::Tensor> run_meta(const at::Tensor& x, const at::Tensor& weight_scale,
                                                   const at::Tensor& activation_scale, double clamp_limit,
                                                   const c10::optional<at::Tensor>& group_counts,
                                                   bool weight_is_interleaved) {
  return {at::empty_symint({x.sym_size(0), x.sym_size(1) / c10::SymInt(2)}, x.options().dtype(at::kChar)),
          at::empty_symint({x.sym_size(0)}, x.options().dtype(at::kFloat))};
}

}  // namespace vllm_ascend::official_shared_swiglu
