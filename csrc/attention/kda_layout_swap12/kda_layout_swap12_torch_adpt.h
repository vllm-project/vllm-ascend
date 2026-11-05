/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef KDA_LAYOUT_SWAP12_TORCH_ADPT_H
#define KDA_LAYOUT_SWAP12_TORCH_ADPT_H

#include <vector>

namespace vllm_ascend {

at::Tensor kda_layout_swap12(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &dependency)
{
    TORCH_CHECK(x.dim() >= 3, "kda_layout_swap12: x must have rank >= 3.");
    auto dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "kda_layout_swap12: x must be float32, float16 or bfloat16.");

    std::vector<int64_t> y_sizes(x.sizes().begin(), x.sizes().end());
    // Swap logical axes 1 and 2 for rank-4 tensors. For the rank-3 TND/NTD
    // layouts, the equivalent operation swaps axes 0 and 1.
    if (x.dim() == 3) {
        std::swap(y_sizes[0], y_sizes[1]);
    } else {
        std::swap(y_sizes[1], y_sizes[2]);
    }
    at::Tensor y = at::empty(y_sizes, x.options());
    at::Tensor dependency_tensor = dependency.value_or(at::Tensor());
    if (dependency_tensor.defined()) {
        TORCH_CHECK(dependency_tensor.sizes() == y.sizes(),
                    "kda_layout_swap12: dependency must have the same shape as output.");
    }

    EXEC_NPU_CMD(aclnnKdaLayoutSwap12, x, dependency_tensor, y);
    return y;
}

} // namespace vllm_ascend

#endif
