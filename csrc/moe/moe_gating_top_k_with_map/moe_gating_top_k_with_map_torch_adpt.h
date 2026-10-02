/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MOE_GATING_TOP_K_WITH_MAP_TORCH_ADPT_H
#define MOE_GATING_TOP_K_WITH_MAP_TORCH_ADPT_H
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_with_map(
    const at::Tensor& x,
    const at::Tensor& log2phy,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt
    )
{
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());

    auto x_size = x.sizes();
    auto rows = x_size[0];
    auto expert_num = x_size[1];
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    if (bias.defined()) {
        TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "The dtype of x and bias should be same");
        TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
        auto bias_size = bias.sizes();
        TORCH_CHECK(bias_size[0] == expert_num, "The bias first dim should be same as x second dim");
    }
    TORCH_CHECK(log2phy.dim() == 1 && log2phy.numel() == expert_num,
                "log2phy must have shape [expert_count]");
    TORCH_CHECK(log2phy.scalar_type() == at::kInt, "log2phy must be int32");
    TORCH_CHECK(log2phy.device() == x.device(), "log2phy and x must be on the same device");
    TORCH_CHECK(log2phy.is_contiguous(), "log2phy must be contiguous");
    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    if (rows == 0) {
        return std::make_tuple(y, expert_idx, out);
    }
    EXEC_NPU_CMD(aclnnMoeGatingTopKWithMap,
                    x,
                    bias,
                    log2phy,
                    k,
                    k_group,
                    group_count,
                    group_select_mode,
                    renorm,
                    norm_type,
                    out_flag,
                    routed_scaling_factor,
                    eps,
                    y,
                    expert_idx,
                    out
                );

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y,expert_idx,out);
}

}
#endif
