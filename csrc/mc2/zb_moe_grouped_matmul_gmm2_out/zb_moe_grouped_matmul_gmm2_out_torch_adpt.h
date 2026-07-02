/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef ZB_MOE_GROUPED_MATMUL_GMM2_OUT_TORCH_ADPT_H
#define ZB_MOE_GROUPED_MATMUL_GMM2_OUT_TORCH_ADPT_H

namespace vllm_ascend {

// ZB-only grouped matmul: write gmm2 output into caller-provided SHMEM ``out``.
// Uses CANN ``aclnnGroupedMatmulWeightNz`` (NZ weight, W8A8/per-token quant path).
at::Tensor &zb_moe_grouped_matmul_gmm2_out(
    const at::Tensor &x,
    const at::TensorList &weight,
    const c10::optional<at::TensorList> &scale,
    const c10::optional<at::TensorList> &per_token_scale,
    const c10::optional<at::TensorList> &bias,
    const at::Tensor &group_list,
    at::Tensor &out,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type,
    int64_t act_type)
{
    TORCH_CHECK(out.defined(), "zb_moe_grouped_matmul_gmm2_out: out must be defined");
    TORCH_CHECK(out.is_privateuseone(), "zb_moe_grouped_matmul_gmm2_out: out must be on NPU");
    TORCH_CHECK(x.is_privateuseone(), "zb_moe_grouped_matmul_gmm2_out: x must be on NPU");
    TORCH_CHECK(weight.size() > 0, "zb_moe_grouped_matmul_gmm2_out: weight must not be empty");

    std::vector<at::Tensor> x_vec = {x};
    std::vector<at::Tensor> y_vec = {out};
    at::TensorList x_list(x_vec);
    at::TensorList y_list(y_vec);

    const c10::optional<at::TensorList> empty_list;
    const c10::optional<at::IntArrayRef> empty_int_array;
    const bool use_quant = scale.has_value() && scale.value().size() > 0;

    if (use_quant) {
        const int64_t quant_group_size = 0;
        EXEC_NPU_CMD(aclnnGroupedMatmulWeightNz,
                     x_list,
                     weight,
                     bias,
                     scale,
                     empty_list,
                     empty_list,
                     empty_list,
                     per_token_scale,
                     group_list,
                     empty_list,
                     empty_list,
                     empty_list,
                     split_item,
                     group_type,
                     group_list_type,
                     act_type,
                     empty_int_array,
                     quant_group_size,
                     y_list,
                     empty_list,
                     empty_list);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulV4,
                     x_list,
                     weight,
                     bias,
                     empty_list,
                     empty_list,
                     empty_list,
                     empty_list,
                     per_token_scale,
                     group_list,
                     empty_list,
                     empty_list,
                     empty_list,
                     split_item,
                     group_type,
                     group_list_type,
                     act_type,
                     y_list,
                     empty_list,
                     empty_list);
    }

    return out;
}

}  // namespace vllm_ascend

#endif
