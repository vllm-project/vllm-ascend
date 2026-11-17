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
 * WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DISPATCH_FFN_COMBINE_TORCH_ADPT_H
#define DISPATCH_FFN_COMBINE_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::TensorList& bias1,
    const at::TensorList& bias2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out,
    at::Tensor& expert_token_nums,
    const c10::optional<at::Tensor>& x_active_mask,
    double swiglu_limit,
    int64_t world_size,
    const c10::optional<at::Tensor>& expert_load_view,
    const c10::optional<at::Tensor>& record_enabled,
    int64_t local_expert_start
) {
    TORCH_CHECK(expert_load_view.has_value() == record_enabled.has_value(),
                "expert_load_view and record_enabled must be passed together");
    if (expert_load_view.has_value()) {
        TORCH_CHECK(expert_load_view->dim() == 1 && expert_load_view->scalar_type() == at::kInt &&
                        expert_load_view->is_contiguous() && expert_load_view->device().is_privateuseone(),
                    "expert_load_view must be a contiguous NPU int32 vector");
        TORCH_CHECK(record_enabled->dim() == 0 && record_enabled->scalar_type() == at::kBool &&
                        record_enabled->device() == expert_load_view->device(),
                    "record_enabled must be a NPU bool scalar");
        TORCH_CHECK(expert_token_nums.scalar_type() == at::kInt && expert_token_nums.is_contiguous() &&
                        expert_token_nums.device() == expert_load_view->device(),
                    "expert_token_nums must be a contiguous int32 tensor on the load device");
        TORCH_CHECK(local_expert_start >= 0 &&
                        local_expert_start + expert_token_nums.numel() <= expert_load_view->numel(),
                    "local expert range is outside expert_load_view");
    }
    char *group_ep_ptr = const_cast<char *>(group.data());
    bool is_int8 = weight1[0].dtype() == at::kChar;
    bool is_int4 = weight1[0].dtype() == at::kInt;
    if (is_int8) {
        EXEC_NPU_CMD(aclnnDispatchFFNCombine,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
                 group_ep_ptr,
                 max_output_size,
                 swiglu_limit,
                 world_size,
                 out,
                 expert_token_nums,
                 expert_load_view,
                 record_enabled,
                 local_expert_start);
    } else if (is_int4){
        EXEC_NPU_CMD(aclnnDispatchFFNCombineW4A8,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 bias1,
                 bias2,
                 probs,
                 x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
                 group_ep_ptr,
                 max_output_size,
                 swiglu_limit,
                 world_size,
                 out,
                 expert_token_nums,
                 expert_load_view,
                 record_enabled,
                 local_expert_start);
    } else {
        EXEC_NPU_CMD(aclnnDispatchFFNCombineBF16,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 group_ep_ptr,
                 max_output_size,
                 world_size,
                 out,
                 expert_token_nums,
                 expert_load_view,
                 record_enabled,
                 local_expert_start);
    }    
    return {out, expert_token_nums};
}
}
#endif
