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
#ifndef DISPATCH_GMM_COMBINE_TORCH_ADPT_H
#define DISPATCH_GMM_COMBINE_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> dispatch_gmm_combine_decode(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const at::TensorList &gmm1_permuted_weight,
    const at::TensorList &gmm1_permuted_weight_scale,
    const at::TensorList &gmm2_weight,
    const at::TensorList &gmm2_weight_scale,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &expert_smooth_scales,
    const c10::optional<at::Tensor> &x_active_mask,
    c10::string_view group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs)
{
    auto x_shape = x.sizes();
    int bs = x_shape[0];
    int h = x_shape[1];

    at::Tensor output = at::empty({bs, h}, x.options());

    bool is_shared_expert = (ep_rank_id < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : moe_expert_num / (ep_rank_size - shared_expert_rank_num);
    auto opts = expert_ids.options().dtype(at::kLong);
    at::Tensor expert_token_nums = at::empty({num_local_experts}, opts);

    vector<char> group_ep_chrs(group_ep.begin(), group_ep.end());
    group_ep_chrs.push_back('\0');
    char *group_ep_ptr = &group_ep_chrs[0];
    EXEC_NPU_CMD(
        // op api
        aclnnDispatchGmmCombineDecode,
        // input tensors
        x,
        expert_ids,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        expert_scales,
        expert_smooth_scales,
        x_active_mask,
        //input attrs
        group_ep_ptr,
        ep_rank_size,
        ep_rank_id,
        moe_expert_num,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        // output tensors
        output,
        expert_token_nums);
    return {output, expert_token_nums};
}

}
#endif