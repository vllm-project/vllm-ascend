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
 * WITHOUT WARRANTIES OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ZB_MOE_DISTRIBUTE_COMBINE_TORCH_ADPT_H
#define ZB_MOE_DISTRIBUTE_COMBINE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor &zb_moe_distribute_combine(
    const at::Tensor &expand_x,
    const at::Tensor &expert_ids,
    const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_count,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &tp_send_count,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &activation_scale,
    const c10::optional<at::Tensor> &weight_scale,
    const c10::optional<at::Tensor> &group_list,
    const c10::optional<at::Tensor> &expand_scales,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &elastic_info,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha1,
    const c10::optional<at::Tensor> &const_expert_alpha2,
    const c10::optional<at::Tensor> &const_expert_v,
    int64_t ep_world_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t tp_world_size,
    int64_t tp_rank_id,
    int64_t expert_shard_type,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t global_bs,
    int64_t out_dtype,
    int64_t comm_quant_mode,
    int64_t ext_info,
    int64_t group_list_type,
    c10::string_view comm_alg,
    int64_t zero_expert_num,
    int64_t copy_expert_num,
    int64_t const_expert_num,
    at::Tensor &combined_x)
{
    std::string comm_alg_str(comm_alg.data(), comm_alg.size());
    char *comm_alg_ptr = comm_alg_str.empty() ? nullptr : const_cast<char *>(comm_alg_str.c_str());

    EXEC_NPU_CMD(aclnnZbMoeDistributeCombine,
                 expand_x,
                 expert_ids,
                 assist_info_for_combine,
                 ep_send_count,
                 expert_scales,
                 tp_send_count.has_value() ? tp_send_count.value() : at::Tensor(),
                 x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
                 activation_scale.has_value() ? activation_scale.value() : at::Tensor(),
                 weight_scale.has_value() ? weight_scale.value() : at::Tensor(),
                 group_list.has_value() ? group_list.value() : at::Tensor(),
                 expand_scales.has_value() ? expand_scales.value() : at::Tensor(),
                 shared_expert_x.has_value() ? shared_expert_x.value() : at::Tensor(),
                 elastic_info.has_value() ? elastic_info.value() : at::Tensor(),
                 ori_x.has_value() ? ori_x.value() : at::Tensor(),
                 const_expert_alpha1.has_value() ? const_expert_alpha1.value() : at::Tensor(),
                 const_expert_alpha2.has_value() ? const_expert_alpha2.value() : at::Tensor(),
                 const_expert_v.has_value() ? const_expert_v.value() : at::Tensor(),
                 ep_world_size,
                 ep_rank_id,
                 moe_expert_num,
                 tp_world_size,
                 tp_rank_id,
                 expert_shard_type,
                 shared_expert_num,
                 shared_expert_rank_num,
                 global_bs,
                 out_dtype,
                 comm_quant_mode,
                 ext_info,
                 group_list_type,
                 comm_alg_ptr,
                 zero_expert_num,
                 copy_expert_num,
                 const_expert_num,
                 combined_x);

    return combined_x;
}

}  // namespace vllm_ascend

#endif  // ZB_MOE_DISTRIBUTE_COMBINE_TORCH_ADPT_H
