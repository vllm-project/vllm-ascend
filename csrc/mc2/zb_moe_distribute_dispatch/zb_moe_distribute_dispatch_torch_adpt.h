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
#ifndef ZB_MOE_DISTRIBUTE_DISPATCH_TORCH_ADPT_H
#define ZB_MOE_DISTRIBUTE_DISPATCH_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>
zb_moe_distribute_dispatch(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &elastic_info,
    int64_t ep_world_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t tp_world_size,
    int64_t tp_rank_id,
    int64_t expert_shard_type,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs,
    int64_t expert_token_nums_type,
    int64_t ext_info,
    c10::string_view comm_alg,
    int64_t zero_expert_num,
    int64_t copy_expert_num,
    int64_t const_expert_num,
    at::Tensor &expand_x_out,
    at::Tensor &dynamic_scales_out,
    at::Tensor &assist_info_for_combine_out,
    at::Tensor &expert_token_nums_out,
    at::Tensor &ep_recv_count_out,
    at::Tensor &tp_recv_count_out)
{
    std::string comm_alg_str(comm_alg.data(), comm_alg.size());
    char *comm_alg_ptr = comm_alg_str.empty() ? nullptr : const_cast<char *>(comm_alg_str.c_str());

    EXEC_NPU_CMD(aclnnZbMoeDistributeDispatch,
                 x,
                 expert_ids,
                 scales.has_value() ? scales.value() : at::Tensor(),
                 x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
                 elastic_info.has_value() ? elastic_info.value() : at::Tensor(),
                 ep_world_size,
                 ep_rank_id,
                 moe_expert_num,
                 tp_world_size,
                 tp_rank_id,
                 expert_shard_type,
                 shared_expert_num,
                 shared_expert_rank_num,
                 quant_mode,
                 global_bs,
                 expert_token_nums_type,
                 ext_info,
                 comm_alg_ptr,
                 zero_expert_num,
                 copy_expert_num,
                 const_expert_num,
                 expand_x_out,
                 dynamic_scales_out,
                 assist_info_for_combine_out,
                 expert_token_nums_out,
                 ep_recv_count_out,
                 tp_recv_count_out);

    return {expand_x_out, dynamic_scales_out, assist_info_for_combine_out,
            expert_token_nums_out, ep_recv_count_out, tp_recv_count_out};
}

}  // namespace vllm_ascend

#endif  // ZB_MOE_DISTRIBUTE_DISPATCH_TORCH_ADPT_H
