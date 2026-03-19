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
#ifndef MOE_COMBINE_NORMAL_TORCH_ADPT_H
#define MOE_COMBINE_NORMAL_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor combine_prefill(const at::Tensor& x, const at::Tensor& topk_idx, const at::Tensor& topk_weights,
                           const at::Tensor& src_idx, const at::Tensor& send_head, c10::string_view groupEp,
                           int64_t rank, int64_t num_ranks) {
    std::vector<char> group_ep_chrs(groupEp.begin(), groupEp.end());
    group_ep_chrs.push_back('\0');
    char* group_ep_ptr = &group_ep_chrs[0];

    TORCH_BIND_ASSERT(x.dim() == 2 and x.is_contiguous());
    at::Tensor recv_x = x;

    at::Tensor topk_idx_p = topk_idx;

    auto topk_idx_int32 = topk_idx_p.to(at::kInt);
    at::Tensor expand_ids = topk_idx_int32;
    at::Tensor token_src_info = src_idx;
    at::Tensor ep_send_counts = send_head;
    auto device = x.device();

    const int num_tokens = topk_idx_p.size(0);
    const int num_topk = topk_idx_p.size(1);

    int64_t hidden = static_cast<int>(recv_x.size(1));
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);
    int64_t global_bs = topk_idx_p.size(0) * num_ranks;

    // Combine data
    auto combined_x = torch::empty({topk_weights.size(0), hidden}, x.options());

    EXEC_NPU_CMD(aclnnMoeCombineNormal,
        recv_x,
        token_src_info,
        ep_send_counts,
        topk_weights,
        tp_send_counts,
        group_ep_ptr,
        num_ranks,
        rank,
        group_ep_ptr,
        tp_world_size,
        tp_rankId,
        moe_expert_number,
        global_bs,
        combined_x);

    return combined_x;
}

}

#endif