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
#ifndef DISPATCH_LAYOUT_TORCH_ADPT_H
#define DISPATCH_LAYOUT_TORCH_ADPT_H

namespace vllm_ascend {
const int LOCAL_RANK_SIZE = 8;
const int MAX_BATCH_SIZE = 4096;
const int EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE;  // 4097

std::tuple<at::Tensor, at::Tensor> get_dispatch_layout(
    const at::Tensor& topk_idx, int64_t num_experts, int64_t num_ranks) {
    // Convert topk_idx to int64 if necessary
    at::Tensor topk_idx_int64 = topk_idx.scalar_type() == at::kLong 
        ? topk_idx 
        : topk_idx.to(at::kLong);
    
    TORCH_BIND_ASSERT(topk_idx_int64.dim() == 2);
    TORCH_BIND_ASSERT(topk_idx_int64.is_contiguous());
    TORCH_BIND_ASSERT(num_experts > 0);

    const int num_tokens = topk_idx_int64.size(0);
    const int num_topk = topk_idx_int64.size(1);
    const int local_ranksize = LOCAL_RANK_SIZE;
    auto server_num = num_ranks / local_ranksize;

    auto device = topk_idx_int64.device();
    auto num_tokens_per_expert = at::zeros({num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));
    const int notify_send_data_size =
        num_experts * EXPERT_DATA_SIZE + server_num + MAX_BATCH_SIZE * (1 + 2 * server_num + num_experts);
    auto send_token_idx_small = at::zeros({num_tokens, num_topk}, at::dtype(at::kInt).device(device));
    auto notify_send_data = at::zeros({notify_send_data_size}, at::dtype(at::kInt).device(device));
    EXEC_NPU_CMD(aclnnDispatchLayout,
        topk_idx_int64,
        num_tokens,
        num_ranks,
        num_experts,
        num_topk,
        local_ranksize,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        notify_send_data,
        send_token_idx_small);

    return std::make_tuple(num_tokens_per_expert, send_token_idx_small);
}

}
#endif