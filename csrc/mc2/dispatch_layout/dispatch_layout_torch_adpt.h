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
std::tuple<at::Tensor, at::Tensor, at::Tensor> get_dispatch_layout(const at::Tensor& topk_idx, int64_t num_experts,
                                                                   int64_t num_ranks) {
    TORCH_BIND_ASSERT(topk_idx.dim() == 2);
    TORCH_BIND_ASSERT(topk_idx.is_contiguous());
    TORCH_BIND_ASSERT(num_experts > 0);

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);

    auto device = topk_idx.device();
    auto num_tokens_per_expert = at::zeros({num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnDispatchLayout,
        topk_idx,
        num_tokens,
        num_ranks,
        num_experts,
        num_topk,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank);

    auto is_token_in_rank_bool = is_token_in_rank.to(at::kBool);

    return std::make_tuple(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank_bool);
}

}
#endif