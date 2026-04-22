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
#ifndef MOE_INIT_ROUTING_CUSTOM_TORCH_ADPT_H
#define MOE_INIT_ROUTING_CUSTOM_TORCH_ADPT_H
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing_custom(
    const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type)
{
    constexpr int64_t DIM_X = 2;
    constexpr int64_t DIM_EXPERT_IDX = 2;
    constexpr int64_t LENGTH_ACTIVE_EXPERT_RANGE = 2;
    constexpr int64_t EXPERT_TOKENS_COUNT = 1;
    constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
    constexpr int64_t QUANT_MODE_UNQUANT = -1;
    constexpr int64_t QUANT_MODE_DYNAMIC_QUANT = 1;
    constexpr int64_t CUMSUM = 0;
    constexpr int64_t COUNT = 1;
    constexpr int64_t KEY_VALUE = 2;

    if (active_expert_range.empty()) {
        active_expert_range =  at::IntArrayRef({0, expert_num});
    }

    int64_t x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_X, "The x should be ", DIM_X, 
                "-Dimension, current is ", x_dim, "-Dimension.");

    int64_t expert_idx_dim = expert_idx.dim();
    TORCH_CHECK(expert_idx_dim == DIM_EXPERT_IDX, "The expert_idx should be ", DIM_EXPERT_IDX, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int64_t active_expert_range_length = active_expert_range.size();
    TORCH_CHECK(active_expert_range_length == LENGTH_ACTIVE_EXPERT_RANGE, "The active_expert_range should be ", LENGTH_ACTIVE_EXPERT_RANGE, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int expert_length = active_expert_range[1] - active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();

    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];
    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;

    if (drop_pad_mode == 1) { // Drop/Pad
        if (quant_mode == QUANT_MODE_UNQUANT) {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
        } else {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
        }
        expanded_scale_len = expert_num * expert_capacity;
    } else { // Dropless / Active
        if (active_num > 0) { // Active
            int64_t num_out_tokens = std::min((int64_t)bs * k, active_num);
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({num_out_tokens, h}, x.options());
            } else {
                expanded_x = at::empty({num_out_tokens, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = num_out_tokens;
        } else { // Dropless
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({bs * k, h}, x.options());
            } else {
                expanded_x = at::empty({bs * k, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = bs * k;
        }
    }

    at::Tensor expanded_row_idx = at::empty({bs * k}, expert_idx.options());
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum = at::empty({expert_length}, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum = at::empty({expert_num, 2}, x.options().dtype(at::kLong));
    }
    at::Tensor expanded_scale = at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMoeInitRoutingCustom,
                 x,
                 expert_idx,
                 scale,
                 offset,
                 active_num,
                 expert_capacity,
                 expert_num,
                 drop_pad_mode,
                 expert_tokens_num_type,
                 expert_tokens_num_flag,
                 quant_mode,
                 active_expert_range,
                 row_idx_type,
                 expanded_x,
                 expanded_row_idx,
                 expert_tokens_count_or_cumsum,
                 expanded_scale);
    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}
}
#endif