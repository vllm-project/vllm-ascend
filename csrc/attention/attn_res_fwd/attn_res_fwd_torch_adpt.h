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

#ifndef ATTN_RES_FWD_TORCH_ADPT_H
#define ATTN_RES_FWD_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor attn_res_fwd(
    const at::Tensor &prefix_sum,
    const at::Tensor &block_residual,
    const at::Tensor &proj_weight,
    const at::Tensor &norm_weight,
    double norm_eps)
{
    TORCH_CHECK(prefix_sum.dim() == 2,
                "attn_res_fwd: prefix_sum must have shape [tokens, hidden_size].");
    TORCH_CHECK(block_residual.dim() == 3,
                "attn_res_fwd: block_residual must have shape [tokens, blocks, hidden_size].");
    TORCH_CHECK(prefix_sum.size(0) == block_residual.size(0) &&
                    prefix_sum.size(1) == block_residual.size(2),
                "attn_res_fwd: prefix_sum and block_residual shapes are incompatible.");

    at::Tensor hidden_states = at::empty_like(prefix_sum);
    at::Tensor inv_rms = at::empty({0}, prefix_sum.options().dtype(at::kFloat));
    at::Tensor probs = at::empty({0}, prefix_sum.options().dtype(at::kFloat));
    constexpr bool need_backward = false;

    EXEC_NPU_CMD(aclnnAttnResFwd,
                 prefix_sum,
                 block_residual,
                 proj_weight,
                 norm_weight,
                 norm_eps,
                 need_backward,
                 hidden_states,
                 inv_rms,
                 probs);
    return hidden_states;
}

} // namespace vllm_ascend

#endif // ATTN_RES_FWD_TORCH_ADPT_H
