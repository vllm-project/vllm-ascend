/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef ATTN_RES_FWD_TORCH_ADPT_H
#define ATTN_RES_FWD_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor attn_res_fwd(const at::Tensor& prefix_sum,
                        const at::Tensor& block_residual,
                        const at::Tensor& proj_weight,
                        const at::Tensor& norm_weight,
                        double norm_eps)
{
    TORCH_CHECK(prefix_sum.dim() == 2, "attn_res_fwd: prefix_sum must be 2D.");
    TORCH_CHECK(block_residual.dim() == 3, "attn_res_fwd: block_residual must be 3D.");
    TORCH_CHECK(prefix_sum.size(0) == block_residual.size(0) &&
                    prefix_sum.size(1) == block_residual.size(2),
                "attn_res_fwd: prefix_sum and block_residual shapes do not match.");
    TORCH_CHECK(proj_weight.dim() == 2 && proj_weight.size(0) == 1 &&
                    proj_weight.size(1) == prefix_sum.size(1),
                "attn_res_fwd: proj_weight must have shape [1, hidden_size].");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == prefix_sum.size(1),
                "attn_res_fwd: norm_weight must have shape [hidden_size].");
    TORCH_CHECK(prefix_sum.scalar_type() == at::kBFloat16 &&
                    block_residual.scalar_type() == at::kBFloat16 &&
                    proj_weight.scalar_type() == at::kBFloat16 &&
                    norm_weight.scalar_type() == at::kBFloat16,
                "attn_res_fwd: all inputs must be bfloat16.");
    TORCH_CHECK(block_residual.device() == prefix_sum.device() &&
                    proj_weight.device() == prefix_sum.device() &&
                    norm_weight.device() == prefix_sum.device(),
                "attn_res_fwd: all inputs must be on the same device.");

    at::Tensor hidden_states = at::empty_like(prefix_sum);
    at::Tensor inv_rms;
    at::Tensor probs;
    bool need_backward = false;
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

#endif
