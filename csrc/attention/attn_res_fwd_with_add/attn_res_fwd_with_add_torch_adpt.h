// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor> attn_res_fwd_with_add(
    const at::Tensor& prefix_sum, const at::Tensor& addend, const at::Tensor& block_residual,
    const at::Tensor& proj_weight, const at::Tensor& norm_weight, double norm_eps)
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

    TORCH_CHECK(addend.sizes() == prefix_sum.sizes() &&
                    addend.scalar_type() == prefix_sum.scalar_type() && addend.device() == prefix_sum.device(),
                "attn_res_fwd_with_add: addend must match prefix_sum shape, dtype and device.");
    auto output = at::empty_like(prefix_sum);
    auto prefix_out = at::empty_like(prefix_sum);
    EXEC_NPU_CMD(aclnnAttnResFwdWithAdd, prefix_sum, block_residual, proj_weight, norm_weight,
                 addend, norm_eps, output, prefix_out);
    return {output, prefix_out};
}
}
