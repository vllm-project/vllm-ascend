// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#ifndef FUSED_REARRANGE_MIX_QKV_GDN_GATING_TORCH_ADPT_H
#define FUSED_REARRANGE_MIX_QKV_GDN_GATING_TORCH_ADPT_H

#include <cstdint>
#include <tuple>

namespace vllm_ascend {

inline void check_fused_rearrange_mix_qkv_gdn_gating_inputs(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    int64_t q_dim,
    int64_t k_dim,
    int64_t v_dim)
{
    constexpr int64_t elements_per_datablock = 16;
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kBFloat16 &&
                    b.scalar_type() == at::kBFloat16,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires bf16 x/a/b tensors");
    const bool same_supported_dtype =
        a_log.scalar_type() == dt_bias.scalar_type() &&
        (a_log.scalar_type() == at::kFloat || a_log.scalar_type() == at::kBFloat16 ||
         a_log.scalar_type() == at::kHalf);
    const bool model_mixed_dtype =
        a_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kBFloat16;
    TORCH_CHECK(same_supported_dtype || model_mixed_dtype,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires matching fp32/bf16/fp16 A_log/dt_bias "
                "or fp32 A_log with bf16 dt_bias");
    TORCH_CHECK(x.dim() == 2, "npu_fused_rearrange_mix_qkv_gdn_gating requires [T, Q + K + V] input");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires [T, num_heads] a/b inputs");
    TORCH_CHECK(a.sizes() == b.sizes(), "npu_fused_rearrange_mix_qkv_gdn_gating requires a and b to match");
    TORCH_CHECK(a.size(0) == x.size(0),
                "npu_fused_rearrange_mix_qkv_gdn_gating requires a to have one row per token");
    TORCH_CHECK(a_log.dim() == 1 && dt_bias.dim() == 1 && a_log.numel() == a.size(1) &&
                    dt_bias.numel() == a.size(1),
                "npu_fused_rearrange_mix_qkv_gdn_gating requires [num_heads] A_log and dt_bias");
    TORCH_CHECK(a.size(1) > 0 && a.size(1) <= 4096,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires a head count in [1, 4096]");
    const int64_t row_dim = x.size(1);
    TORCH_CHECK(q_dim > 0 && k_dim > 0 && v_dim > 0 && q_dim + k_dim + v_dim == row_dim,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires [T, Q + K + V] input");
    TORCH_CHECK(q_dim % elements_per_datablock == 0 && k_dim % elements_per_datablock == 0 &&
                    v_dim % elements_per_datablock == 0,
                "npu_fused_rearrange_mix_qkv_gdn_gating requires 32-byte-aligned Q/K/V widths");
    TORCH_CHECK(x.is_contiguous() && a.is_contiguous() && b.is_contiguous() && a_log.is_contiguous() &&
                    dt_bias.is_contiguous(),
                "npu_fused_rearrange_mix_qkv_gdn_gating requires contiguous inputs");
    TORCH_CHECK(x.device() == a.device() && x.device() == b.device() && x.device() == a_log.device() &&
                    x.device() == dt_bias.device(),
                "npu_fused_rearrange_mix_qkv_gdn_gating requires all inputs on the same device");
}

/**
 * Fused GDN prologue: rearrange the mixed QKV inside the cube cores while the
 * vector cores compute the gating.
 *
 * Returns the packed QKV buffer plus g (fp32) and beta (bf16), both laid out as
 * [tokens, num_heads] so that the caller can view them as [1, tokens, heads].
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_rearrange_mix_qkv_gdn_gating(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    int64_t q_dim,
    int64_t k_dim,
    int64_t v_dim,
    double beta,
    double threshold)
{
    check_fused_rearrange_mix_qkv_gdn_gating_inputs(x, a, b, a_log, dt_bias, q_dim, k_dim, v_dim);

    const int64_t tokens = x.size(0);
    const int64_t num_heads = a.size(1);

    const c10_npu::OptionalNPUGuard guard(x.device());
    at::Tensor packed_qkv = at::empty({x.numel()}, x.options());
    at::Tensor g = at::empty({tokens, num_heads}, x.options().dtype(at::kFloat));
    at::Tensor beta_output = at::empty({tokens, num_heads}, x.options());
    if (tokens == 0) {
        return std::make_tuple(packed_qkv, g, beta_output);
    }
    EXEC_NPU_CMD(
        aclnnFusedRearrangeMixQkvGdnGating,
        x,
        a,
        b,
        a_log,
        dt_bias,
        q_dim,
        k_dim,
        v_dim,
        beta,
        threshold,
        packed_qkv,
        g,
        beta_output);
    return std::make_tuple(packed_qkv, g, beta_output);
}

}  // namespace vllm_ascend

#endif  // FUSED_REARRANGE_MIX_QKV_GDN_GATING_TORCH_ADPT_H
