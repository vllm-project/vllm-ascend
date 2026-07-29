/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef RECURRENT_KDA_TORCH_ADPT_H
#define RECURRENT_KDA_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor recurrent_kda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& gate,
    const at::Tensor& beta,
    at::Tensor& initial_state,
    const at::Tensor& cu_seqlens,
    const at::Tensor& ssm_state_indices,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    double scale,
    bool use_qk_l2norm_in_kernel,
    bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    bool safe_gate,
    double lower_bound)
{
    const bool is_tnd = query.dim() == 3;
    TORCH_CHECK((is_tnd && key.dim() == 3 && value.dim() == 3 && gate.dim() == 3 && beta.dim() == 2) ||
                    (!is_tnd && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
                     gate.dim() == 4 && beta.dim() == 3),
                "recurrent_kda: TND expects q/k [T,H,K], v [T,HV,V], gate [T,HV,K], beta [T,HV]; "
                "BSND expects q/k [B,T,H,K], v [B,T,HV,V], gate [B,T,HV,K], beta [B,T,HV].");
    TORCH_CHECK(query.sizes() == key.sizes(),
                "recurrent_kda: query and key must have identical shapes.");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                    key.scalar_type() == at::kBFloat16 &&
                    value.scalar_type() == at::kBFloat16,
                "recurrent_kda: query/key/value must be bfloat16.");
    TORCH_CHECK((gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kBFloat16 ||
                 gate.scalar_type() == at::kHalf) &&
                    (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16 ||
                     beta.scalar_type() == at::kHalf),
                "recurrent_kda: gate and beta must be float32, bfloat16 or float16.");
    TORCH_CHECK(key.device() == query.device() && value.device() == query.device() &&
                    gate.device() == query.device() && beta.device() == query.device() &&
                    initial_state.device() == query.device(),
                "recurrent_kda: query/key/value/gate/beta/state must be on the same device.");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                "recurrent_kda: cu_seqlens must be a 1D device tensor with at least two elements.");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt || cu_seqlens.scalar_type() == at::kLong,
                "recurrent_kda: cu_seqlens must be int32 or int64.");
    TORCH_CHECK(cu_seqlens.device() == query.device(),
                "recurrent_kda: cu_seqlens must be on the same device as query.");

    const int64_t batch = is_tnd ? 1 : query.size(0);
    const int64_t total_tokens = is_tnd ? query.size(0) : query.size(0) * query.size(1);
    const int64_t seq_num = cu_seqlens.size(0) - 1;
    const int64_t h = is_tnd ? query.size(1) : query.size(2);
    const int64_t k_dim = is_tnd ? query.size(2) : query.size(3);
    const int64_t hv = is_tnd ? value.size(1) : value.size(2);
    const int64_t v_dim = is_tnd ? value.size(2) : value.size(3);
    TORCH_CHECK(total_tokens > 0 && h > 0 && hv > 0,
                "recurrent_kda: token and head dimensions must be positive.");
    TORCH_CHECK(hv % h == 0,
                "recurrent_kda: HV must be divisible by H.");
    TORCH_CHECK(k_dim == 128 && v_dim == 128,
                "recurrent_kda: the Kimi K3 integration requires K=V=128.");
    TORCH_CHECK((is_tnd && value.size(0) == total_tokens && gate.size(0) == total_tokens &&
                 beta.size(0) == total_tokens && gate.size(1) == hv && gate.size(2) == k_dim &&
                 beta.size(1) == hv) ||
                    (!is_tnd && value.size(0) == batch && value.size(1) == query.size(1) &&
                     gate.size(0) == batch && gate.size(1) == query.size(1) && gate.size(2) == hv &&
                     gate.size(3) == k_dim && beta.size(0) == batch && beta.size(1) == query.size(1) &&
                     beta.size(2) == hv),
                "recurrent_kda: value/gate/beta shapes do not match the selected layout.");
    const bool packed_indices = ssm_state_indices.dim() == 1 &&
                                ssm_state_indices.numel() >= total_tokens;
    const bool speculative_indices = ssm_state_indices.dim() == 2 &&
                                     ssm_state_indices.size(0) == seq_num &&
                                     ssm_state_indices.size(1) > 0;
    TORCH_CHECK((ssm_state_indices.scalar_type() == at::kInt ||
                 ssm_state_indices.scalar_type() == at::kLong) &&
                    (packed_indices || speculative_indices),
                "recurrent_kda: ssm_state_indices must be int32/int64 packed [T] or "
                "speculative [seq_num,max_step].");
    TORCH_CHECK(ssm_state_indices.device() == query.device(),
                "recurrent_kda: ssm_state_indices must be on the same device as query.");
    TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(0) >= 1 &&
                    initial_state.size(1) == hv && initial_state.size(2) == v_dim &&
                    initial_state.size(3) == k_dim,
                "recurrent_kda: initial_state must be a non-empty [state_capacity,HV,V,K] pool.");
    TORCH_CHECK(initial_state.scalar_type() == at::kFloat || initial_state.scalar_type() == at::kBFloat16,
                "recurrent_kda: initial_state must be float32 or bfloat16.");
    TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.dim() == 1 && a_log.numel() == hv,
                "recurrent_kda: A_log must be float32 [HV].");
    TORCH_CHECK(dt_bias.scalar_type() == at::kFloat &&
                    ((dt_bias.dim() == 1 && dt_bias.numel() == hv * k_dim) ||
                     (dt_bias.dim() == 2 && dt_bias.size(0) == hv && dt_bias.size(1) == k_dim)),
                "recurrent_kda: dt_bias must be float32 [HV*K] or [HV,K].");
    TORCH_CHECK(a_log.device() == query.device() && dt_bias.device() == query.device(),
                "recurrent_kda: A_log and dt_bias must be on the same device as query.");
    if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
        TORCH_CHECK(num_accepted_tokens->dim() == 1 && num_accepted_tokens->size(0) == seq_num &&
                        (num_accepted_tokens->scalar_type() == at::kInt ||
                         num_accepted_tokens->scalar_type() == at::kLong),
                    "recurrent_kda: num_accepted_tokens must be int32/int64 [seq_num].");
        TORCH_CHECK(num_accepted_tokens->device() == query.device(),
                    "recurrent_kda: num_accepted_tokens must be on the same device as query.");
    }
    TORCH_CHECK(!safe_gate || (lower_bound >= -5.0 && lower_bound < 0.0),
                "recurrent_kda: lower_bound must be in [-5,0) for safe gate.");

    at::Tensor output = at::empty_like(value);
    at::Tensor final_state = initial_state;
    const at::Tensor& accepted = c10::value_or_else(
        num_accepted_tokens, [] { return at::Tensor(); });
    const char* layout = is_tnd ? "TND" : "BSND";
    bool output_final_state = true;
    bool state_v_first = true;
    EXEC_NPU_CMD(
        aclnnRecurrentKda,
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        a_log,
        dt_bias,
        accepted,
        layout,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        lower_bound,
        state_v_first,
        output);
    return output;
}

} // namespace vllm_ascend
#endif
