/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUSED_GDN_DECODE_TORCH_ADPT_H
#define FUSED_GDN_DECODE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_fused_gdn_decode(
    const at::Tensor& mixed_qkv,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    at::Tensor& state,
    const at::Tensor& ssm_state_indices,
    double scale,
    double softplus_threshold)
{
    TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must be [B, packed_dim]");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be [B, HV]");
    TORCH_CHECK(A_log.dim() == 1 && dt_bias.dim() == 1, "A_log and dt_bias must be [HV]");
    TORCH_CHECK(state.dim() == 4, "state must be [slots, HV, V, K]");
    TORCH_CHECK(ssm_state_indices.dim() == 1, "ssm_state_indices must be [B]");
    TORCH_CHECK(mixed_qkv.size(0) == a.size(0) && a.sizes() == b.sizes(),
                "mixed_qkv/a/b batch or shape mismatch.");
    TORCH_CHECK(a.size(1) == state.size(1), "a/b HV must equal state.shape[1].");
    TORCH_CHECK(A_log.size(0) == state.size(1) && dt_bias.size(0) == state.size(1),
                "A_log/dt_bias must have HV elements.");

    at::Tensor out = at::empty({mixed_qkv.size(0), 1, state.size(1), state.size(2)}, mixed_qkv.options());
    float scale_value = static_cast<float>(scale);
    float softplus_threshold_value = static_cast<float>(softplus_threshold);
    int64_t hv_value = state.size(1);
    int64_t v_value = state.size(2);
    int64_t k_value = state.size(3);
    int64_t qk_dim = mixed_qkv.size(1) - hv_value * v_value;
    TORCH_CHECK(qk_dim > 0 && qk_dim % (2 * k_value) == 0, "mixed_qkv packed_dim is inconsistent with state shape.");
    int64_t h_value = qk_dim / (2 * k_value);
    TORCH_CHECK(h_value > 0 && hv_value % h_value == 0, "invalid H/HV relation derived from mixed_qkv/state.");
    EXEC_NPU_CMD(aclnnFusedGdnDecode,
                 mixed_qkv,
                 a,
                 b,
                 A_log,
                 dt_bias,
                 state,
                 ssm_state_indices,
                 scale_value,
                 softplus_threshold_value,
                 out);
    return out;
}

} // namespace vllm_ascend

#endif // FUSED_GDN_DECODE_TORCH_ADPT_H
