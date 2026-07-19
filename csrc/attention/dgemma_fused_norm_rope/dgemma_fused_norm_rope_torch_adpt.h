/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef DGEMMA_FUSED_NORM_ROPE_TORCH_ADPT_H
#define DGEMMA_FUSED_NORM_ROPE_TORCH_ADPT_H
#include <tuple>
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dgemma_fused_norm_rope(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    double epsilon = 1e-6)
{
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                "q/k/v must share dtype");
    TORCH_CHECK(cos.scalar_type() == c10::kFloat && sin.scalar_type() == c10::kFloat,
                "cos/sin must be float32");
    at::Tensor q_out = at::empty_like(q);
    at::Tensor k_out = at::empty_like(k);
    at::Tensor v_out = at::empty_like(v);
    float eps = static_cast<float>(epsilon);
    EXEC_NPU_CMD(aclnnDgemmaFusedNormRope,
                 q, k, v, q_weight, k_weight, cos, sin,
                 eps, num_q_heads, num_kv_heads, head_dim,
                 q_out, k_out, v_out);
    return std::make_tuple(q_out, k_out, v_out);
}
} // namespace vllm_ascend
#endif
