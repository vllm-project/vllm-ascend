/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TORCH_ADPT_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TORCH_ADPT_H
#include <tuple>
namespace vllm_ascend {
// hidden[T, hidden_size], wqkv[qkv_out, hidden_size] (row-major, applied as hidden@wqkv.T)
// qkv_scratch[T, qkv_out]: shape/template buffer for the graph input contract.
// qkv_scratch_out is a real output tensor so graph replay can track the GEMM intermediate.
// -> q[T, num_q_heads, head_dim], k/v[T, num_kv_heads, head_dim]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dgemma_fused_qkv_proj_norm_rope(
    const at::Tensor& hidden, const at::Tensor& wqkv,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& qkv_scratch,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim, int64_t hidden_size,
    int64_t sync_base = 0, double epsilon = 1e-6)
{
    TORCH_CHECK(hidden.scalar_type() == wqkv.scalar_type(), "hidden/wqkv must share dtype");
    TORCH_CHECK(cos.scalar_type() == c10::kFloat && sin.scalar_type() == c10::kFloat,
                "cos/sin must be float32");
    int64_t T = hidden.size(0);
    auto opts = hidden.options();
    bool scratch_only = (static_cast<uint64_t>(sync_base) & 0x100ULL) != 0ULL;
    at::Tensor qkv_scratch_out = at::empty_like(qkv_scratch);
    at::Tensor q_out = scratch_only ? at::empty({1, 1, 1}, opts)
                                    : at::empty({T, num_q_heads, head_dim}, opts);
    at::Tensor k_out = scratch_only ? at::empty({1, 1, 1}, opts)
                                    : at::empty({T, num_kv_heads, head_dim}, opts);
    at::Tensor v_out = scratch_only ? at::empty({1, 1, 1}, opts)
                                    : at::empty({T, num_kv_heads, head_dim}, opts);
    float eps = static_cast<float>(epsilon);
    EXEC_NPU_CMD(aclnnDgemmaFusedQkvProjNormRope,
                 hidden, wqkv, q_weight, k_weight, cos, sin, qkv_scratch,
                 eps, num_q_heads, num_kv_heads, head_dim, hidden_size,
                 sync_base,
                 qkv_scratch_out, q_out, k_out, v_out);
    return std::make_tuple(q_out, k_out, v_out, qkv_scratch_out);
}
} // namespace vllm_ascend
#endif
