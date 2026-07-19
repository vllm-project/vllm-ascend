/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef DGEMMA_FUSED_ROUTER_FRONT_TORCH_ADPT_H
#define DGEMMA_FUSED_ROUTER_FRONT_TORCH_ADPT_H
#include <tuple>
namespace vllm_ascend {
// x[T, hidden_size], scale[hidden_size], proj_weight[num_experts, hidden_size]
//   (row-major, applied as normed_x @ proj_weight.T)
// norm_scratch[T, hidden_size] and fp32 logits_scratch[T, num_experts] are
// caller-provided persistent intermediates (graph-capture safe).
// sync_scratch[128] is persistent int32 GM handoff state for stale-event defense.
// -> topk_weights[T, top_k] fp32, topk_ids[T, top_k] int32
// MIX kernel: AIV norm -> AIC GEMM logits -> AIV topk/renorm/per-expert scale.
std::tuple<at::Tensor, at::Tensor> npu_dgemma_fused_router_front(
    const at::Tensor& x, const at::Tensor& scale, const at::Tensor& proj_weight,
    const at::Tensor& norm_scratch, const at::Tensor& logits_scratch,
    const at::Tensor& per_expert_scale, const at::Tensor& sync_scratch,
    int64_t hidden_size, int64_t num_experts, int64_t top_k,
    int64_t sync_base = 1, double epsilon = 1e-6)
{
    TORCH_CHECK(x.scalar_type() == proj_weight.scalar_type(), "x/proj_weight must share dtype");
    TORCH_CHECK(x.scalar_type() == scale.scalar_type(), "x/scale must share dtype");
    TORCH_CHECK(x.scalar_type() == norm_scratch.scalar_type(), "x/norm_scratch must share dtype");
    TORCH_CHECK(logits_scratch.scalar_type() == at::ScalarType::Float,
                "logits_scratch must be float32");
    TORCH_CHECK(per_expert_scale.scalar_type() == at::ScalarType::Float,
                "per_expert_scale must be float32");
    TORCH_CHECK(sync_scratch.scalar_type() == at::ScalarType::Int,
                "sync_scratch must be int32");
    int64_t T = x.size(0);
    at::Tensor topk_weights = at::empty({T, top_k}, x.options().dtype(at::ScalarType::Float));
    at::Tensor topk_ids = at::empty({T, top_k}, x.options().dtype(at::ScalarType::Int));
    float eps = static_cast<float>(epsilon);
    if (T > 1) {
        EXEC_NPU_CMD(aclnnDgemmaFusedRouterFront,
                     x, scale, proj_weight, norm_scratch, logits_scratch,
                     per_expert_scale, sync_scratch, eps, hidden_size, num_experts, top_k,
                     sync_base,
                     topk_weights, topk_ids);
    }
    EXEC_NPU_CMD(aclnnDgemmaFusedRouterFront,
                 x, scale, proj_weight, norm_scratch, logits_scratch,
                 per_expert_scale, sync_scratch, eps, hidden_size, num_experts, top_k,
                 sync_base,
                 topk_weights, topk_ids);
    return std::make_tuple(topk_weights, topk_ids);
}
} // namespace vllm_ascend
#endif
