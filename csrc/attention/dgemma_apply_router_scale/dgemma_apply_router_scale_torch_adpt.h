/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef DGEMMA_APPLY_ROUTER_SCALE_TORCH_ADPT_H
#define DGEMMA_APPLY_ROUTER_SCALE_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor npu_dgemma_apply_router_scale(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    const at::Tensor& per_expert_scale)
{
    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
                "topk_weights must be float32");
    TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
                "topk_ids must be int32");
    TORCH_CHECK(per_expert_scale.scalar_type() == at::ScalarType::Float,
                "per_expert_scale must be float32");
    TORCH_CHECK(per_expert_scale.dim() == 1, "per_expert_scale must be 1D");
    at::Tensor out = at::empty_like(topk_weights);
    EXEC_NPU_CMD(aclnnDgemmaApplyRouterScale,
                 topk_weights, topk_ids, per_expert_scale, out);
    return out;
}
} // namespace vllm_ascend
#endif
