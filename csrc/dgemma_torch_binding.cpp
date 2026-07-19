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

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/dgemma_apply_router_scale/dgemma_apply_router_scale_torch_adpt.h"
#include "attention/dgemma_fused_norm_rope/dgemma_fused_norm_rope_torch_adpt.h"
#include "attention/dgemma_fused_qkv_proj_norm_rope/dgemma_fused_qkv_proj_norm_rope_torch_adpt.h"
#include "attention/dgemma_fused_router_front/dgemma_fused_router_front_torch_adpt.h"
#include "moe/moe_gating_top_k/dgemma_moe_gating_top_k_scaled_torch_adpt.h"

#ifndef ASCEND_PLATFORM_310P
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "dgemma_moe_gating_top_k_scaled(Tensor x, "
        "Tensor per_expert_scale, "
        "int k, "
        "int k_group, "
        "int group_count, "
        "int group_select_mode, "
        "int renorm, "
        "int norm_type, "
        "bool out_flag, "
        "float routed_scaling_factor, "
        "float eps,"
        "Tensor? bias_opt=None)"
        "-> (Tensor y ,Tensor expert_idx, Tensor out)");
    ops.impl("dgemma_moe_gating_top_k_scaled", torch::kPrivateUse1,
             &vllm_ascend::dgemma_moe_gating_top_k_scaled);

    // Fused q/k/v RMSNorm + neox RoPE for DiffusionGemma pre-attention.
    ops.def(
        "npu_dgemma_fused_norm_rope(Tensor q, "
        "                           Tensor k, "
        "                           Tensor v, "
        "                           Tensor q_weight, "
        "                           Tensor k_weight, "
        "                           Tensor cos, "
        "                           Tensor sin, "
        "                           int num_q_heads, "
        "                           int num_kv_heads, "
        "                           int head_dim, "
        "                           float epsilon=1e-6) -> (Tensor q_out, Tensor k_out, Tensor v_out)");
    ops.impl("npu_dgemma_fused_norm_rope", torch::kPrivateUse1,
             &vllm_ascend::npu_dgemma_fused_norm_rope);

    // Fused qkv_proj GEMM + q/k/v RMSNorm + neox RoPE MIX op.
    ops.def(
        "npu_dgemma_fused_qkv_proj_norm_rope(Tensor hidden, "
        "                                    Tensor wqkv, "
        "                                    Tensor q_weight, "
        "                                    Tensor k_weight, "
        "                                    Tensor cos, "
        "                                    Tensor sin, "
        "                                    Tensor qkv_scratch, "
        "                                    int num_q_heads, "
        "                                    int num_kv_heads, "
        "                                    int head_dim, "
        "                                    int hidden_size, "
        "                                    int sync_base=0, "
        "                                    float epsilon=1e-6) -> "
        "(Tensor q_out, Tensor k_out, Tensor v_out, Tensor qkv_scratch_out)");
    ops.impl("npu_dgemma_fused_qkv_proj_norm_rope", torch::kPrivateUse1,
             &vllm_ascend::npu_dgemma_fused_qkv_proj_norm_rope);

    ops.def(
        "npu_dgemma_apply_router_scale(Tensor topk_weights, "
        "                               Tensor topk_ids, "
        "                               Tensor per_expert_scale) -> Tensor");
    ops.impl("npu_dgemma_apply_router_scale", torch::kPrivateUse1,
             &vllm_ascend::npu_dgemma_apply_router_scale);

    // Full router front: norm/root scale/proj/topk/per-expert scale.
    ops.def(
        "npu_dgemma_fused_router_front(Tensor hidden, "
        "                              Tensor scale, "
        "                              Tensor proj_weight, "
        "                              Tensor norm_scratch, "
        "                              Tensor logits_scratch, "
        "                              Tensor per_expert_scale, "
        "                              Tensor sync_scratch, "
        "                              int hidden_size, "
        "                              int num_experts, "
        "                              int top_k, "
        "                              int sync_base=1, "
        "                              float epsilon=1e-6) -> "
        "(Tensor topk_weights, Tensor topk_ids)");
    ops.impl("npu_dgemma_fused_router_front", torch::kPrivateUse1,
             &vllm_ascend::npu_dgemma_fused_router_front);
}
#endif
