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

#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

namespace vllm_ascend {
namespace meta {

std::tuple<at::Tensor, at::Tensor, at::Tensor> dgemma_moe_gating_top_k_scaled_meta(
    const at::Tensor& x,
    const at::Tensor& per_expert_scale,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt)
{
    (void)per_expert_scale;
    (void)k_group;
    (void)group_count;
    (void)group_select_mode;
    (void)renorm;
    (void)norm_type;
    (void)out_flag;
    (void)routed_scaling_factor;
    (void)eps;
    (void)bias_opt;
    auto rows = x.sym_size(0);
    auto expert_num = x.sym_size(1);
    at::Tensor y = at::empty_symint(
        c10::SymDimVector{rows, c10::SymInt(k)}, x.options());
    at::Tensor expert_idx = at::empty_symint(
        c10::SymDimVector{rows, c10::SymInt(k)}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty_symint(
        c10::SymDimVector{rows, expert_num}, x.options().dtype(at::kFloat));
    return std::make_tuple(y, expert_idx, out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dgemma_fused_norm_rope_meta(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    double epsilon)
{
    (void)q_weight;
    (void)k_weight;
    (void)cos;
    (void)sin;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)epsilon;
    return std::make_tuple(at::empty_like(q), at::empty_like(k), at::empty_like(v));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_dgemma_fused_qkv_proj_norm_rope_meta(
    const at::Tensor& hidden,
    const at::Tensor& wqkv,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& qkv_scratch,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t hidden_size,
    int64_t sync_base,
    double epsilon)
{
    (void)wqkv;
    (void)q_weight;
    (void)k_weight;
    (void)cos;
    (void)sin;
    (void)hidden_size;
    (void)sync_base;
    (void)epsilon;
    auto T = hidden.sym_size(0);
    at::Tensor q_out = at::empty_symint(
        c10::SymDimVector{T, c10::SymInt(num_q_heads), c10::SymInt(head_dim)},
        hidden.options());
    at::Tensor k_out = at::empty_symint(
        c10::SymDimVector{T, c10::SymInt(num_kv_heads), c10::SymInt(head_dim)},
        hidden.options());
    at::Tensor v_out = at::empty_symint(
        c10::SymDimVector{T, c10::SymInt(num_kv_heads), c10::SymInt(head_dim)},
        hidden.options());
    at::Tensor qkv_scratch_out = at::empty_symint(
        qkv_scratch.sym_sizes(), qkv_scratch.options());
    return std::make_tuple(q_out, k_out, v_out, qkv_scratch_out);
}

at::Tensor npu_dgemma_apply_router_scale_meta(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    const at::Tensor& per_expert_scale)
{
    (void)topk_ids;
    (void)per_expert_scale;
    return at::empty_like(topk_weights);
}

std::tuple<at::Tensor, at::Tensor> npu_dgemma_fused_router_front_meta(
    const at::Tensor& hidden,
    const at::Tensor& scale,
    const at::Tensor& proj_weight,
    const at::Tensor& norm_scratch,
    const at::Tensor& logits_scratch,
    const at::Tensor& per_expert_scale,
    const at::Tensor& sync_scratch,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    int64_t sync_base,
    double epsilon)
{
    (void)scale;
    (void)proj_weight;
    (void)norm_scratch;
    (void)logits_scratch;
    (void)per_expert_scale;
    (void)sync_scratch;
    (void)hidden_size;
    (void)num_experts;
    (void)sync_base;
    (void)epsilon;
    auto T = hidden.sym_size(0);
    at::Tensor topk_weights = at::empty_symint(
        c10::SymDimVector{T, c10::SymInt(top_k)},
        hidden.options().dtype(at::kFloat));
    at::Tensor topk_ids = at::empty_symint(
        c10::SymDimVector{T, c10::SymInt(top_k)},
        hidden.options().dtype(at::kInt));
    return std::make_tuple(topk_weights, topk_ids);
}

}  // namespace meta
}  // namespace vllm_ascend

#ifndef ASCEND_PLATFORM_310P
namespace {
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)
{
    ops.impl("dgemma_moe_gating_top_k_scaled",
             &vllm_ascend::meta::dgemma_moe_gating_top_k_scaled_meta);
    ops.impl("npu_dgemma_fused_norm_rope",
             &vllm_ascend::meta::npu_dgemma_fused_norm_rope_meta);
    ops.impl("npu_dgemma_fused_qkv_proj_norm_rope",
             &vllm_ascend::meta::npu_dgemma_fused_qkv_proj_norm_rope_meta);
    ops.impl("npu_dgemma_apply_router_scale",
             &vllm_ascend::meta::npu_dgemma_apply_router_scale_meta);
    ops.impl("npu_dgemma_fused_router_front",
             &vllm_ascend::meta::npu_dgemma_fused_router_front_meta);
}
}  // namespace
#endif
