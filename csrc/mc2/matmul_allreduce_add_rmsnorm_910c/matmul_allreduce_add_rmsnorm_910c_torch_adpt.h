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
#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_910C_TORCH_ADPT_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_910C_TORCH_ADPT_H
#include <ATen/core/dispatch/Dispatcher.h>
#include "../../moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
namespace vllm_ascend {

inline at::Tensor call_vllm_tensor_op(
    const char *op_name,
    c10::Stack stack)
{
    const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(op_name, "");
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1 && stack.back().isTensor(), op_name, " must return one Tensor");
    return std::move(stack.back()).toTensor();
}

inline std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm_fallback(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view fallback_group_name,
    double epsilon)
{
    at::Tensor matmul_out = at::matmul(x1, x2.transpose(-2, -1));
    at::Tensor allreduce_out = call_vllm_tensor_op(
        "vllm::all_reduce", {matmul_out, std::string(fallback_group_name)});
    at::Tensor chunked_residual = call_vllm_tensor_op(
        "vllm::maybe_chunk_residual", {allreduce_out, residual});
    auto norm_out = npu_add_rms_norm_bias(allreduce_out, chunked_residual, gamma, c10::nullopt, epsilon);
    return {std::get<0>(norm_out), std::get<2>(norm_out)};
}

std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm_910c(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out,
    c10::string_view fallback_group_name)
    {
        const int64_t k = x1.dim() > 0 ? x1.size(-1) : 0;
        const int64_t m = k > 0 ? x1.numel() / k : 0;
        const int64_t n = x2.dim() > 0 ? x2.size(0) : 0;
        const bool supported = tp_rank_size == 2 && is_trans_b && x1.scalar_type() == at::kBFloat16 &&
            m >= 1 && m <= 2 && n == 5120 && k == 12800;
        if (!supported) {
            TORCH_CHECK(!fallback_group_name.empty(),
                "fallbackGroupName is required for unsupported matmul_allreduce_add_rmsnorm_910c shapes");
            return matmul_allreduce_add_rmsnorm_fallback(
                x1, x2, residual, gamma, fallback_group_name, epsilon);
        }

        at::Tensor output = at::empty_like(residual);
        at::Tensor add_out = at::empty_like(residual);

        std::string group_tp_str(group_tp);

        char *group_tp_ptr = group_tp_str.data();

        float epsilon_f = static_cast<float>(epsilon);
        EXEC_NPU_CMD(aclnnMatmulAllreduceAddRmsnorm910c,
            // input
            x1, x2, residual, gamma,
            // attr
            group_tp_ptr, tp_rank_size, tp_rank_id, epsilon_f, is_trans_b, is_gather_add_out,
            // output
            output, add_out);

        return {output, add_out};
    }
}
#endif
