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
#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_TORCH_ADPT_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_TORCH_ADPT_H
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out)
    {
        at::Tensor output = at::empty_like(residual);
        at::Tensor add_out = at::empty_like(residual);

        std::string group_tp_str(group_tp);

        char *group_tp_ptr = group_tp_str.data();

        float epsilon_f = static_cast<float>(epsilon);
        EXEC_NPU_CMD(aclnnMatmulAllreduceAddRmsnorm,
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