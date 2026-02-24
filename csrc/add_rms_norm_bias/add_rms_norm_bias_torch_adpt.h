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
#ifndef ADD_RMS_NORM_BIAS_TORCH_ADPT_H
#define ADD_RMS_NORM_BIAS_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor,at::Tensor, at::Tensor> npu_add_rms_norm_bias(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon)
{
    int64_t dim_x = x1.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    std::vector<int64_t> new_shape;
    at::Tensor rstd;
    
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x1_sizes = x1.sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x1_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(1);
        }
    } else {
        new_shape.assign(dim_x, 1);
    }
    rstd = at::empty(new_shape, x1.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x1.sizes(), x1.options());
    at::Tensor x = at::empty(x1.sizes(), x1.options());
    EXEC_NPU_CMD(aclnnAddRmsNormBias, x1, x2, gamma, beta, epsilon, y, rstd, x);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, rstd, x);
}
}
#endif