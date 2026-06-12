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
#ifndef FUSED_GDN_GATING_V310_TORCH_ADPT_H
#define FUSED_GDN_GATING_V310_TORCH_ADPT_H

#include <tuple>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating_310(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    double beta,
    double threshold)
{
    std::vector<int64_t> out_shape = {1, a.size(0), a.size(1)};
    at::Tensor g = at::empty(out_shape, a.options().dtype(at::kFloat));
    at::Tensor beta_output = at::empty(out_shape, a.options());

    float beta_real = static_cast<float>(beta);
    float threshold_real = static_cast<float>(threshold);

    EXEC_NPU_CMD(aclnnFusedGdnGatingV310,
                 a, b, A_log, dt_bias,
                 beta_real, threshold_real,
                 g, beta_output);

    return std::tuple<at::Tensor, at::Tensor>(g, beta_output);
}

} // namespace vllm_ascend
#endif // FUSED_GDN_GATING_V310_TORCH_ADPT_H
