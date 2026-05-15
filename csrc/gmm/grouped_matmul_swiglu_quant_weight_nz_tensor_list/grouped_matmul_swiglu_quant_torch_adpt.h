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
#ifndef GROUPED_MATMUL_SWIGLU_QUANT_TORCH_ADPT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_TORCH_ADPT_H
namespace vllm_ascend {
const int64_t INT4_NUMS_IN_INT32 = 8;
std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weight_scale, const at::Tensor &x_scale,
    const at::Tensor &group_list, const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &offset)
{
    int m = x.sizes()[0];
    int n = weight.sizes()[2];
    bool is_a8w4 = x.dtype() == at::kChar && weight.dtype() == at::kInt;
    if (is_a8w4) {
        n *= INT4_NUMS_IN_INT32;
    }

    at::Tensor output = at::empty({m, n/2}, x.options().dtype(c10::ScalarType::Char));
    at::Tensor output_scale = at::empty({m}, x.options().dtype(c10::ScalarType::Float));
    at::Tensor output_offset = at::empty({}, x.options().dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNZ,
        x,
        weight,
        bias,
        offset,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale,
        output_offset);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_weight_nz_tensor_list(
    const at::Tensor & x,
    const at::TensorList & weight,
    const at::TensorList & weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & bias,
    const c10::optional<at::Tensor> & offset)
{
    auto x_size = x.sizes();
    int n = weight[0].sizes()[1];
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = at::empty({m, n/2}, x.options().dtype(at::kChar));
    at::Tensor output_scale = at::empty({m}, x.options().dtype(at::kFloat));
    at::Tensor output_offset = at::empty({m}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNzTensorList,
        x,
        weight,
        bias,
        offset,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale,
        output_offset);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}
}
#endif