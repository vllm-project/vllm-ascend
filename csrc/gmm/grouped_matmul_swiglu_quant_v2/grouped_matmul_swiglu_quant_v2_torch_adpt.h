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
#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_TORCH_ADPT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_TORCH_ADPT_H
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_v2(
    const at::Tensor & x,
    const at::TensorList &weight,
    const at::TensorList &weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & smooth_scale,
    const c10::optional<at::TensorList> weight_assist_matrix,
    const c10::optional<at::Tensor> & bias,
    c10::optional<int64_t> dequant_mode,
    c10::optional<at::ScalarType> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<at::ScalarType> quant_dtype,
    bool transpose_weight,
    int64_t group_list_type,
    at::IntArrayRef tuning_config,
    double swiglu_limit)
{

    auto x_size = x.sizes();
    int64_t m = x_size[0];
    int64_t quant_mode_real = quant_mode.value_or(0);
    int64_t n = quant_mode_real == 2
        ? weight[0].size(transpose_weight ? 1 : 2)
        : weight_scale[0].sizes().back();

    at::ScalarType output_dtype = quant_dtype.value_or(at::kChar);
    at::Tensor output = at::empty({m, n / 2}, x.options().dtype(output_dtype));
    at::Tensor output_scale;
    if (quant_mode_real == 2) {
        output_scale = at::empty({m, (n / 2 + 63) / 64, 2}, x.options().dtype(at::kFloat8_e8m0fnu));
    } else {
        output_scale = at::empty({m}, x.options().dtype(at::kFloat));
    }
    int64_t dequant_mode_real = dequant_mode.value_or(0);
    int64_t dequant_dtype_real = static_cast<int64_t>(ConvertType(dequant_dtype.value_or(at::kFloat)));
    auto bias_real = bias.value_or(at::Tensor());
    auto smooth_scale_real = smooth_scale.value_or(at::Tensor());
    double swiglu_limit_f = static_cast<double>(swiglu_limit);
    if (quant_dtype.has_value()) {
        // A5 consumes the public ND V2 contract and supports per-token and MX
        // quantization. A2/A3 omit quant_dtype and retain the Weight-NZ bridge
        // required by their packed W8A8/W4A8 layouts.
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantV2,
            x,
            weight,
            weight_scale,
            weight_assist_matrix,
            bias_real,
            x_scale,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type,
            tuning_config,
            swiglu_limit_f,
            output,
            output_scale);
    } else {
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantWeightNzV2,
            x,
            weight,
            weight_scale,
            weight_assist_matrix,
            bias_real,
            x_scale,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type,
            tuning_config,
            swiglu_limit_f,
            output,
            output_scale);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}

}
#endif
