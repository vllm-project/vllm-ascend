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
    c10::optional<int64_t> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<int64_t> quant_dtype,
    bool transpose_weight,
    int64_t group_list_type,
    at::IntArrayRef tuning_config,
    double swiglu_limit)
{

    auto x_size = x.sizes();
    int64_t dequant_mode_real = dequant_mode.value_or(0);
    int64_t dequant_dtype_real = dequant_dtype.value_or(0);
    int64_t quant_mode_real = quant_mode.value_or(0);
    // quant_mode=2 (MX quant) outputs fp8 with e8m0 scales; other modes output int8 with fp32 scales
    bool is_mx_quant = (quant_mode_real == 2);
    // MX mode: weight_scale is [.., N, 2] (4D), n is the second-to-last dim; other modes: [.., N], n is last dim
    int n = is_mx_quant ? static_cast<int>(weight_scale[0].size(-2))
                        : static_cast<int>(weight_scale[0].size(-1));
    int m = x_size[0];
    int k = x_size[1];
    auto out_scalar_type = is_mx_quant ? at::kFloat8_e4m3fn : at::kChar;
    auto scale_scalar_type = is_mx_quant ? at::kFloat8_e8m0fnu : at::kFloat;
    at::Tensor output =  at::empty({m, n/2}, x.options().dtype(out_scalar_type));
    // MX mode: scale shape is [m, ceil(n/2/64), 2] (e8m0, one scale per 64-elem block of the swiglu output);
    // pertoken mode: scale shape is [m] (fp32)
    at::Tensor output_scale = is_mx_quant
        ? at::empty({m, (n/2 + 63) / 64, 2}, x.options().dtype(scale_scalar_type))
        : at::empty({m}, x.options().dtype(scale_scalar_type));
    auto bias_real = bias.value_or(at::Tensor());
    auto smooth_scale_real = smooth_scale.value_or(at::Tensor());
    double swiglu_limit_f = static_cast<double>(swiglu_limit);
    auto ws=weight[0].sizes();
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
    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}

}
#endif