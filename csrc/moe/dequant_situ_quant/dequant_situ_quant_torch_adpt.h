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

#ifndef DEQUANT_SITU_QUANT_TORCH_ADPT_H
#define DEQUANT_SITU_QUANT_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> dequant_situ_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    double beta,
    double linear_beta,
    bool activate_left,
    c10::string_view quant_mode)
{
    TORCH_CHECK(x.dim() == 2,
                "dequant_situ_quant: x must be 2-dimensional [rows, width], but got rank ",
                x.dim());
    const int64_t input_width = x.size(1);

    at::Tensor y = at::empty({x.size(0), input_width / 2}, x.options().dtype(at::kChar));
    at::Tensor scale = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    if (x.size(0) == 0) {
        return {y, scale};
    }

    const at::Tensor weight_scale_value = weight_scale.value_or(at::Tensor());
    const at::Tensor activation_scale_value = activation_scale.value_or(at::Tensor());
    const at::Tensor bias_value = bias.value_or(at::Tensor());
    const at::Tensor quant_scale_value = quant_scale.value_or(at::Tensor());
    const at::Tensor quant_offset_value = quant_offset.value_or(at::Tensor());
    const at::Tensor group_index_value = group_index.value_or(at::Tensor());
    std::string quant_mode_string(quant_mode);
    char* quant_mode_ptr = quant_mode_string.data();

    EXEC_NPU_CMD(aclnnDequantSituQuant,
                 x,
                 weight_scale_value,
                 activation_scale_value,
                 bias_value,
                 quant_scale_value,
                 quant_offset_value,
                 group_index_value,
                 beta,
                 linear_beta,
                 activate_left,
                 quant_mode_ptr,
                 y,
                 scale);
    return {y, scale};
}

} // namespace vllm_ascend

#endif // DEQUANT_SITU_QUANT_TORCH_ADPT_H
