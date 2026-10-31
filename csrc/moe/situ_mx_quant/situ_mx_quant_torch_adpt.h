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

#ifndef SITU_MX_QUANT_TORCH_ADPT_H
#define SITU_MX_QUANT_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> situ_mx_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& group_list,
    double beta,
    double linear_beta,
    bool activate_left,
    int64_t dst_type)
{
    constexpr int64_t DST_TYPE_E5M2 = 35;
    constexpr int64_t DST_TYPE_E4M3FN = 36;
    constexpr int64_t MX_BLOCK_SPAN = 64;
    constexpr int64_t MX_SCALE_ALIGN = 2;

    TORCH_CHECK(x.dim() >= 1,
                "situ_mx_quant: x must be at least 1-dimensional, but got ",
                x.dim());
    TORCH_CHECK(x.size(-1) % 2 == 0,
                "situ_mx_quant: x last dim must be even, but got ", x.size(-1));
    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
                "situ_mx_quant: x must be bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(beta > 0.0,
                "situ_mx_quant: beta must be greater than 0, but got ", beta);
    TORCH_CHECK(dst_type == DST_TYPE_E4M3FN || dst_type == DST_TYPE_E5M2,
                "situ_mx_quant: dst_type must be 36 (E4M3FN) or 35 (E5M2), but got ",
                dst_type);
    if (group_list.has_value()) {
        TORCH_CHECK(group_list->device() == x.device(),
                    "situ_mx_quant: group_list must be on the same device as x");
        TORCH_CHECK(group_list->dim() == 1 && group_list->numel() > 0,
                    "situ_mx_quant: group_list must be a non-empty 1-D tensor");
        TORCH_CHECK(group_list->scalar_type() == at::kLong,
                    "situ_mx_quant: group_list must have int64 dtype, but got ",
                    group_list->scalar_type());
    }

    std::vector<int64_t> y_shape(x.sizes().begin(), x.sizes().end());
    y_shape.back() /= 2;
    std::vector<int64_t> mxscale_shape(y_shape.begin(), y_shape.end());
    mxscale_shape.back() = (y_shape.back() + MX_BLOCK_SPAN - 1) / MX_BLOCK_SPAN;
    mxscale_shape.push_back(MX_SCALE_ALIGN);

    auto y_dtype = dst_type == DST_TYPE_E5M2 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_shape, x.options().dtype(y_dtype));
    at::Tensor mxscale = at::empty(mxscale_shape, x.options().dtype(at::kFloat8_e8m0fnu));
    const at::Tensor group_list_value = group_list.value_or(at::Tensor());

    constexpr int64_t AXIS = -1;
    EXEC_NPU_CMD(aclnnSituMxQuant,
                 x,
                 group_list_value,
                 beta,
                 linear_beta,
                 activate_left,
                 AXIS,
                 dst_type,
                 y,
                 mxscale);
    return {y, mxscale};
}

}  // namespace vllm_ascend

#endif  // SITU_MX_QUANT_TORCH_ADPT_H
