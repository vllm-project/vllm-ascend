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
#ifndef RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H
#define RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk)
{
    TORCH_CHECK(scale.has_value(), "scale cannot be empty.");
    constexpr int64_t stateDimNum = 4;
    TORCH_CHECK(state.dim() == stateDimNum, "state must be a 4D tensor, but got ", state.dim(), " dimensions.");
    for (int64_t dim = 0; dim < stateDimNum; ++dim) {
        TORCH_CHECK(state.size(dim) > 0, "state dimension ", dim, " must be positive, but got ", state.size(dim), ".");
    }

    const int64_t stateStride0 = state.stride(0);
    const int64_t stateStride1 = state.stride(1);
    const int64_t stateStride2 = state.stride(2);
    const int64_t stateStride3 = state.stride(3);
    TORCH_CHECK(stateStride0 > 0 && stateStride1 > 0 && stateStride2 > 0 && stateStride3 > 0,
                "state strides must be positive, but got [", stateStride0, ", ", stateStride1, ", ", stateStride2,
                ", ", stateStride3, "].");
    TORCH_CHECK(stateStride3 == 1, "state must be contiguous in its last dimension, but got stride(3)=",
                stateStride3, ".");
    TORCH_CHECK(stateStride2 == state.size(3),
                "state must be contiguous across its last two dimensions; expected stride(2)=", state.size(3),
                ", but got ", stateStride2, ".");
    TORCH_CHECK(stateStride2 <= stateStride1 / state.size(2),
                "state heads must not overlap; stride(1)=", stateStride1, ", stride(2)=", stateStride2,
                ", size(2)=", state.size(2), ".");
    const int64_t stateHeadSpan = state.size(2) * stateStride2;
    TORCH_CHECK(stateStride0 >= stateHeadSpan &&
                    (state.size(1) == 1 || stateStride1 <= (stateStride0 - stateHeadSpan) / (state.size(1) - 1)),
                "state blocks must not overlap; stride(0)=", stateStride0, ", stride(1)=", stateStride1,
                ", size(1)=", state.size(1), ".");

    auto options = value.options().dtype(at::ScalarType::BFloat16);
    at::Tensor output = at::empty(value.sizes(), options);
    float scale_real = static_cast<float>(scale.value());
    EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRule,
                 query,
                 key,
                 value,
                 beta,
                 state,
                 actual_seq_lengths,
                 ssm_state_indices,
                 g,
                 gk,
                 num_accepted_tokens,
                 scale_real,
                 stateStride0,
                 stateStride1,
                 stateStride2,
                 output);
    return output;
}

} // namespace vllm_ascend
#endif
