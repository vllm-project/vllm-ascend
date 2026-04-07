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
#include <algorithm>

#ifndef RECURRENT_GATED_DELTA_RULE_V310_TORCH_ADPT_H
#define RECURRENT_GATED_DELTA_RULE_V310_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_recurrent_gated_delta_rule_310(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& beta,
    at::Tensor& state,
    const at::Tensor& actual_seq_lengths,
    const at::Tensor& ssm_state_indices,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    double scale_value)
{
    at::Tensor output = at::empty(value.sizes(), value.options());
    float scale_real = static_cast<float>(scale_value);

    if (state.scalar_type() == at::kHalf) {
        EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRuleV310,
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
                     output);
        return output;
    }

    TORCH_CHECK(
        state.scalar_type() == at::kFloat,
        "npu_recurrent_gated_delta_rule_310 only supports float16 or float32 state, got ",
        state.scalar_type());

    // Keep the external cache in float32 while reusing the existing fp16
    // kernel implementation for the recurrent update. Only rows referenced by
    // ssm_state_indices are written back so untouched cache entries preserve
    // their original fp32 values.
    at::Tensor state_fp16 = state.to(at::kHalf);
    EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRuleV310,
                 query,
                 key,
                 value,
                 beta,
                 state_fp16,
                 actual_seq_lengths,
                 ssm_state_indices,
                 g,
                 gk,
                 num_accepted_tokens,
                 scale_real,
                 output);
    // Full-decode graph mode pads the state-index buffer with PAD_SLOT_ID (-1),
    // but the kernel only consumes the logical token prefix described by
    // actual_seq_lengths / num_accepted_tokens. Keep the fp32 cache write-back
    // aligned with that prefix so padded tail entries do not corrupt the last
    // state row during replay.
    int64_t active_token_count = ssm_state_indices.size(0);
    if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
        active_token_count = query.size(0);
    } else if (actual_seq_lengths.dim() == 1) {
        active_token_count = actual_seq_lengths.size(0);
    }
    active_token_count = std::min<int64_t>(active_token_count, ssm_state_indices.size(0));
    if (active_token_count == 0) {
        return output;
    }

    at::Tensor touched_state_indices = ssm_state_indices.narrow(0, 0, active_token_count).to(at::kLong).contiguous();
    at::Tensor touched_state_values = state_fp16.index_select(0, touched_state_indices).to(state.scalar_type());
    state.index_copy_(0, touched_state_indices, touched_state_values);
    return output;
}

}
#endif
