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
#ifndef CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H
#define CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H

namespace vllm_ascend {

// Dispatches to the CANN aclnn operator aclnnChunkGatedDeltaRule.
// The operator is compiled/installed from the chunk_gated_delta_rule source
// tree under csrc/attention/chunk_gated_delta_rule (see build_aclnn.sh) and
// resolved at runtime by name via EXEC_NPU_CMD, so no aclnn header include is
// needed here (same pattern as recurrent_gated_delta_rule_torch_adpt.h).
//
// Inputs (TND layout):
//   query            (T, Nk, Dk)  bf16
//   key              (T, Nk, Dk)  bf16
//   value            (T, Nv, Dv)  bf16
//   beta             (T, Nv)      bf16
//   initial_state     (B, Nv, Dv, Dk)  bf16/fp32
//   actual_seq_lengths (B,)        int32
//   g                (T, Nv)      fp32 (optional; nullptr => all-zero)
//   scale_value      float
// Outputs:
//   out              (T, Nv, Dv)  bf16
//   final_state       (B, Nv, Dv, Dk)  bf16/fp32
inline std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    const at::Tensor& actual_seq_lengths,
    const c10::optional<at::Tensor>& g,
    double scale_value)
{
    TORCH_CHECK(scale_value != 0.0, "scale_value cannot be 0.");

    // out: same shape as value, bf16.
    auto out_options = value.options().dtype(at::ScalarType::BFloat16);
    at::Tensor out = at::empty(value.sizes(), out_options);

    // final_state: same shape and dtype as initial_state (bf16 or fp32).
    at::Tensor final_state = at::empty(initial_state.sizes(), initial_state.options());

    float scale_real = static_cast<float>(scale_value);
    EXEC_NPU_CMD(aclnnChunkGatedDeltaRule,
                 query,
                 key,
                 value,
                 beta,
                 initial_state,
                 actual_seq_lengths,
                 g,
                 scale_real,
                 out,
                 final_state);
    return std::make_tuple(out, final_state);
}

} // namespace vllm_ascend
#endif // CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H
