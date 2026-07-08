/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H
#define CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H

#include <tuple>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    const at::Tensor& actual_seq_lengths,
    const c10::optional<at::Tensor>& g,
    c10::optional<double> scale)
{
    TORCH_CHECK(scale.has_value(), "scale cannot be empty.");

    auto out_options = value.options().dtype(at::ScalarType::BFloat16);
    at::Tensor output = at::empty(value.sizes(), out_options);

    auto state_options = initial_state.options().dtype(at::ScalarType::Float);
    at::Tensor final_state = at::empty(initial_state.sizes(), state_options);

    float scale_real = static_cast<float>(scale.value());

    EXEC_NPU_CMD(aclnnChunkGatedDeltaRule,
                 query,
                 key,
                 value,
                 beta,
                 initial_state,
                 actual_seq_lengths,
                 g,
                 scale_real,
                 output,
                 final_state);

    return std::make_tuple(output, final_state);
}

} // namespace vllm_ascend

#endif // CHUNK_GATED_DELTA_RULE_TORCH_ADPT_H
