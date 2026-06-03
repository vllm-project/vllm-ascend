/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_TORCH_ADPT_H
#define FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_TORCH_ADPT_H

#include "op_host/op_api/aclnn_fused_sigmoid_gating_delta_rule_update.h"

namespace vllm_ascend {

at::Tensor npu_fused_sigmoid_gating_delta_rule_update(
    const at::Tensor& a_log,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const at::Tensor& actual_seq_lengths,
    const at::Tensor& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    double scale_value = 1.0,
    double softplus_beta = 1.0,
    double softplus_threshold = 20.0)
{
    constexpr int64_t QKV_DIM_NUM = 3;
    constexpr int64_t GATE_DIM_NUM = 2;
    constexpr int64_t HEAD_PARAM_DIM_NUM = 1;
    constexpr int64_t STATE_DIM_NUM = 4;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t MAX_HEAD_NUM = 256;
    constexpr int64_t MAX_HEAD_DIM = 512;

    TORCH_CHECK(a_log.dim() == HEAD_PARAM_DIM_NUM,
                "A_log should be 1-D [num_value_heads], got ", a_log.dim(), "D");
    TORCH_CHECK(dt_bias.dim() == HEAD_PARAM_DIM_NUM,
                "dt_bias should be 1-D [num_value_heads], got ", dt_bias.dim(), "D");
    TORCH_CHECK(a.dim() == GATE_DIM_NUM,
                "a should be 2-D [tokens, num_value_heads], got ", a.dim(), "D");
    TORCH_CHECK(b.dim() == GATE_DIM_NUM,
                "b should be 2-D [tokens, num_value_heads], got ", b.dim(), "D");
    TORCH_CHECK(query.dim() == QKV_DIM_NUM,
                "query should be 3-D [tokens, num_key_heads, key_dim], got ", query.dim(), "D");
    TORCH_CHECK(key.dim() == QKV_DIM_NUM,
                "key should be 3-D [tokens, num_key_heads, key_dim], got ", key.dim(), "D");
    TORCH_CHECK(value.dim() == QKV_DIM_NUM,
                "value should be 3-D [tokens, num_value_heads, value_dim], got ", value.dim(), "D");
    TORCH_CHECK(state.dim() == STATE_DIM_NUM,
                "state should be 4-D [state_blocks, num_value_heads, value_dim, key_dim], got ", state.dim(), "D");
    TORCH_CHECK(actual_seq_lengths.dim() == HEAD_PARAM_DIM_NUM,
                "actual_seq_lengths should be 1-D, got ", actual_seq_lengths.dim(), "D");
    TORCH_CHECK(ssm_state_indices.dim() == HEAD_PARAM_DIM_NUM,
                "ssm_state_indices should be 1-D, got ", ssm_state_indices.dim(), "D");
    if (num_accepted_tokens.has_value()) {
        TORCH_CHECK(num_accepted_tokens.value().dim() == HEAD_PARAM_DIM_NUM,
                    "num_accepted_tokens should be 1-D when provided, got ",
                    num_accepted_tokens.value().dim(), "D");
    }

    TORCH_CHECK(query.size(DIM_0) == key.size(DIM_0) &&
                query.size(DIM_1) == key.size(DIM_1) &&
                query.size(DIM_2) == key.size(DIM_2),
                "query and key must have the same shape, got query=", query.sizes(), " key=", key.sizes());
    TORCH_CHECK(value.size(DIM_0) == query.size(DIM_0),
                "value tokens must equal query tokens, got value.size(0)=",
                value.size(DIM_0), " query.size(0)=", query.size(DIM_0));
    TORCH_CHECK(a.size(DIM_0) == query.size(DIM_0) && b.size(DIM_0) == query.size(DIM_0),
                "a and b tokens must equal query tokens, got a.size(0)=",
                a.size(DIM_0), " b.size(0)=", b.size(DIM_0),
                " query.size(0)=", query.size(DIM_0));
    TORCH_CHECK(a.size(DIM_1) == value.size(DIM_1) && b.size(DIM_1) == value.size(DIM_1),
                "a and b num_value_heads must equal value.size(1), got a.size(1)=",
                a.size(DIM_1), " b.size(1)=", b.size(DIM_1),
                " value.size(1)=", value.size(DIM_1));
    TORCH_CHECK(a_log.size(DIM_0) == value.size(DIM_1) &&
                dt_bias.size(DIM_0) == value.size(DIM_1),
                "A_log and dt_bias length must equal num_value_heads, got A_log.size(0)=",
                a_log.size(DIM_0), " dt_bias.size(0)=", dt_bias.size(DIM_0),
                " value.size(1)=", value.size(DIM_1));
    TORCH_CHECK(state.size(DIM_1) == value.size(DIM_1) &&
                state.size(DIM_2) == value.size(DIM_2) &&
                state.size(DIM_3) == query.size(DIM_2),
                "state shape must match [*, value.size(1), value.size(2), query.size(2)], got state=",
                state.sizes(), " value=", value.sizes(), " query=", query.sizes());
    TORCH_CHECK(query.size(DIM_1) > 0 && value.size(DIM_1) % query.size(DIM_1) == 0,
                "num_value_heads must be an integer multiple of num_key_heads, got value.size(1)=",
                value.size(DIM_1), " query.size(1)=", query.size(DIM_1));
    TORCH_CHECK(query.size(DIM_1) <= MAX_HEAD_NUM && value.size(DIM_1) <= MAX_HEAD_NUM &&
                query.size(DIM_2) <= MAX_HEAD_DIM && value.size(DIM_2) <= MAX_HEAD_DIM,
                "num_key_heads and num_value_heads should be <= ", MAX_HEAD_NUM,
                ", key_dim and value_dim should be <= ", MAX_HEAD_DIM,
                ", got num_key_heads=", query.size(DIM_1),
                " num_value_heads=", value.size(DIM_1),
                " key_dim=", query.size(DIM_2),
                " value_dim=", value.size(DIM_2));

    at::Tensor output = at::empty(value.sizes(), value.options());
    float scale_value_f = static_cast<float>(scale_value);
    float softplus_beta_f = static_cast<float>(softplus_beta);
    float softplus_threshold_f = static_cast<float>(softplus_threshold);
    EXEC_NPU_CMD(aclnnFusedSigmoidGatingDeltaRuleUpdate,
                 a_log,
                 a,
                 b,
                 dt_bias,
                 query,
                 key,
                 value,
                 state,
                 actual_seq_lengths,
                 ssm_state_indices,
                 num_accepted_tokens,
                 scale_value_f,
                 softplus_beta_f,
                 softplus_threshold_f,
                 output);
    return output;
}

} // namespace vllm_ascend

#endif // FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_TORCH_ADPT_H
