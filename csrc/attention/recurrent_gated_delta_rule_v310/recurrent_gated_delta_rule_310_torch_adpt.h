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
                 output
                ); 
    return output;
}

}
#endif