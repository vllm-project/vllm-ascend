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
#ifndef CAUSAL_CONV1D_V310_TORCH_ADPT_H
#define CAUSAL_CONV1D_V310_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_causal_conv1d_310(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& conv_states,
    at::IntArrayRef query_start_loc,
    at::IntArrayRef cache_indices,
    at::IntArrayRef initial_state_mode,
    at::IntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{
    at::Tensor output = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnCausalConv1dV310,
                 x,
                 weight,
                 bias,
                 conv_states,
                 query_start_loc,
                 cache_indices,
                 initial_state_mode,
                 num_accepted_tokens,
                 activation_mode,
                 pad_slot_id,
                 run_mode,
                 output
                ); 

    return output;
}

}
#endif