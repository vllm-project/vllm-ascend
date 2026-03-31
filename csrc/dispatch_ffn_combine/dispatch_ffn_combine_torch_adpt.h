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
#ifndef DISPATCH_FFN_COMBINE_TORCH_ADPT_H
#define DISPATCH_FFN_COMBINE_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out,
    at::Tensor& expert_token_nums
) {
    char *group_ep_ptr = const_cast<char *>(group.data());
    bool is_int8 = weight1[0].dtype() == at::kChar;
    if (is_int8) {
        EXEC_NPU_CMD(aclnnDispatchFFNCombine,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 group_ep_ptr,
                 max_output_size,
                 out,
                 expert_token_nums);
    } else {
        EXEC_NPU_CMD(aclnnDispatchFFNCombineBF16,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 group_ep_ptr,
                 max_output_size,
                 out,
                 expert_token_nums);
    }    
    return {out, expert_token_nums};
}
}
#endif