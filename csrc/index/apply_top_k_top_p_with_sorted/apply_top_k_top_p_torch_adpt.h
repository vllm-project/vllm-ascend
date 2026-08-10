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
#ifndef APPLY_TOP_K_TOP_P_TORCH_ADPT_H
#define APPLY_TOP_K_TOP_P_TORCH_ADPT_H
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

namespace vllm_ascend {

at::Tensor npu_apply_top_k_top_p(
    const at::Tensor& logits,
    const c10::optional<at::Tensor>& k_opt,
    const c10::optional<at::Tensor>& p_opt)
{
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf ||
            logits.scalar_type() == at::kBFloat16,
        "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        logits.scalar_type());

    at::Tensor out = at::empty(logits.sizes(), logits.options());

    const at::Tensor& k = c10::value_or_else(k_opt, [] { return at::Tensor(); });
    const at::Tensor& p = c10::value_or_else(p_opt, [] { return at::Tensor(); });

    EXEC_NPU_CMD(aclnnApplyTopKTopP, logits, p, k, out);

    return out;
}

} // namespace vllm_ascend
#endif // APPLY_TOP_K_TOP_P_TORCH_ADPT_H