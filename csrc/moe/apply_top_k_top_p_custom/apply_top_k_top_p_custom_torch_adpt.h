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
#ifndef APPLY_TOP_K_TOP_P_CUSTOM_TORCH_ADPT_H
#define APPLY_TOP_K_TOP_P_CUSTOM_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor npu_apply_top_k_top_p(
    const at::Tensor& logits,
    const c10::optional<at::Tensor>& p,
    const c10::optional<at::Tensor>& k)
{
    TORCH_CHECK(p.has_value() || k.has_value(),
                "apply_top_k_top_p: p and k cannot be None at the same time.");

    at::Tensor out = at::empty_like(logits);

    EXEC_NPU_CMD(
        aclnnApplyTopKTopPCustom,
        logits,
        p,
        k,
        out);

    return out;
}    
}
#endif