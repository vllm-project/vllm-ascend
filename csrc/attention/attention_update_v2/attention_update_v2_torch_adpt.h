/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef ATTENTION_UPDATE_V2_TORCH_ADPT_H
#define ATTENTION_UPDATE_V2_TORCH_ADPT_H

namespace vllm_ascend {

// lse:       [sp, bsh]     float32
// local_out: [sp, bsh, hd] float32/float16/bfloat16
// out:       [bsh, hd]     same dtype as local_out
// lse_out:   [bsh]         float32, valid only when update_type == 1
std::tuple<at::Tensor, at::Tensor> npu_attention_update_v2(
    const at::Tensor& lse,
    const at::Tensor& local_out,
    int64_t update_type)
{
    at::Tensor out = at::empty({local_out.size(1), local_out.size(2)}, local_out.options());
    at::Tensor lse_out = at::empty({lse.size(1)}, lse.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnAttentionUpdateV2, lse, local_out, update_type, out, lse_out);
    return {out, lse_out};
}

} // namespace vllm_ascend
#endif
