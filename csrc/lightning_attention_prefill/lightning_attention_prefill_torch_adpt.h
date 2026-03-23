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

#ifndef LIGHTNING_ATTENTION_PREFILL_TORCH_ADPT_H
#define LIGHTNING_ATTENTION_PREFILL_TORCH_ADPT_H

namespace vllm_ascend {
    std::tuple<at::Tensor, at::Tensor> npu_lightning_attention_prefill(
        const at::Tensor &query,
        const at::Tensor &key,
        const at::Tensor &value,
        const at::Tensor &slope_rate,
        int64_t block_size,
        const c10::optional<at::Tensor> &kv_history,
        at::OptionalIntArrayRef actual_seq_len)
    {
        auto default_seq_len = std::vector<int64_t>(query.size(0), query.size(2));
        auto actual_seq_len_value = actual_seq_len.value_or(default_seq_len);
        auto output_size_0 = {query.size(0), query.size(1), query.size(2), query.size(3)};
        auto output_size_1 = {query.size(0), query.size(1), query.size(3), query.size(3)};
        auto output_dtype_0 = query.scalar_type();
        at::Tensor attention_out = at::empty(output_size_0, query.options().dtype(output_dtype_0));
        at::Tensor kv_caches_out = at::empty(output_size_1, query.options().dtype(output_dtype_0));
        EXEC_NPU_CMD(
            aclnnLightningAttentionPrefill,
            query,
            key,
            value,
            slope_rate,
            kv_history,
            block_size,
            actual_seq_len_value,
            "BNSD",
            attention_out,
            kv_caches_out);
        return std::tuple<at::Tensor, at::Tensor>(attention_out, kv_caches_out);
    }
}

#endif
