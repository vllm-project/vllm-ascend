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

#ifndef LIGHTNING_ATTENTION_DECODE_TORCH_ADPT_H
#define LIGHTNING_ATTENTION_DECODE_TORCH_ADPT_H

namespace vllm_ascend {
    at::Tensor npu_lightning_attention_decode(
        const at::Tensor &query,
        const at::Tensor &key,
        const at::Tensor &value,
        const at::Tensor &kv_caches_ref,
        const at::Tensor &slope_rate,
        const at::Tensor &slot_ids)
    {
        auto output_size_0 = {query.size(0), query.size(1) * query.size(3)};
        auto output_dtype_0 = query.scalar_type();
        at::Tensor attention_out = at::empty(output_size_0, query.options().dtype(output_dtype_0));
        EXEC_NPU_CMD(
            aclnnLightningAttentionDecode,
            query,
            key,
            value,
            slope_rate,
            kv_caches_ref,
            slot_ids,
            "BNSD",
            attention_out);
        return attention_out;
    }
}

#endif
