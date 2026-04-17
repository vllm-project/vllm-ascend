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
#ifndef SCATTER_PA_KV_CACHE_VLLM_TORCH_ADPT_H
#define SCATTER_PA_KV_CACHE_VLLM_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor> npu_scatter_pa_kv_cache_vllm(
    const at::Tensor &key, 
    const at::Tensor &value,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &slot_mapping,
    const c10::optional<at::Tensor>& compress_lens,
    const c10::optional<at::Tensor>& compress_seq_offsets,
    const c10::optional<at::Tensor>& seq_lens,
    c10::optional<c10::string_view> cache_mode)
{
    char* cache_mode_ptr = cache_mode.has_value() ? const_cast<char *>(cache_mode.value().data()) : nullptr;
    char* scatter_mode = "None";
    c10::SmallVector<int64_t, 2> strides_size = {1, 1};
    at::IntArrayRef strides = at::IntArrayRef(strides_size);
    c10::SmallVector<int64_t, 2> offsets_size = {0, 0};
    at::IntArrayRef offsets = at::IntArrayRef(offsets_size);

    EXEC_NPU_CMD(aclnnScatterPaKvCacheVllm, key, key_cache, slot_mapping, value, value_cache,
        compress_lens, compress_seq_offsets, seq_lens, cache_mode_ptr, scatter_mode, strides, offsets);
}

}
#endif