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
#ifndef GATHER_PA_KV_CACHE_VLLM_TORCH_ADPT_H
#define GATHER_PA_KV_CACHE_VLLM_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor> npu_gather_pa_kv_cache_vllm(
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_tables,
    const at::Tensor &seq_lens,
    const at::Tensor &key_ref,
    const at::Tensor &value_ref,
    const c10::optional<at::Tensor> &seq_offset,
    c10::string_view cache_mode,
    bool is_seq_lens_cumsum)
{
    std::string cache_mode_str(cache_mode);
    char *cache_mode_ptr = const_cast<char *>(cache_mode_str.c_str());

    EXEC_NPU_CMD(aclnnGatherPaKvCacheVllm,
                     key_cache, value_cache, block_tables, seq_lens,
                     key_ref, value_ref, seq_offset,
                     cache_mode_ptr, is_seq_lens_cumsum);

    return std::make_tuple(key_ref, value_ref);
}

}
#endif