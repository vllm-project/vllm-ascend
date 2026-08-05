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
#ifndef TURBOQUANT_RESHAPE_AND_CACHE_V310_TORCH_ADPT_H
#define TURBOQUANT_RESHAPE_AND_CACHE_V310_TORCH_ADPT_H
namespace vllm_ascend {

/*
 * TurboQuant KV-cache write: rotate -> norm -> quantize -> pack -> scatter.
 *
 * key_cache / value_cache are mutated in place (packed codes, fp16-typed
 * FRACTAL_NZ). The norm planes are returned rather than written in place so the
 * caller owns their lifetime; they are indexed [slot, kv_head] and must be fed
 * back to the paged-attention op unchanged.
 *
 * `signs` is the D vector of Pi = D*H*D and MUST be the same tensor the read op
 * receives, or the two rotated bases will not match.
 */
std::tuple<at::Tensor, at::Tensor> npu_turboquant_reshape_and_cache_v310(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& signs,
    const at::Tensor& centroids,
    int64_t bits,
    int64_t variant,
    int64_t codebook_mode)
{
    // norm planes: [num_slots, num_kv_heads]; num_slots = num_blocks * block_size
    const int64_t num_slots = key_cache.size(0) * key_cache.size(2);
    const int64_t num_kv_heads = key.size(1);
    auto opts = key.options();
    at::Tensor key_norms = at::zeros({num_slots, num_kv_heads}, opts);
    at::Tensor value_norms = at::zeros({num_slots, num_kv_heads}, opts);

    EXEC_NPU_CMD(aclnnTurboquantReshapeAndCacheV310,
                 key,
                 value,
                 key_cache,
                 value_cache,
                 slot_mapping,
                 signs,
                 centroids,
                 bits,
                 variant,
                 codebook_mode,
                 key_norms,
                 value_norms);
    return std::make_tuple(key_norms, value_norms);
}

}  // namespace vllm_ascend
#endif  // TURBOQUANT_RESHAPE_AND_CACHE_V310_TORCH_ADPT_H
