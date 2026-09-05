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
#ifndef TURBOQUANT_PAGED_ATTENTION_V310_TORCH_ADPT_H
#define TURBOQUANT_PAGED_ATTENTION_V310_TORCH_ADPT_H
namespace vllm_ascend {

/*
 * TurboQuant decode attention straight out of the packed cache.
 *
 * Runs entirely in the rotated basis: the query is rotated once on entry and
 * the accumulator once on exit, so cached K/V are never inverse-rotated.
 * `signs` must be the same D vector the write op used.
 *
 * query: [batch, num_heads, head_dim] (one token per sequence, decode)
 * out:   same shape as query
 */
at::Tensor npu_turboquant_paged_attention_v310(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& key_norms,
    const at::Tensor& value_norms,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& signs,
    const at::Tensor& centroids,
    int64_t bits,
    double scale,
    int64_t variant,
    int64_t codebook_mode)
{
    at::Tensor attn_out = at::empty(query.sizes(), query.options());
    float scale_real = static_cast<float>(scale);
    EXEC_NPU_CMD(aclnnTurboquantPagedAttentionV310,
                 query,
                 key_cache,
                 value_cache,
                 key_norms,
                 value_norms,
                 block_table,
                 seq_lens,
                 signs,
                 centroids,
                 bits,
                 scale_real,
                 variant,
                 codebook_mode,
                 attn_out);
    return attn_out;
}

}  // namespace vllm_ascend
#endif  // TURBOQUANT_PAGED_ATTENTION_V310_TORCH_ADPT_H
