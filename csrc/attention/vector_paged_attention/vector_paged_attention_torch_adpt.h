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
#ifndef VECTOR_PAGED_ATTENTION_TORCH_ADPT_H
#define VECTOR_PAGED_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {
namespace vector_paged_attention_detail {

constexpr int64_t QUERY_DIM_NUM = 3;
constexpr int64_t KV_CACHE_DIM_NUM = 3;
constexpr int64_t KV_CACHE_WITH_HEAD_DIM_NUM = 4;
constexpr int64_t BLOCK_TABLE_DIM_NUM = 2;
constexpr int64_t SEQ_LENS_DIM_NUM = 1;

constexpr int64_t SUPPORTED_HEAD_DIM = 64;
constexpr int64_t MIN_BLOCK_SIZE = 8;
constexpr int64_t MAX_BLOCK_SIZE = 128;
constexpr int64_t MAX_KV_CAPACITY = 4096;
constexpr int64_t MAX_BATCH = 32;
// One AI vector core runs one (request, head), so the whole step must fit in
// the vector cores of a single die. Both SOCs this operator declares support
// for have 48; tiling re-checks against the platform value it is given.
constexpr int64_t MAX_TASKS = 48;

// Normalize [num_blocks, block_size, num_kv_heads, head_dim] to the
// [num_blocks, block_size, num_kv_heads * head_dim] the kernel indexes. Both
// spell the same memory, so this is a view and never a copy.
inline at::Tensor FlattenCache(const at::Tensor& cache)
{
    if (cache.dim() == KV_CACHE_WITH_HEAD_DIM_NUM) {
        return cache.view({cache.size(0), cache.size(1), cache.size(2) * cache.size(3)});
    }
    return cache;
}

inline void CheckVectorPagedAttentionParams(
    const at::Tensor& query, const at::Tensor& keyCache, const at::Tensor& valueCache,
    const at::Tensor& blockTable, const at::Tensor& seqLens, int64_t numKvHeads)
{
    TORCH_CHECK(query.dim() == QUERY_DIM_NUM,
                "query must be [batch, num_heads, head_dim]; got ", query.dim(), " dims");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16,
                "vector paged attention is bfloat16 only; query is ", query.scalar_type());
    TORCH_CHECK(keyCache.scalar_type() == at::kBFloat16 &&
                    valueCache.scalar_type() == at::kBFloat16,
                "key_cache and value_cache must be bfloat16");
    TORCH_CHECK(keyCache.sizes() == valueCache.sizes(),
                "key_cache and value_cache must have the same shape");
    TORCH_CHECK(keyCache.dim() == KV_CACHE_DIM_NUM || keyCache.dim() == KV_CACHE_WITH_HEAD_DIM_NUM,
                "key_cache must be [num_blocks, block_size, num_kv_heads*head_dim] or "
                "[num_blocks, block_size, num_kv_heads, head_dim]");
    TORCH_CHECK(keyCache.is_contiguous() && valueCache.is_contiguous(),
                "key_cache and value_cache must be contiguous");
    TORCH_CHECK(blockTable.dim() == BLOCK_TABLE_DIM_NUM && blockTable.scalar_type() == at::kInt,
                "block_table must be a 2D int32 tensor");
    TORCH_CHECK(seqLens.dim() == SEQ_LENS_DIM_NUM && seqLens.scalar_type() == at::kInt,
                "seq_lens must be a 1D int32 tensor");

    const int64_t batch = query.size(0);
    const int64_t numHeads = query.size(1);
    const int64_t headDim = query.size(2);
    const at::Tensor flatKey = FlattenCache(keyCache);
    const int64_t blockSize = flatKey.size(1);
    const int64_t kvStride = flatKey.size(2);
    const int64_t maxBlocks = blockTable.size(1);

    TORCH_CHECK(blockTable.size(0) == batch && seqLens.size(0) == batch,
                "block_table and seq_lens must have one row per request; got ",
                blockTable.size(0), " and ", seqLens.size(0), " for batch ", batch);
    TORCH_CHECK(headDim == SUPPORTED_HEAD_DIM,
                "vector paged attention supports head_dim ", SUPPORTED_HEAD_DIM,
                " only; got ", headDim);
    TORCH_CHECK(numKvHeads == numHeads,
                "vector paged attention is multi-head only: num_kv_heads (", numKvHeads,
                ") must equal num_heads (", numHeads, ")");
    TORCH_CHECK(kvStride == numKvHeads * headDim,
                "key_cache row must hold num_kv_heads*head_dim (", numKvHeads * headDim,
                ") elements; got ", kvStride);
    TORCH_CHECK(batch >= 1 && batch <= MAX_BATCH,
                "batch must be in [1, ", MAX_BATCH, "]; got ", batch);
    TORCH_CHECK(blockSize >= MIN_BLOCK_SIZE && blockSize <= MAX_BLOCK_SIZE &&
                    (blockSize & (blockSize - 1)) == 0,
                "block_size must be a power of two in [", MIN_BLOCK_SIZE, ", ", MAX_BLOCK_SIZE,
                "]; got ", blockSize);
    TORCH_CHECK(blockSize * maxBlocks <= MAX_KV_CAPACITY,
                "block_size*block_table.size(1) must not exceed ", MAX_KV_CAPACITY, "; got ",
                blockSize * maxBlocks);
    TORCH_CHECK(batch * numHeads <= MAX_TASKS,
                "batch*num_heads must not exceed the die's AI vector core count (", MAX_TASKS,
                "); got ", batch * numHeads);
}

}  // namespace vector_paged_attention_detail

// Single-query paged attention for small decode shapes, executed entirely on
// the AI vector cores. See csrc/attention/vector_paged_attention/README.md for
// the declared domain and when this is faster than the general fused operator.
at::Tensor npu_vector_paged_attention(
    const at::Tensor& query, const at::Tensor& keyCache, const at::Tensor& valueCache,
    const at::Tensor& blockTable, const at::Tensor& seqLens,
    int64_t numKvHeads, double scale)
{
    vector_paged_attention_detail::CheckVectorPagedAttentionParams(
        query, keyCache, valueCache, blockTable, seqLens, numKvHeads);

    const at::Tensor flatKey = vector_paged_attention_detail::FlattenCache(keyCache);
    const at::Tensor flatValue = vector_paged_attention_detail::FlattenCache(valueCache);
    const int64_t numHeads = query.size(1);
    at::Tensor attnOut = at::empty(query.sizes(), query.options());

    EXEC_NPU_CMD(aclnnVectorPagedAttention, query, flatKey, flatValue, blockTable, seqLens,
                 numHeads, numKvHeads, scale, attnOut);
    return attnOut;
}

}  // namespace vllm_ascend

#endif  // VECTOR_PAGED_ATTENTION_TORCH_ADPT_H
