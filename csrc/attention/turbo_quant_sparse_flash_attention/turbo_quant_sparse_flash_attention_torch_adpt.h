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

#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

#include <limits>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> turboquant_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const at::Tensor &block_table,
    const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &actual_seq_lengths_kv,
    double scale_value, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t sparse_block_size, c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t attention_mode,
    int64_t quant_scale_repo_mode, int64_t tile_size, int64_t rope_head_dim,
    bool return_softmax_lse)
{
    constexpr int64_t QUERY_DIM = 3;
    constexpr int64_t KV_DIM = 4;
    constexpr int64_t SPARSE_INDICES_DIM = 3;
    constexpr int64_t BLOCK_TABLE_DIM = 2;
    constexpr int64_t SEQ_LENGTHS_DIM = 1;
    constexpr int64_t TQ_QUERY_HEAD_DIM = 576;
    constexpr int64_t TQ_KV_HEADS = 1;
    constexpr int64_t TQ_KV_SLOT_BYTES = 386;
    constexpr int64_t TQ_MAX_QUERY_HEADS = 128;
    constexpr int64_t TQ_MAX_BLOCK_SIZE = 1024;
    constexpr int64_t TQ_BLOCK_ALIGNMENT = 16;
    constexpr int64_t TQ_MAX_SPARSE_BLOCK_SIZE = 16;
    constexpr int64_t TQ_ROPE_HEAD_DIM = 64;
    constexpr int64_t TQ_QUANT_MODE = 3;
    constexpr int64_t TQ_TILE_SIZE = 128;
    constexpr int64_t kUnboundedTokens = std::numeric_limits<int64_t>::max();

    std::string layout_query_str(layout_query);
    std::string layout_kv_str(layout_kv);
    TORCH_CHECK(layout_query_str == "TND", "TurboQuant query layout must be TND, but got ", layout_query_str);
    TORCH_CHECK(layout_kv_str == "PA_BSND", "TurboQuant KV layout must be PA_BSND, but got ", layout_kv_str);
    TORCH_CHECK(query.dim() == QUERY_DIM, "TurboQuant query must be rank 3, but got ", query.dim());
    TORCH_CHECK(key.dim() == KV_DIM, "TurboQuant key must be rank 4, but got ", key.dim());
    TORCH_CHECK(value.dim() == KV_DIM, "TurboQuant value must be rank 4, but got ", value.dim());
    TORCH_CHECK(sparse_indices.dim() == SPARSE_INDICES_DIM,
                "TurboQuant sparse_indices must be rank 3, but got ", sparse_indices.dim());
    TORCH_CHECK(block_table.dim() == BLOCK_TABLE_DIM,
                "TurboQuant block_table must be rank 2, but got ", block_table.dim());
    TORCH_CHECK(actual_seq_lengths_query.dim() == SEQ_LENGTHS_DIM,
                "TurboQuant actual_seq_lengths_query must be rank 1, but got ", actual_seq_lengths_query.dim());
    TORCH_CHECK(actual_seq_lengths_kv.dim() == SEQ_LENGTHS_DIM,
                "TurboQuant actual_seq_lengths_kv must be rank 1, but got ", actual_seq_lengths_kv.dim());

    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "TurboQuant query must have bfloat16 dtype");
    TORCH_CHECK(key.scalar_type() == at::kChar, "TurboQuant key must have int8 dtype");
    TORCH_CHECK(value.scalar_type() == at::kChar, "TurboQuant value must have int8 dtype");
    TORCH_CHECK(sparse_indices.scalar_type() == at::kInt, "TurboQuant sparse_indices must have int32 dtype");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "TurboQuant block_table must have int32 dtype");
    TORCH_CHECK(actual_seq_lengths_query.scalar_type() == at::kInt,
                "TurboQuant actual_seq_lengths_query must have int32 dtype");
    TORCH_CHECK(actual_seq_lengths_kv.scalar_type() == at::kInt,
                "TurboQuant actual_seq_lengths_kv must have int32 dtype");
    TORCH_CHECK(!key_dequant_scale.has_value() || key_dequant_scale->scalar_type() == at::kFloat,
                "TurboQuant key_dequant_scale must have float32 dtype");
    TORCH_CHECK(!value_dequant_scale.has_value() || value_dequant_scale->scalar_type() == at::kFloat,
                "TurboQuant value_dequant_scale must have float32 dtype");

    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0,
                "TurboQuant query token and head dimensions must be non-zero");
    TORCH_CHECK(query.size(1) <= TQ_MAX_QUERY_HEADS && (query.size(1) & (query.size(1) - 1)) == 0,
                "TurboQuant query head count must be a power of two no greater than ", TQ_MAX_QUERY_HEADS);
    TORCH_CHECK(query.size(2) == TQ_QUERY_HEAD_DIM,
                "TurboQuant query head dimension must be ", TQ_QUERY_HEAD_DIM, ", but got ", query.size(2));
    TORCH_CHECK(key.size(0) > 0 && key.size(1) > 0,
                "TurboQuant KV block count and block size must be non-zero");
    TORCH_CHECK(key.size(2) == TQ_KV_HEADS,
                "TurboQuant key must contain exactly one KV head, but got ", key.size(2));
    TORCH_CHECK(key.size(3) == TQ_KV_SLOT_BYTES,
                "TurboQuant KV slot width must be ", TQ_KV_SLOT_BYTES, ", but got ", key.size(3));
    TORCH_CHECK(key.sizes() == value.sizes(),
                "TurboQuant key and value shapes must match, but got ", key.sizes(), " and ", value.sizes());
    TORCH_CHECK(sparse_indices.size(0) == query.size(0) && sparse_indices.size(1) == key.size(2),
                "TurboQuant sparse_indices token/KV-head dimensions must match query and key");
    TORCH_CHECK(actual_seq_lengths_query.size(0) > 0,
                "TurboQuant actual_seq_lengths_query must be non-empty");
    TORCH_CHECK(actual_seq_lengths_kv.size(0) == actual_seq_lengths_query.size(0),
                "TurboQuant query and KV sequence-length tensors must have the same batch size");
    TORCH_CHECK(block_table.size(0) == actual_seq_lengths_query.size(0) && block_table.size(1) > 0,
                "TurboQuant block_table must match the sequence batch and contain at least one block column");
    TORCH_CHECK(query.size(1) % key.size(2) == 0,
                "TurboQuant query head count must be divisible by the KV head count");

    TORCH_CHECK(key_quant_mode == TQ_QUANT_MODE && value_quant_mode == TQ_QUANT_MODE,
                "TurboQuant key/value quant modes must both be ", TQ_QUANT_MODE);
    TORCH_CHECK(sparse_block_size > 0 && sparse_block_size <= TQ_MAX_SPARSE_BLOCK_SIZE &&
                    (sparse_block_size & (sparse_block_size - 1)) == 0,
                "TurboQuant sparse_block_size must be a power of two no greater than ",
                TQ_MAX_SPARSE_BLOCK_SIZE);
    TORCH_CHECK(key.size(1) <= TQ_MAX_BLOCK_SIZE && key.size(1) % TQ_BLOCK_ALIGNMENT == 0 &&
                    key.size(1) % sparse_block_size == 0,
                "TurboQuant KV block size must be 16-aligned, no greater than ", TQ_MAX_BLOCK_SIZE,
                ", and divisible by sparse_block_size");
    TORCH_CHECK(sparse_mode == 0 || sparse_mode == 3, "TurboQuant sparse_mode must be 0 or 3");
    TORCH_CHECK(attention_mode == 2, "TurboQuant attention_mode must be 2");
    TORCH_CHECK(quant_scale_repo_mode == 1, "TurboQuant quant_scale_repo_mode must be 1");
    TORCH_CHECK(tile_size == TQ_TILE_SIZE, "TurboQuant tile_size must be ", TQ_TILE_SIZE);
    TORCH_CHECK(rope_head_dim == TQ_ROPE_HEAD_DIM,
                "TurboQuant rope_head_dim must be ", TQ_ROPE_HEAD_DIM, ", but got ", rope_head_dim);

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    auto query_shape = query.sizes();
    at::Tensor output = at::empty(
        {query_shape[0], query_shape[1], query_shape[2] - rope_head_dim}, query.options());

    // TQ SFA is TND-layout only; mirror kv_quant's TND softmax_lse shape so the DCP
    // decode merge can consume softmax_max/softmax_sum. Empty {0} when not requested.
    at::SmallVector<int64_t, 8> softmax_size;
    if (return_softmax_lse) {
        softmax_size = {TQ_KV_HEADS, query.size(0), query.size(1) / TQ_KV_HEADS};
    } else {
        softmax_size = {0};
    }
    at::Tensor softmax_max = at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum = at::empty(softmax_size, query.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnTurboQuantSparseFlashAttention,
                 query,
                 key,
                 value,
                 sparse_indices,
                 key_dequant_scale,
                 value_dequant_scale,
                 block_table,
                 actual_seq_lengths_query,
                 actual_seq_lengths_kv,
                 scale_value,
                 key_quant_mode,
                 value_quant_mode,
                 sparse_block_size,
                 layout_query_ptr,
                 layout_kv_ptr,
                 sparse_mode,
                 kUnboundedTokens,
                 kUnboundedTokens,
                 attention_mode,
                 quant_scale_repo_mode,
                 tile_size,
                 rope_head_dim,
                 return_softmax_lse,
                 output,
                 softmax_max,
                 softmax_sum);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, softmax_max, softmax_sum);
}

} // namespace vllm_ascend
#endif
