/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_TORCH_ADPT_H
#define SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_TORCH_ADPT_H

namespace vllm_ascend {

inline bool is_sfa_offload_q_head_count_supported(int64_t q_head_count)
{
    return q_head_count == 1 || q_head_count == 2 || q_head_count == 4 ||
           q_head_count == 8 || q_head_count == 16 || q_head_count == 32 ||
           q_head_count == 64 || q_head_count == 128;
}

inline at::Tensor npu_sparse_flash_attention_for_offload(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const at::Tensor& tail_info,
    double scale_value,
    int64_t sparse_block_size,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    constexpr int64_t block_size = 128;
    constexpr int64_t sparse_capacity = 16384;
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 512,
                "query must have decode TND shape [B, N1, 512].");
    const int64_t batch_size = query.size(0);
    const int64_t q_head_count = query.size(1);
    TORCH_CHECK(is_sfa_offload_q_head_count_supported(q_head_count),
                "query N1 must be one of 1, 2, 4, 8, 16, 32, 64, 128, "
                "but got ", q_head_count, ".");
    TORCH_CHECK(key.dim() == 4 && key.size(1) == block_size &&
                    key.size(2) == 1 && key.size(3) == 512,
                "key must have PA_BSND shape [blocks, 128, 1, 512].");
    TORCH_CHECK(value.sizes() == key.sizes() && value.is_same(key),
                "MLA key and value must alias the same latent KV cache.");
    TORCH_CHECK(query_rope.dim() == 3 &&
                    query_rope.size(0) == batch_size &&
                    query_rope.size(1) == q_head_count &&
                    query_rope.size(2) == 64,
                "query_rope must have shape [B, N1, 64] with the same "
                "N1 as query.");
    TORCH_CHECK(key_rope.dim() == 4 && key_rope.size(0) == key.size(0) &&
                    key_rope.size(1) == block_size && key_rope.size(2) == 1 &&
                    key_rope.size(3) == 64,
                "key_rope must have shape [blocks, 128, 1, 64].");
    TORCH_CHECK(sparse_indices.dim() == 3 &&
                    sparse_indices.size(0) == batch_size &&
                    sparse_indices.size(1) == 1 &&
                    sparse_indices.size(2) == sparse_capacity,
                "sparse_indices must have shape [B, 1, 16384].");
    TORCH_CHECK(tail_info.dim() == 2 && tail_info.size(0) == batch_size &&
                    tail_info.size(1) == 2,
                "tail_info must have shape [B, 2].");
    TORCH_CHECK(block_table.dim() == 2 &&
                    block_table.size(0) == batch_size &&
                    block_table.size(1) > 0,
                "block_table must have shape [B, max_blocks].");
    TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                    actual_seq_lengths_query.size(0) == batch_size &&
                    actual_seq_lengths_kv.dim() == 1 &&
                    actual_seq_lengths_kv.size(0) == batch_size,
                "actual sequence lengths must have shape [B].");
    TORCH_CHECK(query.scalar_type() == at::kHalf ||
                    query.scalar_type() == at::kBFloat16,
                "SFA-Offload floating tensors must be fp16 or bf16.");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() &&
                    value.scalar_type() == query.scalar_type() &&
                    query_rope.scalar_type() == query.scalar_type() &&
                    key_rope.scalar_type() == query.scalar_type(),
                "SFA-Offload floating tensor dtypes must match.");
    TORCH_CHECK(sparse_indices.scalar_type() == at::kInt &&
                    tail_info.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt &&
                    actual_seq_lengths_query.scalar_type() == at::kInt &&
                    actual_seq_lengths_kv.scalar_type() == at::kInt,
                "SFA-Offload metadata tensors must be int32.");
    TORCH_CHECK(layout_query == "TND" && layout_kv == "PA_BSND",
                "SFA-Offload only supports TND/PA_BSND layouts.");
    TORCH_CHECK(sparse_block_size == 1,
                "SFA-Offload sparse_block_size must be 1.");
    TORCH_CHECK(sparse_mode == 3, "SFA-Offload sparse_mode must be 3.");

    at::Tensor output = at::empty(query.sizes(), query.options());
    std::string layout_query_string(layout_query);
    std::string layout_kv_string(layout_kv);
    char* layout_query_ptr = layout_query_string.data();
    char* layout_kv_ptr = layout_kv_string.data();
    EXEC_NPU_CMD(aclnnSparseFlashAttentionForOffload,
                 query, key, value, sparse_indices, tail_info, block_table,
                 actual_seq_lengths_query, actual_seq_lengths_kv, query_rope,
                 key_rope, scale_value, sparse_block_size, layout_query_ptr,
                 layout_kv_ptr, sparse_mode, output);
    return output;
}

}  // namespace vllm_ascend

#endif  // SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_TORCH_ADPT_H
